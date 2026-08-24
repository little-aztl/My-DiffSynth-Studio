"""Training acceleration for the MVHandGen Wan video branch.

This module keeps :mod:`wan_video_dit` as the checkpoint-compatible source of
parameters, while replacing its Python execution path with one specialized for
the trainable MVHandGen video branch:

* T2V only (no image, reference, or MANO branches);
* camera features are projected before their temporal broadcast;
* token-wise FFNs, camera modulation, and RoPE are evaluated in chunks;
* the fixed training RoPE grid is cached as a non-persistent buffer;
* the repeated DiT block can optionally be compiled only while training.

``accelerate_video_branch_wan_model`` changes the classes of an initialized
Wan model and its blocks in place.  It does not copy or rename parameters, so
existing optimizer construction and state-dict keys remain unchanged.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, List, Optional, Sequence

import torch
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint

from .wan_video_dit import DiTBlock, WanModel, modulate, sinusoidal_embedding_1d


def _positive_chunk_size(value: int | None, name: str) -> int:
    if value is None:
        return 0
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _chunked_sequence_forward(
    module: torch.nn.Module,
    x: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Apply a token-wise module without materializing its full hidden width."""
    if chunk_size <= 0 or x.shape[1] <= chunk_size:
        return module(x)
    return torch.cat(
        [
            module(x[:, start : start + chunk_size])
            for start in range(0, x.shape[1], chunk_size)
        ],
        dim=1,
    )


def _rope_apply_real_chunked(
    x: torch.Tensor,
    rotations: torch.Tensor,
    num_heads: int,
    chunk_size: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply RoPE using the real form of complex multiplication.

    The original implementation temporarily converts the complete Q/K tensor
    to float64 complex values.  This function retains that arithmetic by
    default but bounds the temporary tensor to ``chunk_size`` sequence tokens.
    """
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    if chunk_size <= 0 or x.shape[1] <= chunk_size:
        return _rope_rotate_real(x, rotations, compute_dtype)
    return torch.cat(
        [
            _rope_rotate_real(
                x[:, start : start + chunk_size],
                rotations[start : start + chunk_size],
                compute_dtype,
            )
            for start in range(0, x.shape[1], chunk_size)
        ],
        dim=1,
    )


def _rope_rotate_real(
    part: torch.Tensor,
    rotation: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Tensor-only RoPE kernel used by both eager and regional compile."""
    original_dtype = part.dtype
    pairs = part.to(compute_dtype).reshape(
        part.shape[0], part.shape[1], part.shape[2], -1, 2
    )
    rotation = rotation.to(device=part.device, dtype=compute_dtype)
    real = pairs[..., 0] * rotation[..., 0] - pairs[..., 1] * rotation[..., 1]
    imag = pairs[..., 1] * rotation[..., 0] + pairs[..., 0] * rotation[..., 1]
    return torch.stack((real, imag), dim=-1).flatten(2).to(original_dtype)


class AcceleratedVideoBranchDiTBlock(DiTBlock):
    """Trainable Wan block specialized for video and camera conditioning."""

    _video_branch_ffn_chunk_size: int
    _video_branch_camera_chunk_size: int
    _video_branch_rope_chunk_size: int
    _video_branch_rope_compute_dtype: torch.dtype

    def _video_self_attention(
        self,
        x: torch.Tensor,
        rotations: torch.Tensor,
    ) -> torch.Tensor:
        self_attn = self.self_attn
        q = self_attn.norm_q(self_attn.q(x))
        k = self_attn.norm_k(self_attn.k(x))
        v = self_attn.v(x)
        q = _rope_apply_real_chunked(
            q,
            rotations,
            self_attn.num_heads,
            self._video_branch_rope_chunk_size,
            self._video_branch_rope_compute_dtype,
        )
        k = _rope_apply_real_chunked(
            k,
            rotations,
            self_attn.num_heads,
            self._video_branch_rope_chunk_size,
            self._video_branch_rope_compute_dtype,
        )
        return self_attn.o(self_attn.attn(q, k, v))

    def _apply_camera_shift(
        self,
        attention_output: torch.Tensor,
        plucker_fea: torch.Tensor,
        *,
        num_views: int,
        grid_size: tuple[int, int, int],
        view_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute camera modulation without repeating camera projection in time."""
        num_frames, height, width = grid_size
        spatial_tokens = height * width
        if plucker_fea.ndim != 4:
            raise ValueError(
                "plucker_fea must have shape (B, V, N, D), got "
                f"{tuple(plucker_fea.shape)}"
            )
        if plucker_fea.shape[1] != num_views:
            raise ValueError(
                f"plucker_fea has V={plucker_fea.shape[1]}, expected {num_views}"
            )
        if plucker_fea.shape[2] != spatial_tokens:
            raise ValueError(
                "Camera token count must equal the video spatial token count, "
                f"got {plucker_fea.shape[2]} and {spatial_tokens}"
            )

        batch_size = plucker_fea.shape[0]
        cross_attn = self.cross_attn
        # Linear(repeat(camera, F)) == repeat(Linear(camera), F).  Projecting
        # first avoids F identical 2048x2048 projections per Wan block.
        projected_camera = cross_attn.plucker_linear_encoder(plucker_fea)
        attention_output = rearrange(
            attention_output,
            "(b v) (f n) d -> b v f n d",
            b=batch_size,
            v=num_views,
            f=num_frames,
            n=spatial_tokens,
        )

        chunk_size = self._video_branch_camera_chunk_size
        frames_per_chunk = (
            num_frames
            if chunk_size <= 0
            else max(1, chunk_size // spatial_tokens)
        )
        outputs = []
        valid = None
        if view_mask is not None:
            valid = view_mask[:, :, None, None, None].to(
                dtype=attention_output.dtype
            )
        for start in range(0, num_frames, frames_per_chunk):
            latent_chunk = attention_output[:, :, start : start + frames_per_chunk]
            combined = cross_attn.latent_linear_encoder(latent_chunk)
            combined = combined + projected_camera[:, :, None]
            shift = cross_attn.camera_modulate(combined)
            if valid is not None:
                shift = shift * valid
            # Fuse the residual addition into the per-frame chunks. Keeping a
            # complete shift tensor until after concatenation costs another
            # full (B*V,FHW,D) activation at the block peak.
            outputs.append(latent_chunk + shift)

        attention_output = (
            outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)
        )
        return rearrange(attention_output, "b v f n d -> (b v) (f n) d")

    def _video_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_attention_bias: torch.Tensor | None,
        plucker_fea: torch.Tensor,
        *,
        num_views: int,
        grid_size: tuple[int, int, int],
        view_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cross_attn = self.cross_attn
        q = cross_attn.norm_q(cross_attn.q(x))
        k = cross_attn.norm_k(cross_attn.k(context))
        v = cross_attn.v(context)
        # Every view in one clip uses the same text condition.  Keeping the
        # context at clip batch size lets the expensive 4096/5120-wide K/V
        # projections run once per clip; repeat only their much smaller
        # per-head outputs for SDPA.  A B*V context remains supported for
        # callers that intentionally provide different text per view.
        if k.shape[0] != q.shape[0]:
            expected_query_batch = k.shape[0] * num_views
            if q.shape[0] != expected_query_batch:
                raise ValueError(
                    "Cross-attention context batch must be B or B*V; got "
                    f"Q batch {q.shape[0]}, K/V batch {k.shape[0]}, V={num_views}"
                )
            k = repeat(k, "b s d -> (b v) s d", v=num_views)
            v = repeat(v, "b s d -> (b v) s d", v=num_views)
        attention_mask = None
        if context_attention_bias is not None:
            if (
                context_attention_bias.ndim != 2
                or context_attention_bias.shape[1] != k.shape[1]
            ):
                raise ValueError(
                    "context_attention_bias must have shape (B, S) or "
                    f"(B*V, S); got {tuple(context_attention_bias.shape)} "
                    f"for context length {k.shape[1]}"
                )
            if context_attention_bias.shape[0] != q.shape[0]:
                expected_query_batch = context_attention_bias.shape[0] * num_views
                if q.shape[0] != expected_query_batch:
                    raise ValueError(
                        "context_attention_bias batch must be B or B*V; got "
                        f"Q batch {q.shape[0]}, bias batch "
                        f"{context_attention_bias.shape[0]}, V={num_views}"
                    )
                context_attention_bias = repeat(
                    context_attention_bias, "b s -> (b v) s", v=num_views
                )
            attention_mask = context_attention_bias[:, None, None, :].to(
                dtype=q.dtype, device=q.device
            )
        attention_output = cross_attn.attn(q, k, v, attn_mask=attention_mask)
        attention_output = self._apply_camera_shift(
            attention_output,
            plucker_fea,
            num_views=num_views,
            grid_size=grid_size,
            view_mask=view_mask,
        )
        return cross_attn.o(attention_output)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_attention_bias: torch.Tensor | None,
        t_mod: torch.Tensor,
        rotations: torch.Tensor,
        num_views: int,
        grid_size: tuple[int, int, int],
        t_mod_view: torch.Tensor,
        view_mask: torch.Tensor | None,
        plucker_fea: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)

        input_x = modulate(self.layer_norm_bare(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self._video_self_attention(input_x, rotations))
        x = x + self._video_cross_attention(
            self.norm3(x),
            context,
            context_attention_bias,
            plucker_fea,
            num_views=num_views,
            grid_size=grid_size,
            view_mask=view_mask,
        )

        num_frames, height, width = grid_size
        x = rearrange(
            x,
            "(b v) (f h w) d -> (b f) (v h w) d",
            v=num_views,
            f=num_frames,
            h=height,
            w=width,
        )
        expanded_view_mask = None
        if view_mask is not None:
            expanded_view_mask = repeat(
                view_mask,
                "b v -> b f v h w",
                f=num_frames,
                h=height,
                w=width,
            )
            expanded_view_mask = rearrange(
                expanded_view_mask, "b f v h w -> (b f) (v h w)"
            )

        shift_view, scale_view, gate_view = (
            self.view_modulation.to(dtype=t_mod_view.dtype, device=t_mod_view.device)
            + t_mod_view
        ).chunk(3, dim=1)
        input_x = modulate(self.view_norm(x), shift_view, scale_view)
        x = self.gate(
            x,
            gate_view,
            self.view_attn(input_x, attn_mask=expanded_view_mask),
        )
        x = rearrange(
            x,
            "(b f) (v h w) d -> (b v) (f h w) d",
            f=num_frames,
            h=height,
            w=width,
            v=num_views,
        )

        input_x = modulate(self.layer_norm_bare(x), shift_mlp, scale_mlp)
        ffn_output = _chunked_sequence_forward(
            self.ffn,
            input_x,
            self._video_branch_ffn_chunk_size,
        )
        return self.gate(x, gate_mlp, ffn_output)


def _call_video_branch_block(
    block: AcceleratedVideoBranchDiTBlock,
    x: torch.Tensor,
    context: torch.Tensor,
    context_attention_bias: torch.Tensor | None,
    t_mod: torch.Tensor,
    rotations: torch.Tensor,
    num_views: int,
    grid_size: tuple[int, int, int],
    t_mod_view: torch.Tensor,
    view_mask: torch.Tensor | None,
    plucker_fea: torch.Tensor,
) -> torch.Tensor:
    """One shared callable lets all identical blocks reuse compiled code."""
    return block(
        x,
        context,
        context_attention_bias,
        t_mod,
        rotations,
        num_views,
        grid_size,
        t_mod_view,
        view_mask,
        plucker_fea,
    )


class AcceleratedVideoBranchWanModel(WanModel):
    """WanModel execution specialized for trainable MVHandGen video output."""

    _video_branch_cached_grid_size: tuple[int, int, int]
    _video_branch_compiled_block: Callable | None
    _video_branch_compiled_signatures: set[tuple]
    _video_branch_checkpoint_group_size: int

    def _build_rotations(
        self,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        freqs = torch.cat(
            [
                self.freqs[0][:num_frames]
                .view(num_frames, 1, 1, -1)
                .expand(num_frames, height, width, -1),
                self.freqs[1][:height]
                .view(1, height, 1, -1)
                .expand(num_frames, height, width, -1),
                self.freqs[2][:width]
                .view(1, 1, width, -1)
                .expand(num_frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(num_frames * height * width, 1, -1)
        return torch.view_as_real(freqs).to(
            device=device,
            dtype=self._video_branch_rope_compute_dtype,
        )

    def _rotations_for_grid(
        self,
        grid_size: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if grid_size == self._video_branch_cached_grid_size:
            return self._video_branch_cached_rotations
        return self._build_rotations(*grid_size, device=device)

    def _run_block(
        self,
        block: AcceleratedVideoBranchDiTBlock,
        *inputs,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> torch.Tensor:
        block_call = (
            self._video_branch_compiled_block
            if self.training and self._video_branch_compiled_block is not None
            else _call_video_branch_block
        )
        if self.training and use_gradient_checkpointing:
            if self._video_branch_requires_compile_warmup:
                # The first call for a new B/V/mask signature must finish AOT
                # compilation before entering checkpoint's saved-tensor hooks.
                # Otherwise the first forward and its recomputation can expose
                # different internal saved-tensor sets. A single discarded
                # warm-up block makes all 30 repeated blocks share the stable
                # compiled graph; it does not change model state or gradients.
                signature = self._compile_signature(*inputs)
                if signature not in self._video_branch_compiled_signatures:
                    warmup_output = block_call(block, *inputs)
                    del warmup_output
                    self._video_branch_compiled_signatures.add(signature)
            context = (
                torch.autograd.graph.save_on_cpu(
                    pin_memory=self._video_branch_checkpoint_offload_pin_memory
                )
                if use_gradient_checkpointing_offload
                else nullcontext()
            )
            with context:
                return checkpoint(
                    block_call,
                    block,
                    *inputs,
                    use_reentrant=self._video_branch_checkpoint_use_reentrant,
                )
        return block_call(block, *inputs)

    def _run_block_group(
        self,
        blocks: Sequence[AcceleratedVideoBranchDiTBlock],
        *inputs,
        use_gradient_checkpointing_offload: bool,
    ) -> torch.Tensor:
        """Checkpoint several consecutive blocks as one recompute segment.

        Reentrant checkpointing saves every tensor argument at each checkpoint
        boundary.  At production resolution the hidden-state boundary alone is
        close to a GiB for a two-clip, three-view batch.  One boundary per
        group instead of one per block therefore removes most of the resident
        checkpoint inputs without changing the executed block graph.
        """
        if not blocks:
            raise ValueError("A checkpoint block group must not be empty")
        block_call = (
            self._video_branch_compiled_block
            if self.training and self._video_branch_compiled_block is not None
            else _call_video_branch_block
        )
        if self._video_branch_requires_compile_warmup:
            signature = self._compile_signature(*inputs)
            if signature not in self._video_branch_compiled_signatures:
                warmup_output = block_call(blocks[0], *inputs)
                del warmup_output
                self._video_branch_compiled_signatures.add(signature)

        def group_call(*group_inputs):
            hidden = group_inputs[0]
            shared_inputs = group_inputs[1:]
            for block in blocks:
                hidden = block_call(block, hidden, *shared_inputs)
            return hidden

        context = (
            torch.autograd.graph.save_on_cpu(
                pin_memory=self._video_branch_checkpoint_offload_pin_memory
            )
            if use_gradient_checkpointing_offload
            else nullcontext()
        )
        with context:
            return checkpoint(
                group_call,
                *inputs,
                use_reentrant=self._video_branch_checkpoint_use_reentrant,
            )

    @staticmethod
    def _compile_signature(
        x: torch.Tensor,
        context: torch.Tensor,
        context_attention_bias: torch.Tensor | None,
        t_mod: torch.Tensor,
        rotations: torch.Tensor,
        num_views: int,
        grid_size: tuple[int, int, int],
        t_mod_view: torch.Tensor,
        view_mask: torch.Tensor | None,
        plucker_fea: torch.Tensor,
    ) -> tuple:
        del t_mod, rotations, t_mod_view
        return (
            tuple(x.shape),
            tuple(context.shape),
            (
                None
                if context_attention_bias is None
                else tuple(context_attention_bias.shape)
            ),
            int(num_views),
            tuple(grid_size),
            None if view_mask is None else tuple(view_mask.shape),
            tuple(plucker_fea.shape),
            x.dtype,
            x.device,
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        plucker_fea: torch.Tensor | None = None,
        camera_pose_encoding: torch.Tensor | None = None,
        num_views: int | None = None,
        view_mask: torch.Tensor | None = None,
        context_attention_bias: torch.Tensor | None = None,
        output_layers: List[int] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.has_image_input or clip_feature is not None or y is not None:
            raise ValueError("Accelerated video-branch execution supports Wan T2V only.")
        if hasattr(self, "ref_patch_embedding"):
            raise ValueError("Reference patches are incompatible with this path.")
        if hasattr(self, "time_projection_mano"):
            raise ValueError("MANO attention is incompatible with this path.")
        if output_layers is not None:
            raise ValueError("output_layers is not supported by this training path.")
        if num_views is None or int(num_views) <= 0:
            raise ValueError(f"num_views must be positive, got {num_views}")
        if plucker_fea is None:
            raise ValueError("plucker_fea is required for video-branch training.")
        del camera_pose_encoding, kwargs
        num_views = int(num_views)

        x, grid_size = self.patchify(x)
        num_frames, height, width = (int(value) for value in grid_size)
        grid_size = (num_frames, height, width)
        if x.shape[0] % num_views != 0:
            raise ValueError(
                f"Flattened video batch {x.shape[0]} is not divisible by V={num_views}"
            )
        batch_size = x.shape[0] // num_views
        flattened_batch_size = batch_size * num_views
        if timestep.ndim != 1 or timestep.shape[0] not in {
            batch_size,
            flattened_batch_size,
        }:
            raise ValueError(
                "timestep must have shape (B,) or (B*V,), got "
                f"{tuple(timestep.shape)} for B={batch_size}, V={num_views}"
            )
        if context.ndim != 3 or context.shape[0] not in {
            batch_size,
            flattened_batch_size,
        }:
            raise ValueError(
                "context must have shape (B, S, D) or (B*V, S, D), got "
                f"{tuple(context.shape)} for B={batch_size}, V={num_views}"
            )
        if context_attention_bias is not None and (
            context_attention_bias.ndim != 2
            or context_attention_bias.shape[0]
            not in {batch_size, flattened_batch_size}
            or context_attention_bias.shape[1] != context.shape[1]
        ):
            raise ValueError(
                "context_attention_bias must have shape (B, S) or (B*V, S); "
                f"got {tuple(context_attention_bias.shape)} for B={batch_size}, "
                f"V={num_views}, S={context.shape[1]}"
            )
        if view_mask is not None and view_mask.shape != (batch_size, num_views):
            raise ValueError(
                f"view_mask must have shape {(batch_size, num_views)}, "
                f"got {tuple(view_mask.shape)}"
            )
        if plucker_fea.shape[:2] != (batch_size, num_views):
            raise ValueError(
                f"plucker_fea must start with {(batch_size, num_views)}, "
                f"got {tuple(plucker_fea.shape[:2])}"
            )

        t_condition = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep)
        )
        if timestep.shape[0] == batch_size:
            t_view = t_condition
            t = repeat(t_condition, "b d -> (b v) d", v=num_views)
            t_mod = self.time_projection(t_condition).unflatten(
                1, (6, self.dim)
            )
            t_mod = repeat(t_mod, "b n d -> (b v) n d", v=num_views)
        else:
            t = t_condition
            t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
            t_view = rearrange(t, "(b v) d -> b v d", v=num_views)[:, 0]

        if not hasattr(self, "replace_textemb") or not self.replace_textemb:
            context = self.text_embedding(context)

        rotations = self._rotations_for_grid(grid_size, x.device)
        t_view = repeat(t_view, "b d -> (b f) d", f=num_frames)
        t_mod_view = self.time_projection_view(t_view).unflatten(1, (3, self.dim))

        shared_block_inputs = (
            context,
            context_attention_bias,
            t_mod,
            rotations,
            num_views,
            grid_size,
            t_mod_view,
            view_mask,
            plucker_fea,
        )
        group_size = self._video_branch_checkpoint_group_size
        if self.training and use_gradient_checkpointing and group_size > 1:
            for start in range(0, len(self.blocks), group_size):
                x = self._run_block_group(
                    tuple(self.blocks[start : start + group_size]),
                    x,
                    *shared_block_inputs,
                    use_gradient_checkpointing_offload=(
                        use_gradient_checkpointing_offload
                    ),
                )
        else:
            for block in self.blocks:
                x = self._run_block(
                    block,
                    x,
                    *shared_block_inputs,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=(
                        use_gradient_checkpointing_offload
                    ),
                )

        x = self.head(x, t)
        return self.unpatchify(x, grid_size)


def _validate_video_branch_model(dit: WanModel) -> None:
    if dit.has_image_input:
        raise ValueError("The accelerated video branch requires a Wan T2V model.")
    if hasattr(dit, "ref_patch_embedding") or hasattr(dit, "time_projection_mano"):
        raise ValueError("Reference and MANO branches are not supported.")
    if not hasattr(dit, "time_projection_view"):
        raise ValueError("View attention has not been installed on the Wan model.")
    for block_id, block in enumerate(dit.blocks):
        if not hasattr(block, "view_attn"):
            raise ValueError(f"blocks.{block_id}.view_attn is missing")
        cross_attn = block.cross_attn
        for name in (
            "plucker_linear_encoder",
            "latent_linear_encoder",
            "camera_modulate",
        ):
            if not hasattr(cross_attn, name):
                raise ValueError(f"blocks.{block_id}.cross_attn.{name} is missing")


def accelerate_video_branch_wan_model(
    dit: WanModel,
    *,
    expected_grid_size: Sequence[int] = (13, 30, 40),
    ffn_chunk_size: int | None = 3600,
    camera_chunk_size: int | None = 3600,
    rope_chunk_size: int | None = 3600,
    rope_compute_dtype: str = "float64",
    checkpoint_offload_pin_memory: bool = False,
    checkpoint_use_reentrant: bool = False,
    checkpoint_block_group_size: int = 1,
    compile_enabled: bool = False,
    compile_backend: str = "inductor",
    compile_mode: str = "default",
    compile_fullgraph: bool = True,
    compile_dynamic: bool = False,
) -> AcceleratedVideoBranchWanModel:
    """Install the optimized video-branch execution path in place."""
    if isinstance(dit, AcceleratedVideoBranchWanModel):
        return dit
    if not isinstance(dit, WanModel):
        raise TypeError(f"Expected WanModel, got {type(dit)!r}")
    _validate_video_branch_model(dit)

    grid_size = tuple(int(value) for value in expected_grid_size)
    if len(grid_size) != 3 or any(value <= 0 for value in grid_size):
        raise ValueError(
            "expected_grid_size must contain three positive values: "
            f"{grid_size}"
        )
    dtype_by_name = {"float64": torch.float64, "float32": torch.float32}
    if rope_compute_dtype not in dtype_by_name:
        raise ValueError(
            "rope_compute_dtype must be 'float64' or 'float32', got "
            f"{rope_compute_dtype!r}"
        )
    compute_dtype = dtype_by_name[rope_compute_dtype]

    ffn_chunk_size = _positive_chunk_size(ffn_chunk_size, "ffn_chunk_size")
    camera_chunk_size = _positive_chunk_size(camera_chunk_size, "camera_chunk_size")
    rope_chunk_size = _positive_chunk_size(rope_chunk_size, "rope_chunk_size")
    checkpoint_block_group_size = int(checkpoint_block_group_size)
    if checkpoint_block_group_size <= 0:
        raise ValueError(
            "checkpoint_block_group_size must be positive, got "
            f"{checkpoint_block_group_size}"
        )
    for block in dit.blocks:
        block.__class__ = AcceleratedVideoBranchDiTBlock
        block._video_branch_ffn_chunk_size = ffn_chunk_size
        block._video_branch_camera_chunk_size = camera_chunk_size
        block._video_branch_rope_chunk_size = rope_chunk_size
        block._video_branch_rope_compute_dtype = compute_dtype

    dit.__class__ = AcceleratedVideoBranchWanModel
    dit._video_branch_cached_grid_size = grid_size
    dit._video_branch_rope_compute_dtype = compute_dtype
    dit._video_branch_checkpoint_offload_pin_memory = bool(
        checkpoint_offload_pin_memory
    )
    dit._video_branch_checkpoint_use_reentrant = bool(checkpoint_use_reentrant)
    dit._video_branch_checkpoint_group_size = checkpoint_block_group_size
    dit._video_branch_compact_conditioning = True
    dit._video_branch_exact_text_padding_compression = True

    parameter = next(dit.parameters())
    rotations = dit._build_rotations(*grid_size, device=parameter.device)
    dit.register_buffer(
        "_video_branch_cached_rotations",
        rotations,
        persistent=False,
    )

    compiled_block = None
    if compile_enabled:
        compiled_block = torch.compile(
            _call_video_branch_block,
            backend=compile_backend,
            mode=compile_mode,
            fullgraph=bool(compile_fullgraph),
            dynamic=bool(compile_dynamic),
        )
    # Avoid registering a compiler wrapper as an nn.Module child.
    object.__setattr__(dit, "_video_branch_compiled_block", compiled_block)
    dit._video_branch_requires_compile_warmup = bool(compile_enabled)
    object.__setattr__(
        dit,
        "_video_branch_compiled_signatures",
        set(),
    )
    return dit
