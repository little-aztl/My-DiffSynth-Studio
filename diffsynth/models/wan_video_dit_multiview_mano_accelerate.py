"""Accelerated frozen-video path for multi-view MANO training.

The multi-view MANO stage freezes the complete video generator.  The legacy
``WanModel.forward`` nevertheless lets the trainable MANO branch enter the
video activation graph through a residual whose frozen gate is exactly zero.
That makes autograd retain and checkpoint the much larger video path even
though its gradient is mathematically zero.

This module installs a checkpoint-compatible execution path that:

* evaluates the frozen video path under ``torch.no_grad``;
* omits the zero-gated MANO-to-video attention direction;
* checkpoints only the trainable video-to-MANO path;
* skips the unused diffusion head during pose training;
* reuses the camera/RoPE/chunking optimizations from video-branch training.

Classes are changed in place and no parameter is copied or renamed, so old
checkpoints and optimizer parameter names remain compatible.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional, Sequence

import torch
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint

from .wan_video_dit import WanModel, modulate, sinusoidal_embedding_1d
from .wan_video_dit_train_accelerate import (
    AcceleratedVideoBranchDiTBlock,
    AcceleratedVideoBranchWanModel,
    _chunked_sequence_forward,
    _positive_chunk_size,
    _rope_apply_real_chunked,
)


class AcceleratedMultiviewManoDiTBlock(AcceleratedVideoBranchDiTBlock):
    """Split a DiT block into an inference-only video path and MANO path."""

    def forward_frozen_video(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        rotations: torch.Tensor,
        num_views: int,
        grid_size: tuple[int, int, int],
        t_mod_view: torch.Tensor,
        view_mask: torch.Tensor | None,
        plucker_fea: torch.Tensor,
        compute_video_ffn: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (next-block video, video features consumed by MANO)."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)

        input_x = modulate(self.layer_norm_bare(x), shift_msa, scale_msa)
        x = self.gate(
            x,
            gate_msa,
            self._video_self_attention(input_x, rotations),
        )
        x = x + self._video_cross_attention(
            self.norm3(x),
            context,
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
                expanded_view_mask,
                "b f v h w -> (b f) (v h w)",
            )

        shift_view, scale_view, gate_view = (
            self.view_modulation.to(
                dtype=t_mod_view.dtype,
                device=t_mod_view.device,
            )
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

        # MANO consumes the video state before the video's final FFN.  The two
        # branches are independent because gate4video_after_bicross is frozen
        # at zero, so computing the video FFN first is an equivalent reorder.
        x_for_mano = x
        if not compute_video_ffn:
            return x, x_for_mano
        input_x = modulate(self.layer_norm_bare(x), shift_mlp, scale_mlp)
        ffn_output = _chunked_sequence_forward(
            self.ffn,
            input_x,
            self._video_branch_ffn_chunk_size,
        )
        x = self.gate(x, gate_mlp, ffn_output)
        return x, x_for_mano

    def _self_attention_real_rope(
        self,
        attention,
        x: torch.Tensor,
        rotations: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = attention.norm_q(attention.q(x))
        k = attention.norm_k(attention.k(x))
        v = attention.v(x)
        if rotations is not None:
            q = _rope_apply_real_chunked(
                q,
                rotations,
                attention.num_heads,
                self._video_branch_rope_chunk_size,
                self._video_branch_rope_compute_dtype,
            )
            k = _rope_apply_real_chunked(
                k,
                rotations,
                attention.num_heads,
                self._video_branch_rope_chunk_size,
                self._video_branch_rope_compute_dtype,
            )
        if attention_mask is not None:
            # SDPA expects the key-validity mask to broadcast over heads and
            # query positions: (B, 1, 1, K).
            attention_mask = attention_mask[:, None, None, :]
        return attention.o(attention.attn(q, k, v, attn_mask=attention_mask))

    def _video_to_mano_attention(
        self,
        x_video: torch.Tensor,
        x_mano: torch.Tensor,
        video_rotations: torch.Tensor,
        mano_rotations: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate only the direction that can affect the training loss."""
        attention = self.bi_cross_attn
        k_video = attention.norm_k_video(attention.k_video(x_video))
        k_video = _rope_apply_real_chunked(
            k_video,
            video_rotations,
            attention.num_heads,
            self._video_branch_rope_chunk_size,
            self._video_branch_rope_compute_dtype,
        )
        v_video = attention.v_video(x_video)

        q_mano = attention.norm_q_mano(attention.q_mano(x_mano))
        q_mano = _rope_apply_real_chunked(
            q_mano,
            mano_rotations,
            attention.num_heads,
            self._video_branch_rope_chunk_size,
            self._video_branch_rope_compute_dtype,
        )
        output_mano = attention.attn(q_mano, k_video, v_video)
        return attention.o_mano(output_mano)

    def forward_mano(
        self,
        x_mano: torch.Tensor,
        x_video: torch.Tensor,
        t_mod_mano1: torch.Tensor,
        t_mod_mano2: torch.Tensor,
        video_rotations: torch.Tensor,
        mano_rotations: torch.Tensor,
        num_views: int,
        grid_size: tuple[int, int, int],
        view_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Update multi-view MANO tokens with frozen video features."""
        num_frames = grid_size[0]
        mano_tokens_per_frame = x_mano.shape[1] // num_frames
        (
            shift_mano_attn1,
            scale_mano_attn1,
            gate_mano_attn1,
            shift_mano_mlp,
            scale_mano_mlp,
            gate_mano_mlp,
        ) = (
            self.mano_modulation[:, :6].to(
                dtype=t_mod_mano1.dtype,
                device=t_mod_mano1.device,
            )
            + t_mod_mano1
        ).chunk(6, dim=1)
        shift_mano_attn2, scale_mano_attn2, gate_mano_attn2 = (
            self.mano_modulation[:, 6:].to(
                dtype=t_mod_mano2.dtype,
                device=t_mod_mano2.device,
            )
            + t_mod_mano2
        ).chunk(3, dim=1)

        input_mano = modulate(
            self.layer_norm_bare(x_mano),
            shift_mano_attn1,
            scale_mano_attn1,
        )
        x_mano = self.gate(
            x_mano,
            gate_mano_attn1,
            self._self_attention_real_rope(
                self.mano_inter_view_attention,
                input_mano,
                mano_rotations,
            ),
        )

        x_mano = rearrange(
            x_mano,
            "(b v) (f k) d -> (b f) (v k) d",
            v=num_views,
            f=num_frames,
            k=mano_tokens_per_frame,
        )
        mano_view_mask = None
        if view_mask is not None:
            mano_view_mask = repeat(
                view_mask,
                "b v -> (b f) (v k)",
                f=num_frames,
                k=mano_tokens_per_frame,
            )
        input_mano = modulate(
            self.layer_norm_bare(x_mano),
            shift_mano_attn2,
            scale_mano_attn2,
        )
        x_mano = self.gate(
            x_mano,
            gate_mano_attn2,
            self._self_attention_real_rope(
                self.mano_across_view_attention,
                input_mano,
                rotations=None,
                attention_mask=mano_view_mask,
            ),
        )
        x_mano = rearrange(
            x_mano,
            "(b f) (v k) d -> (b v) (f k) d",
            v=num_views,
            f=num_frames,
            k=mano_tokens_per_frame,
        )

        if hasattr(self, "bi_cross_attn"):
            output_mano = self._video_to_mano_attention(
                x_video=self.layer_norm_bare(x_video),
                x_mano=self.layer_norm_bare(x_mano),
                video_rotations=video_rotations,
                mano_rotations=mano_rotations,
            )
            x_mano = self.gate(
                x_mano,
                self.gate4mano_after_bicross,
                output_mano,
            )

        input_mano = modulate(
            self.layer_norm_bare(x_mano),
            shift_mano_mlp,
            scale_mano_mlp,
        )
        return self.gate(
            x_mano,
            gate_mano_mlp,
            self.ffn_mano(input_mano),
        )


def _call_frozen_video_block(
    block: AcceleratedMultiviewManoDiTBlock,
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    rotations: torch.Tensor,
    num_views: int,
    grid_size: tuple[int, int, int],
    t_mod_view: torch.Tensor,
    view_mask: torch.Tensor | None,
    plucker_fea: torch.Tensor,
    compute_video_ffn: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return block.forward_frozen_video(
        x,
        context,
        t_mod,
        rotations,
        num_views,
        grid_size,
        t_mod_view,
        view_mask,
        plucker_fea,
        compute_video_ffn,
    )


def _call_multiview_mano_block(
    block: AcceleratedMultiviewManoDiTBlock,
    x_mano: torch.Tensor,
    x_video: torch.Tensor,
    t_mod_mano1: torch.Tensor,
    t_mod_mano2: torch.Tensor,
    video_rotations: torch.Tensor,
    mano_rotations: torch.Tensor,
    num_views: int,
    grid_size: tuple[int, int, int],
    view_mask: torch.Tensor | None,
) -> torch.Tensor:
    return block.forward_mano(
        x_mano,
        x_video,
        t_mod_mano1,
        t_mod_mano2,
        video_rotations,
        mano_rotations,
        num_views,
        grid_size,
        view_mask,
    )


class AcceleratedMultiviewManoWanModel(AcceleratedVideoBranchWanModel):
    """Wan execution specialized for a frozen multi-view video generator."""

    def _rotations_for_grid(
        self,
        grid_size: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Return full-precision real RoPE after Lightning dtype casting.

        ``precision=bf16-true`` calls ``module.to(bfloat16)`` after model
        construction and therefore also casts registered floating buffers.
        Rebuild a cached rotation tensor from the original complex frequencies
        on first use instead of silently using quantized BF16 angles.
        """
        if grid_size == self._video_branch_cached_grid_size:
            cached = self._video_branch_cached_rotations
            if (
                cached.device != device
                or cached.dtype != self._video_branch_rope_compute_dtype
            ):
                cached = self._build_rotations(*grid_size, device=device)
                self._video_branch_cached_rotations = cached
            return cached
        return self._build_rotations(*grid_size, device=device)

    def _build_mano_tokens(
        self,
        x: torch.Tensor,
        camera_pose_encoding: torch.Tensor,
        *,
        num_views: int,
        num_frames: int,
        height: int,
        width: int,
        mano_tokens_per_frame: int,
    ) -> torch.Tensor:
        x_mano = rearrange(
            x,
            "(b v) (f h w) d -> (b v f) d h w",
            v=num_views,
            f=num_frames,
            h=height,
            w=width,
        )
        x_mano = self.copy_video_latent_and_subsample(x_mano)
        x_mano = rearrange(
            x_mano,
            "(b v f) (k d) 1 1 -> b v f k d",
            v=num_views,
            f=num_frames,
            k=mano_tokens_per_frame,
            d=self.dim,
        )
        camera_pose_encoding = camera_pose_encoding.to(
            device=x.device,
            dtype=x.dtype,
        )
        camera_token = self.mano_camera_token_encoder(camera_pose_encoding)
        camera_token = repeat(
            camera_token,
            "b v d -> b v f 1 d",
            f=num_frames,
        )
        x_mano = torch.cat([x_mano, camera_token], dim=3)
        return rearrange(x_mano, "b v f n d -> (b v) (f n) d")

    def _mano_rotations_for_frames(
        self,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        expected_frames = self._video_branch_cached_grid_size[0]
        if num_frames == expected_frames:
            cached = self._multiview_mano_cached_rotations
            if (
                cached.device == device
                and cached.dtype == self._video_branch_rope_compute_dtype
            ):
                return cached
        rotations = self.mano_f_freqs[:num_frames].repeat_interleave(
            self.mano_branch_tokens_per_frame,
            dim=0,
        )
        rotations = rotations.view(
            num_frames * self.mano_branch_tokens_per_frame,
            1,
            -1,
        )
        rotations = torch.view_as_real(rotations).to(
            device=device,
            dtype=self._video_branch_rope_compute_dtype,
        )
        if num_frames == expected_frames:
            self._multiview_mano_cached_rotations = rotations
        return rotations

    def _run_frozen_video_block(
        self,
        block: AcceleratedMultiviewManoDiTBlock,
        *inputs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return _call_frozen_video_block(block, *inputs)

    def _run_mano_block(
        self,
        block: AcceleratedMultiviewManoDiTBlock,
        *inputs,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ) -> torch.Tensor:
        if self.training and use_gradient_checkpointing:
            context = (
                torch.autograd.graph.save_on_cpu(
                    pin_memory=self._video_branch_checkpoint_offload_pin_memory
                )
                if use_gradient_checkpointing_offload
                else nullcontext()
            )
            with context:
                return checkpoint(
                    _call_multiview_mano_block,
                    block,
                    *inputs,
                    use_reentrant=False,
                )
        return _call_multiview_mano_block(block, *inputs)

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
        output_layers: List[int] | None = None,
        return_noise: bool = True,
        **kwargs,
    ):
        if self.has_image_input or clip_feature is not None or y is not None:
            raise ValueError(
                "Accelerated multi-view MANO execution supports Wan T2V only."
            )
        if hasattr(self, "ref_patch_embedding"):
            raise ValueError("Reference patches are incompatible with this path.")
        if output_layers is not None:
            raise ValueError("output_layers is not supported by this training path.")
        if num_views is None or int(num_views) <= 0:
            raise ValueError(f"num_views must be positive, got {num_views}")
        if plucker_fea is None:
            raise ValueError("plucker_fea is required for multi-view training.")
        if camera_pose_encoding is None:
            raise ValueError(
                "camera_pose_encoding is required for MANO attention."
            )
        del kwargs
        num_views = int(num_views)

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        x, raw_grid_size = self.patchify(x)
        num_frames, height, width = (int(value) for value in raw_grid_size)
        grid_size = (num_frames, height, width)
        if x.shape[0] % num_views != 0:
            raise ValueError(
                f"Flattened video batch {x.shape[0]} is not divisible by "
                f"V={num_views}."
            )
        batch_size = x.shape[0] // num_views
        # Text conditioning is identical across camera views in this pipeline.
        # Project B prompts once and only then make the inexpensive B*V view.
        # View-specific B*V contexts remain supported for inference/debugging.
        if context.shape[0] not in (batch_size, batch_size * num_views):
            raise ValueError(
                "context batch dimension must be B or B*V, got "
                f"{context.shape[0]} for B={batch_size}, V={num_views}."
            )
        if not hasattr(self, "replace_textemb") or not self.replace_textemb:
            context = self.text_embedding(context)
        if context.shape[0] == batch_size:
            context = repeat(context, "b n d -> (b v) n d", v=num_views)
        if view_mask is not None and view_mask.shape != (batch_size, num_views):
            raise ValueError(
                f"view_mask must have shape {(batch_size, num_views)}, "
                f"got {tuple(view_mask.shape)}."
            )
        if plucker_fea.shape[:2] != (batch_size, num_views):
            raise ValueError(
                f"plucker_fea must start with {(batch_size, num_views)}, "
                f"got {tuple(plucker_fea.shape[:2])}."
            )

        video_rotations = self._rotations_for_grid(grid_size, x.device)
        mano_rotations = self._mano_rotations_for_frames(num_frames, x.device)
        t_view = rearrange(t, "(b v) d -> b v d", v=num_views)[:, 0]
        t_view = repeat(t_view, "b d -> (b f) d", f=num_frames)
        t_mod_view = self.time_projection_view(t_view).unflatten(
            1, (3, self.dim)
        )

        t_mod_mano = self.time_projection_mano(t).unflatten(1, (9, self.dim))
        t_mod_mano1 = t_mod_mano[:, :6]
        t_mod_mano2 = rearrange(
            t_mod_mano[:, 6:],
            "(b v) n d -> b v n d",
            v=num_views,
        )[:, 0]
        t_mod_mano2 = repeat(
            t_mod_mano2,
            "b n d -> (b f) n d",
            f=num_frames,
        )

        mano_start_block_idx = self.mano_start_block_idx
        x_mano = None
        for block_id, block in enumerate(self.blocks):
            if block_id == mano_start_block_idx:
                x_mano = self._build_mano_tokens(
                    x=x,
                    camera_pose_encoding=camera_pose_encoding,
                    num_views=num_views,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    mano_tokens_per_frame=self.mano_tokens_per_frame,
                )

            x, x_for_mano = self._run_frozen_video_block(
                block,
                x,
                context,
                t_mod,
                video_rotations,
                num_views,
                grid_size,
                t_mod_view,
                view_mask,
                plucker_fea,
                not (
                    block_id == len(self.blocks) - 1 and not return_noise
                ),
            )
            if block_id < mano_start_block_idx:
                # No later consumer holds this pre-FFN video state. Drop the
                # ~685 MiB production tensor before starting the next block.
                del x_for_mano
                continue
            if x_mano is None:
                raise RuntimeError("MANO tokens were not initialized.")
            x_mano = self._run_mano_block(
                block,
                x_mano,
                x_for_mano,
                t_mod_mano1,
                t_mod_mano2,
                video_rotations,
                mano_rotations,
                num_views,
                grid_size,
                view_mask,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=(
                    use_gradient_checkpointing_offload
                ),
            )

        if x_mano is None:
            raise RuntimeError("The MANO branch did not run.")
        x_mano = rearrange(
            x_mano,
            "(b v) (f n) d -> b v f n d",
            v=num_views,
            f=num_frames,
            n=self.mano_branch_tokens_per_frame,
        )
        if not return_noise:
            return x_mano

        noise = self.head(x, t)
        noise = self.unpatchify(noise, grid_size)
        return noise, x_mano


def _assert_frozen(module: torch.nn.Module, name: str) -> None:
    trainable = [
        parameter_name
        for parameter_name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    if trainable:
        raise RuntimeError(
            f"{name} must be frozen for accelerated multi-view MANO "
            f"execution; trainable parameters: {trainable[:8]}"
        )


def _validate_multiview_mano_model(dit: WanModel) -> None:
    if dit.has_image_input:
        raise ValueError("The accelerated path requires a Wan T2V model.")
    for attribute in (
        "time_projection_view",
        "time_projection_mano",
        "mano_camera_token_encoder",
        "copy_video_latent_and_subsample",
        "mano_f_freqs",
        "mano_start_block_idx",
    ):
        if not hasattr(dit, attribute):
            raise ValueError(f"MANO model is missing {attribute}.")
    if hasattr(dit, "ref_patch_embedding"):
        raise ValueError("Reference attention is incompatible with this path.")

    _assert_frozen(dit.patch_embedding, "patch_embedding")
    _assert_frozen(dit.text_embedding, "text_embedding")
    _assert_frozen(dit.time_embedding, "time_embedding")
    _assert_frozen(dit.time_projection, "time_projection")
    _assert_frozen(dit.time_projection_view, "time_projection_view")
    _assert_frozen(dit.head, "head")

    for block_id, block in enumerate(dit.blocks):
        for name in (
            "self_attn",
            "cross_attn",
            "view_attn",
            "ffn",
        ):
            if not hasattr(block, name):
                raise ValueError(f"blocks.{block_id}.{name} is missing.")
            _assert_frozen(getattr(block, name), f"blocks.{block_id}.{name}")
        for name in ("modulation", "view_modulation"):
            parameter = getattr(block, name, None)
            if parameter is None:
                raise ValueError(f"blocks.{block_id}.{name} is missing.")
            if parameter.requires_grad:
                raise RuntimeError(f"blocks.{block_id}.{name} must be frozen.")

        if block_id < dit.mano_start_block_idx:
            continue
        for name in (
            "mano_inter_view_attention",
            "mano_across_view_attention",
            "mano_modulation",
            "ffn_mano",
            "bi_cross_attn",
            "gate4mano_after_bicross",
            "gate4video_after_bicross",
        ):
            if not hasattr(block, name):
                raise ValueError(f"blocks.{block_id}.{name} is missing.")
        video_gate = block.gate4video_after_bicross
        if (
            video_gate.requires_grad
            or torch.count_nonzero(video_gate.detach()).item() != 0
        ):
            raise RuntimeError(
                f"blocks.{block_id}.gate4video_after_bicross must be frozen "
                "at zero."
            )
        # The omitted MANO-to-video half must stay frozen. If a future training
        # policy opens it, failing here is safer than silently dropping grads.
        for name in (
            "q_video",
            "norm_q_video",
            "k_mano",
            "norm_k_mano",
            "v_mano",
            "o_video",
        ):
            _assert_frozen(
                getattr(block.bi_cross_attn, name),
                f"blocks.{block_id}.bi_cross_attn.{name}",
            )


def accelerate_multiview_mano_wan_model(
    dit: WanModel,
    *,
    expected_grid_size: Sequence[int] = (13, 30, 40),
    ffn_chunk_size: int | None = 0,
    camera_chunk_size: int | None = 0,
    rope_chunk_size: int | None = 0,
    rope_compute_dtype: str = "float64",
    checkpoint_offload_pin_memory: bool = False,
) -> AcceleratedMultiviewManoWanModel:
    """Install the optimized multi-view MANO execution path in place."""
    if isinstance(dit, AcceleratedMultiviewManoWanModel):
        return dit
    if not isinstance(dit, WanModel):
        raise TypeError(f"Expected WanModel, got {type(dit)!r}.")
    _validate_multiview_mano_model(dit)

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
    camera_chunk_size = _positive_chunk_size(
        camera_chunk_size, "camera_chunk_size"
    )
    rope_chunk_size = _positive_chunk_size(rope_chunk_size, "rope_chunk_size")

    for block in dit.blocks:
        block.__class__ = AcceleratedMultiviewManoDiTBlock
        block._video_branch_ffn_chunk_size = ffn_chunk_size
        block._video_branch_camera_chunk_size = camera_chunk_size
        block._video_branch_rope_chunk_size = rope_chunk_size
        block._video_branch_rope_compute_dtype = compute_dtype

    dit.__class__ = AcceleratedMultiviewManoWanModel
    dit._video_branch_cached_grid_size = grid_size
    dit._video_branch_rope_compute_dtype = compute_dtype
    dit._video_branch_checkpoint_offload_pin_memory = bool(
        checkpoint_offload_pin_memory
    )

    parameter = next(dit.parameters())
    video_rotations = dit._build_rotations(*grid_size, device=parameter.device)
    dit.register_buffer(
        "_video_branch_cached_rotations",
        video_rotations,
        persistent=False,
    )
    mano_rotations = dit.mano_f_freqs[: grid_size[0]].repeat_interleave(
        dit.mano_branch_tokens_per_frame,
        dim=0,
    )
    mano_rotations = mano_rotations.view(
        grid_size[0] * dit.mano_branch_tokens_per_frame,
        1,
        -1,
    )
    mano_rotations = torch.view_as_real(mano_rotations).to(
        device=parameter.device,
        dtype=compute_dtype,
    )
    dit.register_buffer(
        "_multiview_mano_cached_rotations",
        mano_rotations,
        persistent=False,
    )

    return dit
