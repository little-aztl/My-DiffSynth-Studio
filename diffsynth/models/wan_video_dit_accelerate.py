"""Accelerated frozen-video Wan DiT path for monocular MANO training.

This module deliberately leaves :mod:`wan_video_dit` untouched.  It reuses the
same parameters and state-dict layout, but specializes execution for the
MVHandGen monocular setup:

* there is exactly one valid view;
* the video backbone is frozen;
* the MANO-to-video residual gate is frozen at exactly zero;
* across-view attention is therefore an identity operation;
* the diffusion noise prediction is optional.

``accelerate_monocular_wan_model`` changes the Python classes of an already
constructed Wan model and its blocks in place.  No parameter is copied or
renamed, which keeps both memory use and checkpoint compatibility unchanged.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional

import torch
from einops import rearrange, repeat

from .wan_video_dit import (
    DiTBlock,
    WanModel,
    modulate,
    rope_apply,
    sinusoidal_embedding_1d,
)


class AcceleratedMonocularDiTBlock(DiTBlock):
    """DiT block with a detached frozen-video path and a trainable MANO path."""

    @torch.no_grad()
    def forward_video_pre_mano(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        plucker_fea: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run frozen video attention up to the MANO injection point."""
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)

        input_x = modulate(self.layer_norm_bare(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(
            self.norm3(x),
            context,
            plucker_fea=plucker_fea,
            # The monocular pipeline contains one real view and no padding.
            view_mask=None,
        )

        if hasattr(self, "ref_attn"):
            raise RuntimeError(
                "AcceleratedMonocularDiTBlock does not support reference attention."
            )

        # No fake view-attention branch is run here.  In the legacy monocular
        # path it only rearranged x, materialized zeros, and added a zero residual.
        return x, shift_mlp, scale_mlp, gate_mlp

    @torch.no_grad()
    def forward_video_post_mano(
        self,
        x: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
    ) -> torch.Tensor:
        """Finish the frozen video block and return a detached feature tensor."""
        input_x = modulate(self.layer_norm_bare(x), shift_mlp, scale_mlp)
        return self.gate(x, gate_mlp, self.ffn(input_x))

    def forward_video_only(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        plucker_fea: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run a fully frozen block before the MANO branch starts."""
        x, shift_mlp, scale_mlp, gate_mlp = self.forward_video_pre_mano(
            x=x,
            context=context,
            t_mod=t_mod,
            freqs=freqs,
            plucker_fea=plucker_fea,
        )
        return self.forward_video_post_mano(
            x=x,
            shift_mlp=shift_mlp,
            scale_mlp=scale_mlp,
            gate_mlp=gate_mlp,
        )

    def _video_to_mano_attention(
        self,
        x_video: torch.Tensor,
        x_mano: torch.Tensor,
        freqs_video: torch.Tensor,
        freqs_mano: torch.Tensor,
    ) -> torch.Tensor:
        """Compute only the video-key/value -> MANO-query attention direction."""
        bi_cross_attn = self.bi_cross_attn

        k_video = bi_cross_attn.norm_k_video(bi_cross_attn.k_video(x_video))
        k_video = rope_apply(k_video, freqs_video, bi_cross_attn.num_heads)
        v_video = bi_cross_attn.v_video(x_video)

        q_mano = bi_cross_attn.norm_q_mano(bi_cross_attn.q_mano(x_mano))
        q_mano = rope_apply(q_mano, freqs_mano, bi_cross_attn.num_heads)

        output_mano = bi_cross_attn.attn(q_mano, k_video, v_video)
        return bi_cross_attn.o_mano(output_mano)

    def forward_mano(
        self,
        x_mano: torch.Tensor,
        x_video: torch.Tensor,
        t_mod_mano: torch.Tensor,
        freqs_video: torch.Tensor,
        freqs_mano: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """Update MANO tokens while treating video features as constants."""
        (
            shift_mano_attn,
            scale_mano_attn,
            gate_mano_attn,
            shift_mano_mlp,
            scale_mano_mlp,
            gate_mano_mlp,
        ) = (
            self.mano_modulation[:, :6].to(
                dtype=t_mod_mano.dtype,
                device=t_mod_mano.device,
            )
            + t_mod_mano
        ).chunk(6, dim=1)

        input_mano = modulate(
            self.layer_norm_bare(x_mano),
            shift_mano_attn,
            scale_mano_attn,
        )
        x_mano = self.gate(
            x_mano,
            gate_mano_attn,
            self.mano_inter_view_attention(input_mano, freqs=freqs_mano),
        )

        # V=1 across-view attention in the legacy path was a parameter-free zero
        # residual.  Skipping it also removes two rearranges and mask expansion.
        if x_mano.shape[1] % num_frames != 0:
            raise ValueError(
                "MANO token count must be divisible by the latent frame count, "
                f"got tokens={x_mano.shape[1]}, frames={num_frames}."
            )

        if hasattr(self, "bi_cross_attn"):
            output_mano = self._video_to_mano_attention(
                x_video=self.layer_norm_bare(x_video),
                x_mano=self.layer_norm_bare(x_mano),
                freqs_video=freqs_video,
                freqs_mano=freqs_mano,
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


class AcceleratedMonocularWanModel(WanModel):
    """WanModel execution specialized for frozen-DiT monocular MANO training."""

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
            "(B V) (f h w) D -> (B V f) D h w",
            V=num_views,
            f=num_frames,
            h=height,
            w=width,
        )
        x_mano = self.copy_video_latent_and_subsample(x_mano)
        x_mano = rearrange(
            x_mano,
            "(B V f) (k D) 1 1 -> B V f k D",
            V=num_views,
            f=num_frames,
            k=mano_tokens_per_frame,
            D=self.dim,
        )

        camera_pose_encoding = camera_pose_encoding.to(
            device=x.device,
            dtype=x.dtype,
        )
        camera_token = self.mano_camera_token_encoder(camera_pose_encoding)
        camera_token = repeat(
            camera_token,
            "B V D -> B V f 1 D",
            f=num_frames,
        )
        x_mano = torch.cat([x_mano, camera_token], dim=3)
        return rearrange(x_mano, "B V f n D -> (B V) (f n) D")

    @staticmethod
    def _checkpoint_mano(
        block: AcceleratedMonocularDiTBlock,
        x_mano: torch.Tensor,
        x_video: torch.Tensor,
        t_mod_mano: torch.Tensor,
        freqs_video: torch.Tensor,
        freqs_mano: torch.Tensor,
        num_frames: int,
        *,
        offload: bool,
    ) -> torch.Tensor:
        context = torch.autograd.graph.save_on_cpu() if offload else nullcontext()
        with context:
            return torch.utils.checkpoint.checkpoint(
                block.forward_mano,
                x_mano,
                x_video,
                t_mod_mano,
                freqs_video,
                freqs_mano,
                num_frames,
                use_reentrant=False,
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
        output_layers: List[int] | None = None,
        return_noise: bool = True,
        **kwargs,
    ):
        """Run the accelerated path, optionally omitting the diffusion head."""
        if self.has_image_input or clip_feature is not None or y is not None:
            raise ValueError(
                "AcceleratedMonocularWanModel supports the Wan T2V backbone only."
            )
        if hasattr(self, "ref_patch_embedding"):
            raise ValueError(
                "AcceleratedMonocularWanModel does not support reference patches."
            )
        if output_layers is not None:
            raise ValueError(
                "AcceleratedMonocularWanModel does not support output_layers."
            )
        if num_views != 1:
            raise ValueError(
                "AcceleratedMonocularWanModel requires num_views=1, "
                f"got {num_views}."
            )
        if camera_pose_encoding is None:
            raise ValueError(
                "camera_pose_encoding is required for monocular MANO attention."
            )

        # ``view_mask`` is intentionally unused: this specialized path contains
        # one real view, so the only valid mask is an all-true (B, 1) tensor.
        del view_mask, kwargs

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        if not hasattr(self, "replace_textemb") or not self.replace_textemb:
            context = self.text_embedding(context)

        x, (num_frames, height, width) = self.patchify(x)
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
        ).reshape(num_frames * height * width, 1, -1).to(x.device)

        if not hasattr(self, "time_projection_mano"):
            raise RuntimeError("MANO attention modules are not installed.")
        # Compute only the six modulation rows used by MANO temporal attention
        # and its MLP.  Slicing the existing Linear weights preserves checkpoint
        # keys while avoiding the final three rows for removed across-view work.
        mano_time_hidden = self.time_projection_mano[0](t)
        mano_time_linear = self.time_projection_mano[1]
        mano_time_rows = 6 * self.dim
        t_mod_mano = torch.nn.functional.linear(
            mano_time_hidden,
            mano_time_linear.weight[:mano_time_rows],
            None
            if mano_time_linear.bias is None
            else mano_time_linear.bias[:mano_time_rows],
        ).unflatten(1, (6, self.dim))

        mano_tokens_per_frame = self.mano_tokens_per_frame
        mano_branch_tokens_per_frame = self.mano_branch_tokens_per_frame
        mano_freqs = (
            self.mano_f_freqs[:num_frames]
            .repeat_interleave(mano_branch_tokens_per_frame, dim=0)
            .view(num_frames * mano_branch_tokens_per_frame, 1, -1)
            .to(x.device)
        )
        mano_start_block_idx = self.mano_start_block_idx

        x_mano = None
        for block_id, block in enumerate(self.blocks):
            if block_id < mano_start_block_idx:
                x = block.forward_video_only(
                    x=x,
                    context=context,
                    t_mod=t_mod,
                    freqs=freqs,
                    plucker_fea=plucker_fea,
                )
                continue

            if block_id == mano_start_block_idx:
                x_mano = self._build_mano_tokens(
                    x=x,
                    camera_pose_encoding=camera_pose_encoding,
                    num_views=num_views,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    mano_tokens_per_frame=mano_tokens_per_frame,
                )

            x, shift_mlp, scale_mlp, gate_mlp = block.forward_video_pre_mano(
                x=x,
                context=context,
                t_mod=t_mod,
                freqs=freqs,
                plucker_fea=plucker_fea,
            )
            if self.training and use_gradient_checkpointing:
                x_mano = self._checkpoint_mano(
                    block=block,
                    x_mano=x_mano,
                    x_video=x,
                    t_mod_mano=t_mod_mano,
                    freqs_video=freqs,
                    freqs_mano=mano_freqs,
                    num_frames=num_frames,
                    offload=use_gradient_checkpointing_offload,
                )
            else:
                x_mano = block.forward_mano(
                    x_mano=x_mano,
                    x_video=x,
                    t_mod_mano=t_mod_mano,
                    freqs_video=freqs,
                    freqs_mano=mano_freqs,
                    num_frames=num_frames,
                )
            x = block.forward_video_post_mano(
                x=x,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
            )

        if x_mano is None:
            raise RuntimeError("The MANO branch did not run.")
        x_mano = rearrange(
            x_mano,
            "(B V) (f n) D -> B V f n D",
            V=num_views,
            f=num_frames,
            n=mano_branch_tokens_per_frame,
        )

        if not return_noise:
            return x_mano

        noise = self.head(x, t)
        noise = self.unpatchify(noise, (num_frames, height, width))
        return noise, x_mano


def _assert_frozen(module: torch.nn.Module, name: str) -> None:
    trainable = [
        parameter_name
        for parameter_name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    if trainable:
        raise RuntimeError(
            f"{name} must be frozen for accelerated monocular execution; "
            f"trainable parameters: {trainable[:8]}"
        )


def accelerate_monocular_wan_model(
    dit: WanModel,
    *,
    mano_start_block_idx: int,
) -> AcceleratedMonocularWanModel:
    """Enable the accelerated execution path on an initialized frozen Wan model."""
    if isinstance(dit, AcceleratedMonocularWanModel):
        return dit
    if not isinstance(dit, WanModel):
        raise TypeError(f"Expected WanModel, got {type(dit)!r}.")
    if dit.has_image_input:
        raise ValueError(
            "The accelerated monocular path requires has_image_input=False."
        )
    if dit.mano_start_block_idx != mano_start_block_idx:
        raise ValueError(
            "MANO start block mismatch: "
            f"model={dit.mano_start_block_idx}, requested={mano_start_block_idx}."
        )

    _assert_frozen(dit.patch_embedding, "patch_embedding")
    _assert_frozen(dit.text_embedding, "text_embedding")
    _assert_frozen(dit.time_embedding, "time_embedding")
    _assert_frozen(dit.time_projection, "time_projection")
    _assert_frozen(dit.head, "head")

    for block_id, block in enumerate(dit.blocks):
        _assert_frozen(block.self_attn, f"blocks.{block_id}.self_attn")
        _assert_frozen(block.cross_attn, f"blocks.{block_id}.cross_attn")
        _assert_frozen(block.ffn, f"blocks.{block_id}.ffn")
        if block.modulation.requires_grad:
            raise RuntimeError(f"blocks.{block_id}.modulation must be frozen.")
        if hasattr(block, "ref_attn"):
            raise RuntimeError("Reference attention is incompatible with this path.")

        if block_id >= mano_start_block_idx:
            video_gate = block.gate4video_after_bicross
            if (
                video_gate.requires_grad
                or torch.count_nonzero(video_gate.detach()).item() != 0
            ):
                raise RuntimeError(
                    f"blocks.{block_id}.gate4video_after_bicross must be frozen at zero."
                )

        # Class reassignment retains the existing module registry and parameters.
        block.__class__ = AcceleratedMonocularDiTBlock

    # Compatibility placeholders used while the legacy checkpoint was loaded
    # are not part of accelerated execution and contain no checkpoint state.
    for block in dit.blocks:
        for attribute in ("view_attn", "view_norm", "view_modulation"):
            if hasattr(block, attribute):
                delattr(block, attribute)
    for block in dit.blocks[mano_start_block_idx:]:
        if hasattr(block, "mano_across_view_attention"):
            delattr(block, "mano_across_view_attention")
    if hasattr(dit, "time_projection_view"):
        delattr(dit, "time_projection_view")

    dit.__class__ = AcceleratedMonocularWanModel
    return dit
