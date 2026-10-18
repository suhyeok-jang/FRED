"""Wan2.1-I2V-14B-480P (diffusers) feature extractor — ``backbone_type=wan21_14b``.

Subclasses ``Wan22FeatureExtractor`` so the WHOLE recipe is inherited unchanged (forward (a) σ=1.0 extraction,
forward (b) bell flow matching, the self-EMA teacher-forced target, SelfDistill capture, text cache, EMA/compile/
early-exit machinery) and overrides only the model layer (``_load_model_layer``) and the per-forward conditioning
kwargs (``_condition_kwargs``).

Model layer differences vs Wan2.2-TI2V-5B:
  • transformer: 40 blocks, dim 5120, in_channels 36 = 16 (noisy latent) ∥ 4 (mask) ∥ 16 (cond latent), out 16;
    a CLIP image feature enters through the img cross-attention (``encoder_hidden_states_image``).
  • timestep is SCALAR per sample — Wan2.1 has no per-token AdaLN. The recipe's per-token maps {cond=0,
    future=σ} therefore collapse to σ: the history is pinned by (i) the clean cond latents in the x stream
    (``cond_mask`` pin, inherited) and (ii) the y = mask4∥cond16 channels, exactly as diffusers' I2V/FLF2V
    pipelines mark observed frames. The adapter validates that every (B,S) map it receives has that
    {0, σ} structure before reducing it.
  • VAE: AutoencoderKLWan z_dim 16, spatial 8×, temporal 4× (5B: 48ch / 16×) — all geometry is derived.
  • y (per forward, same tensor for the live copy AND the EMA teacher): mask4 = diffusers mask fold (cond pixel
    frames = 1, future = 0; first frame repeated vae_t_scale×, then (T_lat, vae_t_scale) view) — its slot pattern must equal the
    inherited ``cond_indicator`` (raise otherwise); cond16 = VAE encode of [cond frames, zero-padded future]
    (NOT the full-window GT latents the training forward hoists: those would leak the future into y).
  • CLIP conditioning: the CURRENT frame (last of the cond stack; ``config.wan21_clip_cond_frame="first"`` for frame 0),
    CLIPImageProcessor → CLIPVisionModelWithProjection ``hidden_states[-2]`` (diffusers ``encode_image`` parity),
    frozen, no_grad.
Memory: ``config.video_ema_sharded`` (bf16 EMA copy + per-rank fp32 master) and gradient checkpointing.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from fred.model.modules.wan import Wan21DiTAdapter, WanVAEEncoder
from fred.utils.model import module_dtype
from fred.utils.dist import is_global_zero

from .wan22_extractor import Wan22FeatureExtractor


class Wan21I2V14BFeatureExtractor(Wan22FeatureExtractor):
    """Wan2.1-I2V-14B-480P extractor: 2.1 model layer, inherited WAM recipe (see module docstring)."""

    _BLOCK_L2_REF = {0: None, 19: None, 39: None}   # no baselined constants for 2.1 → finiteness sanity only

    def _load_model_layer(self, repo: str, load_dev: str) -> None:
        from diffusers import AutoencoderKLWan, WanTransformer3DModel  # noqa: PLC0415
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection  # noqa: PLC0415

        assert self.config.wan21_clip_cond_frame in ("current", "first"), f"wan21_clip_cond_frame must be current|first, got {self.config.wan21_clip_cond_frame!r}"
        self._clip_frame_mode = self.config.wan21_clip_cond_frame

        dit = WanTransformer3DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=self.torch_dtype)
        dc = dit.config
        assert int(dc.in_channels) == 36 and int(dc.out_channels) == 16 and len(dit.blocks) == 40, f"wan21_14b: transformer config mismatch — in={dc.in_channels} out={dc.out_channels} blocks={len(dit.blocks)}"
        assert int(dc.image_dim) == 1280 and int(dc.added_kv_proj_dim) == 5120, f"wan21_14b: img cross-attn config mismatch — image_dim={dc.image_dim} added_kv_proj_dim={dc.added_kv_proj_dim}"
        vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=self.torch_dtype)
        dit = dit.to(load_dev)
        self._assert_load_fingerprint(dit, self._BLOCK_L2_REF)
        self.video_dit = Wan21DiTAdapter(dit).to(load_dev)
        self.vae = vae.to(load_dev)
        self.vae_codec = WanVAEEncoder(self.vae, encode_chunk=self.config.vae_encode_chunk, fallback_dtype=self.torch_dtype)
        assert self.vae_codec.latent_channels == 16 and self.vae_codec.scale_factor_temporal == 4 and self.vae_codec.scale_factor_spatial == 8, (
            f"wan21_14b: VAE geometry mismatch — z_dim={self.vae_codec.latent_channels} temporal={self.vae_codec.scale_factor_temporal} spatial={self.vae_codec.scale_factor_spatial}"
        )
        if is_global_zero():
            print(f"[wan21-14b] transformer loaded: 40 blocks, dim {self.video_dit.hidden_dim}, in=36ch (16 latent ∥ 4 mask ∥ 16 cond), out=16ch, "
                  f"img cross-attn 1280→5120, patch {self.video_dit.patch_size} ('{repo}'); tap L{self.extract_layer}", flush=True)

        # CLIP image conditioning (frozen; per-forward encode of the current frame)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(repo, subfolder="image_encoder", torch_dtype=self.torch_dtype).to(load_dev)
        self.image_encoder.eval()
        self.image_encoder.requires_grad_(False)
        self.image_processor = CLIPImageProcessor.from_pretrained(repo, subfolder="image_processor")

    def _condition_kwargs(
        self, *, videos_bcthw: torch.Tensor, cond_latents: torch.Tensor, cond_indicator: torch.Tensor,
        num_frames_out: int, gt_latents: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        clip_feat = self._encode_clip_condition(videos_bcthw)
        y = self._cond_y(videos_bcthw=videos_bcthw, cond_latents=cond_latents, cond_indicator=cond_indicator, num_frames_out=num_frames_out, gt_latents=gt_latents)
        return {"y": y, "clip_feat": clip_feat}

    def _cond_y(self, *, videos_bcthw: torch.Tensor, cond_latents: torch.Tensor, cond_indicator: torch.Tensor, num_frames_out: int, gt_latents: Optional[torch.Tensor]) -> torch.Tensor:
        """y = mask4 ∥ cond16 (B, 20, T_lat, H_lat, W_lat), diffusers I2V/FLF2V parity (see the module docstring)."""
        B = int(cond_latents.shape[0])
        vae_t_scale = int(self.vae_codec.scale_factor_temporal)
        nf = int(num_frames_out)
        T_lat, H_lat, W_lat = (int(s) for s in cond_latents.shape[2:])
        assert (nf - 1) % vae_t_scale == 0 and (nf + vae_t_scale - 1) // vae_t_scale == T_lat, f"wan21_14b: mask fold geometry mismatch — num_frames_out={nf}, vae_t_scale={vae_t_scale}, T_lat={T_lat} (need nf = 4k+1 and (nf+3)//4 == T_lat)"
        n_cond_px = int(videos_bcthw.shape[2])
        mask_px = torch.ones(B, 1, nf, H_lat, W_lat, dtype=cond_latents.dtype, device=cond_latents.device)
        mask_px[:, :, n_cond_px:] = 0.0
        first = mask_px[:, :, 0:1].repeat_interleave(vae_t_scale, dim=2)
        mask_seq = torch.cat([first, mask_px[:, :, 1:]], dim=2)                                      # (B,1,nf+3,H,W)
        mask4 = mask_seq.view(B, T_lat, vae_t_scale, H_lat, W_lat).transpose(1, 2).contiguous()             # (B,4,T_lat,H,W)
        slot_full = (mask4.amin(dim=1)[:, :, 0, 0] > 0.5)
        ind = (cond_indicator.reshape(B, T_lat) > 0.5)
        assert bool(torch.equal(slot_full, ind)), (
            f"wan21_14b: history mask fold {slot_full[0].int().tolist()} != cond_indicator {ind[0].int().tolist()} "
            f"(num_frames_in={n_cond_px}, vae_t_scale={vae_t_scale}) — the cond pixel-frame count is not on a causal-VAE slot boundary (1+4k)"
        )
        # cond16 = encode([cond frames, zero-padded future]) — NEVER the hoisted full-window GT latents (future leak into y).
        # Without a future clip the inherited cond_latents already are that encode.
        if gt_latents is None:
            cond16 = cond_latents
        else:
            cond16 = self._encode_cond_padded(videos_bcthw, nf).to(device=cond_latents.device, dtype=cond_latents.dtype)
            assert tuple(cond16.shape) == tuple(cond_latents.shape), f"wan21_14b: zero-padded cond encode {tuple(cond16.shape)} != latent grid {tuple(cond_latents.shape)}"
        return torch.cat([mask4, cond16.to(mask4.dtype)], dim=1).detach()                             # (B,20,T_lat,H,W)

    def _encode_clip_condition(self, videos_bcthw: torch.Tensor) -> torch.Tensor:
        """(B, 257, 1280) penultimate CLIP hidden state of the conditioning frame (current by default)."""
        fi = 0 if self._clip_frame_mode == "first" else int(videos_bcthw.shape[2]) - 1
        frame = videos_bcthw[:, :, fi]
        imgs01 = ((frame.detach().float() + 1.0) / 2.0).clamp(0.0, 1.0)
        inputs = self.image_processor(images=[im for im in imgs01.cpu().numpy()], do_rescale=False, return_tensors="pt")
        pv = inputs["pixel_values"].to(device=next(self.image_encoder.parameters()).device, dtype=module_dtype(self.image_encoder, fallback=self.torch_dtype))
        with torch.no_grad():
            hs = self.image_encoder(pixel_values=pv, output_hidden_states=True)
        return hs.hidden_states[-2].detach()
