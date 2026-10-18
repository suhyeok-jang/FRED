"""Wan building blocks (diffusers Wan2.2-TI2V-5B / Wan2.1-I2V-14B): DiT adapters behind one call signature,
umT5 text conditioning (encoder/tokenizer adapters, the precomputed embedding cache, the loader shared with
scripts/precompute_text_embeds.py) and the VAE encoder (pixels → normalized latents)."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from fred.utils.model import module_dtype


def token_grid(x: torch.Tensor, patch_size: Tuple[int, int, int]) -> Tuple[int, int]:
    """(tokens per latent frame, sequence length) of the frame-major Wan patchify of ``x`` (B,C,T,H,W)."""
    _B, _C, T, H, W = x.shape
    pt, ph, pw = patch_size
    assert not (T % pt or H % ph or W % pw), f"latent grid {(T, H, W)} not divisible by patch {patch_size}"
    tpf = (H // ph) * (W // pw)
    return tpf, (T // pt) * tpf


class WanDiTAdapter(nn.Module):
    """Wan2.2-TI2V-5B: ``timestep`` is the recipe's per-token ``[B, seq_len]`` map (the caller owns the cond-token
    zeros); the diffusers TI2V path consumes ``timestep.ndim == 2`` natively (per-token AdaLN). Attention is full
    bidirectional; velocity target = ε − x0."""

    def __init__(self, dit: nn.Module):
        super().__init__()
        self.dit = dit
        cfg = dit.config
        self.patch_size = tuple(int(p) for p in cfg.patch_size)                        # (1, 2, 2)
        self.latent_channels = int(cfg.in_channels)                                             # 48
        self.hidden_dim = int(cfg.num_attention_heads) * int(cfg.attention_head_dim)   # 3072

    @property
    def blocks(self) -> nn.ModuleList:
        return self.dit.blocks

    def forward(self, *, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        _tpf, seq = token_grid(x, self.patch_size)
        assert tuple(timestep.shape) == (int(x.shape[0]), seq), (
            f"per-token `timestep` must be (B, seq)=({int(x.shape[0])}, {seq}) for latent grid {tuple(x.shape[2:])} / patch "
            f"{self.patch_size}, got {tuple(timestep.shape)}"
        )
        ts2d = timestep.to(dtype=torch.float32)
        return self.dit(hidden_states=x, timestep=ts2d, encoder_hidden_states=context, return_dict=False)[0]   # [B, 48, T, H, W] velocity


class Wan21DiTAdapter(nn.Module):
    """Wan2.1-I2V-14B: ``latent_channels`` reports 16 = the latent channels the recipe produces/consumes; the physical
    36ch input is assembled here from x(16) ∥ y(mask4 ∥ cond16). The recipe's per-token (B, S) ``timestep`` map is
    reduced to one scalar per sample after checking it is {0 on the cond tokens, one value on the rest}."""

    def __init__(self, dit: nn.Module):
        super().__init__()
        self.dit = dit
        cfg = dit.config
        self.patch_size = tuple(int(p) for p in cfg.patch_size)                        # (1, 2, 2)
        self.latent_dim = int(cfg.out_channels)                                        # 16
        self.latent_channels = self.latent_dim
        self.full_in_dim = int(cfg.in_channels)                                        # 36
        self.hidden_dim = int(cfg.num_attention_heads) * int(cfg.attention_head_dim)   # 5120
        assert self.full_in_dim == 2 * self.latent_dim + 4, f"Wan2.1-I2V channel layout unexpected: in_channels={self.full_in_dim} != 2*{self.latent_dim}+4 (noisy∥mask∥cond)"

    @property
    def blocks(self) -> nn.ModuleList:
        return self.dit.blocks

    def _scalar_timestep(self, x: torch.Tensor, timestep: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B = int(x.shape[0])
        assert timestep.ndim == 2 and int(timestep.shape[0]) == B, f"timestep must be (B, S), got {tuple(timestep.shape)}"
        ts = timestep.to(torch.float32)
        tpf, seq = token_grid(x, self.patch_size)
        assert int(ts.shape[1]) == seq, f"per-token timestep (B,{int(ts.shape[1])}) does not match the latent grid {tuple(x.shape[2:])} / patch {self.patch_size}"
        # cond slots = slots whose 4 mask channels are all 1 (same pattern the extractor validated against cond_indicator)
        cond_slots = (y[:, :4].amin(dim=1)[:, :, 0, 0] > 0.5)                       # (B, T_lat)
        n_cond_tok = int(cond_slots[0].sum().item()) * tpf
        sig = ts[:, n_cond_tok:].amax(dim=1) if n_cond_tok < int(ts.shape[1]) else ts.amax(dim=1)
        assert n_cond_tok == 0 or float(ts[:, :n_cond_tok].abs().max().item()) == 0.0, "wan21_14b: per-token map has non-zero timesteps on the cond tokens — not the history-pinned recipe"
        assert n_cond_tok >= int(ts.shape[1]) or bool(torch.allclose(ts[:, n_cond_tok:], sig.view(B, 1).expand_as(ts[:, n_cond_tok:]), atol=1e-3)), (
            "wan21_14b: the future tokens carry more than one timestep per sample (patch-mix) — Wan2.1 takes a scalar t"
        )
        return sig

    def forward(self, *, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, y: torch.Tensor, clip_feat: torch.Tensor) -> torch.Tensor:
        """``y``: (B, 20, T_lat, H_lat, W_lat) = mask4 ∥ cond16; ``clip_feat``: (B, 257, 1280) CLIP image feature."""
        B = int(x.shape[0])
        assert int(y.shape[0]) == B and int(clip_feat.shape[0]) == B, f"wan21_14b: conditioning batch mismatch — x B={B}, y B={int(y.shape[0])}, clip B={int(clip_feat.shape[0])}"
        assert tuple(y.shape[2:]) == tuple(x.shape[2:]) and int(y.shape[1]) == self.full_in_dim - self.latent_dim, f"wan21_14b: y {tuple(y.shape)} incompatible with x {tuple(x.shape)}"
        assert int(x.shape[1]) == self.latent_dim, f"wan21_14b: x must be the {self.latent_dim}ch latent stream, got {int(x.shape[1])}ch"
        t = self._scalar_timestep(x, timestep, y)
        x_in = torch.cat([x, y.to(device=x.device, dtype=x.dtype)], dim=1)          # [noisy(16) ∥ mask(4) ∥ cond(16)]
        return self.dit(
            hidden_states=x_in, timestep=t, encoder_hidden_states=context,
            encoder_hidden_states_image=clip_feat.to(device=x.device, dtype=x.dtype), return_dict=False,
        )[0]   # (B, 16, T, H, W) velocity, latent channels only


# ───────────────────────── text ─────────────────────────
class TextEmbedCache:
    """Precomputed umT5 embeddings keyed by the normalized prompt (payload of precompute_text_embeds.py:
    {"embeddings": {prompt: [L, 4096]}, "masks": {prompt: [L]}, "seq_len": 512}). Trimmed entries are zero-padded to
    ``seq_len`` on lookup. ``require_hit``: a miss is an error (the encoder is never needed)."""

    def __init__(self, path: Optional[str], require_hit: bool = True):
        self.path = path
        self.require_hit = bool(require_hit)
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.masks: Optional[Dict[str, torch.Tensor]] = None
        self.pad_to: Optional[int] = None
        if not path:
            return
        assert os.path.isfile(path), f"text_embed_cache='{path}' does not exist."
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        assert isinstance(payload, dict) and "embeddings" in payload, f"text_embed_cache='{path}' is not an embeds payload (missing 'embeddings')"
        self.embeddings = payload["embeddings"]
        self.masks = payload.get("masks", None)
        self.pad_to = int(payload.get("seq_len") or 0) or None

    def __len__(self) -> int:
        return len(self.embeddings)

    def _pad(self, t: torch.Tensor) -> torch.Tensor:
        if self.pad_to and int(t.shape[0]) < self.pad_to:
            return torch.cat([t, t.new_zeros((self.pad_to - int(t.shape[0]),) + tuple(t.shape[1:]))], dim=0)
        return t

    def lookup(self, prompts: List[str], *, device: torch.device, dtype: torch.dtype) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """(embeds [B,L,4096], mask [B,L] bool or None) when every prompt is cached; None when some are missing
        (raises instead under ``require_hit``)."""
        cached = [self.embeddings.get(p) for p in prompts]
        missing = [p for p, t in zip(prompts, cached) if t is None]
        if missing:
            assert not self.require_hit, f"text_embed_cache miss for prompt(s): {missing[:3]} (and {max(0, len(missing) - 3)} more)"
            return None
        embeds = torch.stack([self._pad(c) for c in cached], dim=0).to(device=device, dtype=dtype)
        mask = None
        if self.masks is not None:
            mcached = [self.masks.get(p) for p in prompts]
            if all(m is not None for m in mcached):
                mask = torch.stack([self._pad(m) for m in mcached], dim=0).to(device=device).bool()
        return embeds, mask


class WanVAEEncoder:
    def __init__(self, vae: nn.Module, *, encode_chunk: int = 0, fallback_dtype: torch.dtype = torch.bfloat16):
        self.vae = vae
        cfg = vae.config
        self.latents_mean = torch.tensor(list(cfg.latents_mean), dtype=torch.float32)
        self.latents_std = torch.tensor(list(cfg.latents_std), dtype=torch.float32)
        self.scale_factor_temporal = int(cfg.scale_factor_temporal)
        self.scale_factor_spatial = int(cfg.scale_factor_spatial)
        self.latent_channels = int(cfg.z_dim)
        self.dtype = module_dtype(vae, fallback=fallback_dtype)
        self.encode_chunk = max(0, int(encode_chunk))

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    def _chunks(self, n: int):
        c = self.encode_chunk
        return [(i, i + c) for i in range(0, n, c)] if c and 0 < c < n else [(0, n)]

    @torch.no_grad()
    def encode(self, video_bcthw: torch.Tensor) -> torch.Tensor:
        """(B,3,T,H,W) in [-1,1] → normalized latents (B,C,T_lat,H_lat,W_lat) fp32: deterministic μ, then (x−mean)/std."""
        assert video_bcthw.ndim == 5 and video_bcthw.shape[1] == 3, f"video_bcthw must be (B,3,T,H,W), got {tuple(video_bcthw.shape)}"
        vb = video_bcthw.to(device=self.device, dtype=self.dtype)
        lat = torch.cat([self.vae.encode(vb[a:b]).latent_dist.mode() for a, b in self._chunks(vb.shape[0])], dim=0)
        mean = self.latents_mean.to(device=lat.device, dtype=lat.dtype).view(1, -1, 1, 1, 1)
        std = self.latents_std.to(device=lat.device, dtype=lat.dtype).view(1, -1, 1, 1, 1)
        return ((lat - mean) / std).to(dtype=torch.float32)
