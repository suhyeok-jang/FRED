"""Wan2.2-TI2V-5B feature extractor: the video-model side of the WAM.

Holds the diffusers Wan DiT (``WanDiTAdapter``), the Wan VAE encoder, the umT5 text path (an embedding cache,
optionally the encoder) and ``video_ema``, an EMA copy of the DiT that is the self-distillation teacher and the
weights used at evaluation. ``Wan21I2V14BFeatureExtractor`` subclasses it and overrides ``_load_model_layer`` /
``_condition_kwargs``.

Latent window: ``num_frames_out`` pixel frames encoded by the causal VAE. The leading slots hold the clean
[history + current] latents (cond slots), the rest are future slots. Every DiT call uses a per-token timestep map:
cond tokens at t = 0, future tokens at one timestep per sample.

Forward passes (``forward``):
  (a) extraction: future slots = Gaussian noise ε at σ = 1. The blocks[extract_layer] output on the future tokens
      (detached) conditions the action head; blocks[selfdistill_student_layer] (with grad) is the self-distillation
      student feature. Always on the live DiT.
  (b) video flow matching (training only): future slots = (1−t)·x0 + t·ε' with t ~ shift-warped uniform; loss =
      bell-weighted MSE on v = ε' − x0 over the future slots.
  (c) teacher (training only): ``video_ema`` run once without grad, teacher-forced: future slots = (1−σ)·x0 + σ·ε
      with the ε of (a), σ = selfdistill_teacher_forcing_sigma. Its blocks[extract_layer] future tokens are the
      self-distillation target. The trainer steps the EMA after every optimizer step.

Checkpoint keys: backbone.extractor.video_dit.dit.* (live DiT), backbone.extractor.vae.*,
backbone.extractor.video_ema.model.dit.* (EMA DiT).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from fred.model.modules.ema import EMA
from fred.model.modules.flow_matching import WanContinuousFlowMatchScheduler
from fred.model.modules.wan import TextEmbedCache, WanDiTAdapter, WanVAEEncoder, token_grid
from fred.model.modules.wan_t5 import load_wan_t5
from fred.utils.model import module_dtype, resolve_repo
from fred.utils.dist import is_global_zero


class _ExtractEarlyExit(Exception):
    """``config.extract_early_exit`` sentinel: aborts a forward the instant blocks[extract_layer]'s capture
    hooks have run, so blocks (L+1)..N-1 + the output head never execute. Raised ONLY from the
    dedicated exit hook (registered last on the target block) and caught ONLY around the forwards
    whose return value is discarded (forward (a) and the teacher forward), so the captured feature
    is bit-identical to the full-depth forward. The block loop draws no RNG, so the stream is
    unchanged too."""


class Wan22FeatureExtractor(nn.Module):
    """See the module docstring. Built by ``WanBackbone``."""

    # fp32 param-L2 of reference blocks of Wan-AI/Wan2.2-TI2V-5B-Diffusers (catches a silent random
    # init / meta-tensor / wrong-revision load). A ``None`` reference (14B) checks finiteness only.
    _BLOCK_L2_REF: Dict[int, Optional[float]] = {0: 276.641093, 15: 327.768725, 29: 284.287580}
    _BLOCK_L2_RTOL: float = 5e-3

    def __init__(self, model_id_or_path: str, *, extract_layer: int, config: Any):
        super().__init__()
        self.config = config
        self.torch_dtype = torch.bfloat16
        self.extract_layer = int(extract_layer)
        load_dev = "cuda" if torch.cuda.is_available() else "cpu"
        repo = resolve_repo(model_id_or_path)

        self._load_model_layer(repo, load_dev)   # transformer (adapter), vae, vae_codec
        self.scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=float(config.wan_fm_train_shift))

        # teacher-forcing σ for the self-EMA teacher's future slots (x_σ = (1−σ)·GT + σ·ε)
        self.selfdistill_teacher_forcing_sigma = float(config.selfdistill_teacher_forcing_sigma)
        assert 0.0 <= self.selfdistill_teacher_forcing_sigma <= 1.0, f"selfdistill_teacher_forcing_sigma must be in [0, 1], got {self.selfdistill_teacher_forcing_sigma}"
        self.selfdistill_teacher_fp32 = bool(config.selfdistill_teacher_fp32)
        self.video_ema_sharded = bool(config.video_ema_sharded)
        # extract_early_exit: forward (a) and the teacher forward stop right after the blocks[extract_layer]
        # capture (feature bit-identical; forward (b) always runs full depth).
        self._extract_early_exit = bool(config.extract_early_exit)
        self._extract_exit_now = False
        # hook capture state (student)
        self._cached_hidden: Optional[torch.Tensor] = None
        self._capture_hidden_enabled = False
        self._sd_student_hidden: Optional[torch.Tensor] = None
        self._sd_student_capture = False
        # hook capture state (EMA teacher)
        self._sd_teacher_hidden: Optional[torch.Tensor] = None
        self._sd_teacher_capture = False
        self._sd_teacher_exit_now = False

        self.video_ema: Optional[EMA] = None   # built in _register_hook
        self._register_hook()
        self._init_text(repo, load_dev)
        self._blocks_truncated = False
        if is_global_zero():
            print(
                f"[{type(self).__name__}] self-distillation teacher σ={self.selfdistill_teacher_forcing_sigma}, FM train shift={self.scheduler.shift}",
                flush=True,
            )

    # ── model layer ──────────────────────────────────────────────────────────────

    def _load_model_layer(self, repo: str, load_dev: str) -> None:
        """Sets ``transformer`` (DiT adapter), ``vae`` and ``vae_codec`` on ``load_dev``."""
        from diffusers import AutoencoderKLWan, WanTransformer3DModel  # noqa: PLC0415

        dit = WanTransformer3DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=self.torch_dtype)
        vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=self.torch_dtype)
        dit = dit.to(load_dev)
        self._assert_load_fingerprint(dit, self._BLOCK_L2_REF)
        self.video_dit = WanDiTAdapter(dit).to(load_dev)
        self.vae = vae.to(load_dev)
        self.vae_codec = WanVAEEncoder(self.vae, encode_chunk=self.config.vae_encode_chunk, fallback_dtype=self.torch_dtype)

    def _assert_load_fingerprint(self, dit, refs: Dict[int, Optional[float]]) -> None:
        """fp32 param-L2 of the reference blocks: within ``_BLOCK_L2_RTOL`` of the baked value, or (no baked value)
        finite and > 1 — either way a meta / zero-init / NaN / wrong-revision load fails loud."""
        blocks = dit.blocks
        got = {L: math.sqrt(sum(float(p.detach().float().pow(2).sum().item()) for p in blocks[L].parameters())) for L in refs}
        if is_global_zero():
            report = ", ".join(f"L{L}={got[L]:.4f}" + (f"(ref {refs[L]:.4f})" if refs[L] is not None else "") for L in sorted(got))
            print(f"[{type(self).__name__}] LOAD-FINGERPRINT block param-L2(fp32): {report}", flush=True)
        for L, g in got.items():
            exp = refs[L]
            if exp is None:
                assert math.isfinite(g) and g > 1.0, f"[{type(self).__name__}] LOAD-FINGERPRINT SANITY FAIL at block[{L}]: param-L2={g} (meta / zero-init / NaN load)"
            else:
                assert math.isfinite(g) and abs(g - exp) <= self._BLOCK_L2_RTOL * exp, (
                    f"[{type(self).__name__}] LOAD-FINGERPRINT MISMATCH at block[{L}]: param-L2(fp32)={g:.6f} != expected {exp:.6f} "
                    f"(rtol {self._BLOCK_L2_RTOL}) — the Wan transformer did NOT load the pretrained weights. If the repo weights "
                    f"legitimately changed, re-baseline _BLOCK_L2_REF."
                )

    def truncate_above_extraction(self) -> None:
        """Eval-only memory optimization (called by the eval loader): drop the DiT blocks above ``extract_layer``.
        Lossless for the action path (only the blocks[extract_layer] output is consumed); ``forward`` refuses a
        training forward on a truncated DiT."""
        dit = self.video_dit.dit
        blocks = dit.blocks
        keep_upto = min(int(self.extract_layer), len(blocks) - 1)
        if not (0 <= keep_upto < len(blocks) - 1):
            return
        n_before = len(blocks)
        dit.blocks = blocks[: keep_upto + 1]
        self._blocks_truncated = True
        print(f"[{type(self).__name__}] eval: transformer blocks {n_before} -> {keep_upto + 1} (extract_layer=L{self.extract_layer})", flush=True)

    # ── hooks, EMA copy, compile ──────────────────────────────────────────────────

    def _register_hook(self) -> None:
        """blocks[extract_layer] → detached feature into ``_cached_hidden`` (action conditioning);
        blocks[selfdistill_student_layer] → grad-on SelfDistill student capture; the early-exit hook is registered
        last on the extraction block. Then the EMA copy is taken from the eager student and the student
        blocks are compiled."""
        blocks = self.video_dit.blocks
        assert 0 <= self.extract_layer < len(blocks), f"extract_layer={self.extract_layer} out of bounds for {len(blocks)} Wan blocks"

        def _extract_hook(module, inp, out):
            if self._capture_hidden_enabled:
                self._cached_hidden = out.detach()

        blocks[self.extract_layer].register_forward_hook(_extract_hook)

        def sd_hook(module, inp, out):
            if self._sd_student_capture:
                self._sd_student_hidden = out  # no detach: keep the graph for the SelfDistill loss

        sd_student_layer = int(self.extract_layer if self.config.selfdistill_student_layer is None else self.config.selfdistill_student_layer)
        assert 0 <= sd_student_layer < len(blocks), f"selfdistill_student_layer={sd_student_layer} out of bounds for {len(blocks)} blocks"
        assert not (self._extract_early_exit and sd_student_layer > self.extract_layer), (
            f"selfdistill_student_layer L{sd_student_layer} > extract_layer L{self.extract_layer} with config.extract_early_exit — the forward exits before the tap fires."
        )
        blocks[sd_student_layer].register_forward_hook(sd_hook)

        def _extract_exit_hook(module, inp, out):
            if self._extract_exit_now:
                raise _ExtractEarlyExit

        blocks[self.extract_layer].register_forward_hook(_extract_exit_hook)

        self._build_video_ema()
        self._compile_blocks(blocks, "transformer (student)")

        if is_global_zero():
            print(
                "[LAYER-TAPS] 0-indexed blocks[i]: action head reads blocks[%d]; SelfDistill student=blocks[%d] -> teacher(EMA)=blocks[%d]"
                % (self.extract_layer, sd_student_layer, self.extract_layer),
                flush=True,
            )

    def _build_video_ema(self) -> None:
        """EMA copy of the video DiT (``video_ema.model``): the self-distillation teacher, hooked at blocks[extract_layer]
        for the teacher capture; block-compiled like the student."""
        self.video_ema = EMA(self.video_dit, sharded=self.video_ema_sharded)
        core = self.video_ema.model
        blocks = core.blocks

        def _sd_teacher_hook(mod, inp, out):   # teacher capture (detached)
            if self._sd_teacher_capture:
                self._sd_teacher_hidden = out.detach()

        def _sd_teacher_exit_hook(mod, inp, out):
            if self._sd_teacher_exit_now:
                raise _ExtractEarlyExit

        blocks[self.extract_layer].register_forward_hook(_sd_teacher_hook)
        blocks[self.extract_layer].register_forward_hook(_sd_teacher_exit_hook)
        self._compile_blocks(blocks, "video_ema.model (self-teacher)")
        if is_global_zero():
            n = sum(p.numel() for p in core.parameters())
            print(f"[{type(self).__name__}] video EMA copy built ({n / 1e9:.2f}B params, {'bf16 + per-rank fp32 shard' if self.video_ema_sharded else 'fp32'}); "
                  "it is the self-distillation teacher and the weights evaluation runs.", flush=True)

    def _compile_blocks(self, blocks, label: str) -> None:
        """torch.compile(mode=config.compile_mode) on blocks[1:] (block0 stays eager); None = no compile."""
        mode = self.config.compile_mode
        assert mode in (None, "default", "reduce-overhead", "max-autotune"), f"compile_mode={mode!r}: expected None, default, reduce-overhead or max-autotune"
        if mode is None:
            return
        for blk in blocks[1:]:
            blk.compile(mode=mode)
        if is_global_zero():
            print(f"[{type(self).__name__}] torch.compile(mode={mode!r}) on {label}: {len(blocks) - 1}/{len(blocks)} blocks (block0 eager)", flush=True)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def feature_dim(self) -> int:
        return int(self.video_dit.hidden_dim)  # 3072 (5B) / 5120 (14B)

    # ── text ─────────────────────────────────────────────────────────────────────

    def _init_text(self, repo: str, load_dev: str) -> None:
        """umT5 only when the config asks for it (cache misses are encoded live); otherwise the cache serves every prompt."""
        self.text_cache = TextEmbedCache(self.config.text_embed_cache, require_hit=self.config.text_embed_require_hit)
        self.text_encoder: Optional[nn.Module] = None
        self.tokenizer = None
        if self.config.load_text_encoder:
            self.text_encoder, self.tokenizer = load_wan_t5(torch_dtype=self.torch_dtype, device=load_dev)
        if is_global_zero():
            print(f"[{type(self).__name__}] text: cache={self.config.text_embed_cache or '<none>'} ({len(self.text_cache)} entries, require_hit={self.text_cache.require_hit}), "
                  f"umT5 {'loaded' if self.text_encoder is not None else 'not loaded'}", flush=True)

    def encode_prompt(self, prompts: List[str], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """prompt_embeds [B, L, 4096]: cache first, umT5 live otherwise. The pad positions are zeroed at the source
        (the DiT cross-attends all 512 positions — the pretrained-Wan convention)."""
        hit = self.text_cache.lookup(prompts, device=device, dtype=dtype)
        if hit is not None:
            embeds, mask = hit
        else:
            assert self.text_encoder is not None, f"umT5 is not loaded and the cache misses {prompts[:3]}: rebuild the text cache (scripts/precompute_text_embeds.py)"
            ids, mask = self.tokenizer(prompts, return_mask=True)
            ids, mask = ids.to(device), mask.to(device).bool()
            with torch.no_grad():
                embeds = self.text_encoder(ids, mask).to(dtype=dtype, device=device)
        assert mask is not None, "the text cache carries no masks; rebuild it with masks (needed to zero the pad positions)"
        return embeds * mask.unsqueeze(-1).to(embeds.dtype)

    # ── VAE ──────────────────────────────────────────────────────────────────────

    def _encode_video_to_latents_norm(self, video_bcthw: torch.Tensor) -> torch.Tensor:
        return self.vae_codec.encode(video_bcthw).to(device=self.device)

    def _encode_cond_padded(self, video_bcthw: torch.Tensor, num_frames_out: int) -> torch.Tensor:
        """Normalized latents of the conditioning clip zero-padded to ``num_frames_out`` pixel frames (causal VAE: the
        cond slots do not depend on the padding)."""
        video_bcthw = video_bcthw.to(device=self.device, dtype=self.vae_codec.dtype)
        n_pad = int(num_frames_out) - int(video_bcthw.shape[2])
        assert n_pad >= 0, f"conditioning clip has {int(video_bcthw.shape[2])} frames > num_frames_out={int(num_frames_out)}"
        if n_pad > 0:
            zero_pad = torch.zeros(
                (video_bcthw.shape[0], video_bcthw.shape[1], n_pad, video_bcthw.shape[3], video_bcthw.shape[4]),
                device=video_bcthw.device, dtype=video_bcthw.dtype,
            )
            video_bcthw = torch.cat([video_bcthw, zero_pad], dim=2)
        with torch.no_grad():
            return self._encode_video_to_latents_norm(video_bcthw)

    def prepare_latents(
        self, *, video: torch.Tensor, num_frames_out: int, precomputed_cond_latents: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(latents, cond_latents, cond_indicator) over the window of ``num_frames_out`` pixel frames, fp32:
        latents = Gaussian noise (B,C,T_lat,H_lat,W_lat); cond_latents = normalized VAE latents of the conditioning
        clip (``precomputed_cond_latents`` when the full-window encode already ran, else the zero-padded clip);
        cond_indicator (B,1,T_lat,1,1) marks the leading cond latent slots."""
        B = int(video.shape[0])
        C = int(self.video_dit.latent_channels)
        vae_t_scale = int(self.vae_codec.scale_factor_temporal)
        vae_s_scale = int(self.vae_codec.scale_factor_spatial)
        T_lat = (int(num_frames_out) - 1) // vae_t_scale + 1
        H_lat = int(video.shape[3]) // vae_s_scale
        W_lat = int(video.shape[4]) // vae_s_scale
        shape_out = (B, C, T_lat, H_lat, W_lat)
        if precomputed_cond_latents is not None:
            cond_latents = precomputed_cond_latents.to(device=self.device, dtype=torch.float32)
        else:
            cond_latents = self._encode_cond_padded(video, num_frames_out).to(device=self.device, dtype=torch.float32)
        assert tuple(cond_latents.shape) == shape_out, f"cond_latents shape {tuple(cond_latents.shape)} != expected {shape_out}"
        latents = torch.randn(shape_out, device=self.device, dtype=torch.float32)
        num_cond_latent_frames = min(T_lat, (int(video.shape[2]) - 1) // vae_t_scale + 1)
        cond_indicator = latents.new_zeros(B, 1, T_lat, 1, 1)
        cond_indicator[:, :, 0:num_cond_latent_frames] = 1.0
        return latents, cond_latents, cond_indicator

    def _condition_kwargs(
        self, *, videos_bcthw: torch.Tensor, cond_latents: torch.Tensor, cond_indicator: torch.Tensor,
        num_frames_out: int, gt_latents: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Extra per-forward conditioning kwargs for the DiT adapter (student AND teacher); the 5B adapter takes none."""
        return {}

    # ── per-token timestep map ───────────────────────────────────────────────────

    @staticmethod
    def _per_token_timesteps(t: torch.Tensor, *, n_cond_tok: int, n_all_tok: int) -> torch.Tensor:
        """(B, S) map: cond tokens pinned t=0, every future token at that sample's t (B,)."""
        b = int(t.shape[0])
        t_col = t.to(torch.float32).reshape(b, 1)
        tau = torch.zeros((b, int(n_all_tok)), device=t.device, dtype=torch.float32)
        tau[:, int(n_cond_tok):] = t_col
        return tau

    def _sd_teacher_future_tokens(
        self,
        *,
        prompt_embeds: torch.Tensor,
        hist_latents: torch.Tensor,
        latents: torch.Tensor,
        gt_latents: torch.Tensor,
        cond_count: int,
        transformer_dtype: torch.dtype,
        cond: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """One teacher-forced forward of the EMA DiT over the student's window: cond slots = ``hist_latents`` (clean
        [history+current], pinned t=0 per token), future slots = x_σ = (1−σ)·GT + σ·ε with ε = the student's own
        σ=1.0 future noise (``latents``), future tokens at σ·1000. Returns the detached future-token slice of
        blocks[extract_layer]."""
        teacher = self.video_ema.model
        b = int(latents.shape[0])
        device = latents.device
        tpf, n_all_tok = token_grid(latents, self.video_dit.patch_size)
        n_cond_tok = cond_count * tpf
        tf_sig = torch.full((b,), float(self.selfdistill_teacher_forcing_sigma), device=device, dtype=torch.float32)
        sig5 = tf_sig.view(b, 1, 1, 1, 1)
        eps_fut = latents[:, :, cond_count:].to(torch.float32)     # student σ=1.0 future == ε (shared)
        gt_fut = gt_latents[:, :, cond_count:].to(torch.float32)   # clean GT future
        xsig_fut = (1.0 - sig5) * gt_fut + sig5 * eps_fut
        latents_t_eff = torch.cat([hist_latents.to(latents.dtype), xsig_fut.to(latents.dtype)], dim=2)
        cond_latents_t = torch.cat([hist_latents.to(torch.float32), torch.zeros_like(eps_fut)], dim=2)
        cond_mask_t_t = latents.new_zeros(b, 1, int(latents.shape[2]), 1, 1)
        cond_mask_t_t[:, :, :cond_count] = 1.0
        tau = self._per_token_timesteps(tf_sig * float(self.scheduler.num_train_timesteps), n_cond_tok=n_cond_tok, n_all_tok=n_all_tok)
        amp_dev = "cuda" if torch.cuda.is_available() else "cpu"
        teacher_fp32 = self.selfdistill_teacher_fp32
        teacher_dtype = torch.float32 if teacher_fp32 else transformer_dtype
        in_lat = (
            cond_mask_t_t.to(torch.float32) * cond_latents_t.to(torch.float32)
            + (1.0 - cond_mask_t_t.to(torch.float32)) * latents_t_eff.to(torch.float32)
        ).to(teacher_dtype)
        with torch.no_grad(), torch.autocast(device_type=amp_dev, dtype=transformer_dtype, enabled=not teacher_fp32):
            self._sd_teacher_hidden = None
            self._sd_teacher_capture = True
            self._sd_teacher_exit_now = self._extract_early_exit
            try:
                teacher(x=in_lat, timestep=tau, context=(prompt_embeds.to(torch.float32) if teacher_fp32 else prompt_embeds), **cond)
            except _ExtractEarlyExit:
                pass
            finally:
                self._sd_teacher_capture = False
                self._sd_teacher_exit_now = False
        feat = self._sd_teacher_hidden
        self._sd_teacher_hidden = None
        assert feat is not None, "[sd-teacher] teacher hook captured no hidden (check extract_layer)."
        return feat[:, n_cond_tok:, :].float().detach()

    # ── forward ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        *,
        prompts: List[str],
        videos: torch.Tensor,              # (B,3,T_in,H,W) in [-1,1]: the [history+current] stack
        num_frames_out: int,               # cond + future pixel frames of the latent window
        gt_future_videos: Optional[torch.Tensor] = None,   # (B,3,T_f,H,W) in [-1,1]
    ):
        """Action mode (``gt_future_videos is None``): the action-conditioning feature (B, S_future, hidden_size).
        Training mode (future clip given): ``(feature, video_fm_loss, sd_student_future, sd_teacher_future)`` — the
        SelfDistill pair is ``(None, None)`` unless ``self.training`` (the teacher path runs only then)."""
        assert videos.ndim == 5 and videos.shape[1] == 3, f"`videos` must be (B,3,T,H,W), got {tuple(videos.shape)}"
        height, width = int(videos.shape[-2]), int(videos.shape[-1])
        assert height % 16 == 0 and width % 16 == 0, f"`height` and `width` must be divisible by 16, got {height} and {width}."
        assert not (self._blocks_truncated and gt_future_videos is not None), "the video DiT was truncated above the extraction layer (eval-only); the training forward needs the full DiT."
        device = self.device
        transformer_dtype = module_dtype(self.video_dit, fallback=self.torch_dtype)
        b = int(videos.shape[0])
        prompts_list = list(prompts)
        assert len(prompts_list) == b, f"len(prompts)={len(prompts_list)} must match batch size B={b}"

        prompt_embeds = self.encode_prompt(prompts_list, device=device, dtype=transformer_dtype)

        videos_bcthw = videos.to(device=device, dtype=self.vae_codec.dtype)
        num_frames_in = int(videos_bcthw.shape[2])
        vae_t_scale = int(self.vae_codec.scale_factor_temporal)
        assert num_frames_in >= 2, f"[selfdistill] the student `videos` must carry the full [history+current] stack, got {num_frames_in} frame(s)."
        with_future = gt_future_videos is not None

        # ── training: one encode of the real [history+current+future] window = FM x0 + cond-slot latents ──
        gt_latents: Optional[torch.Tensor] = None
        if with_future:
            assert gt_future_videos.ndim == 5 and gt_future_videos.shape[1] == 3, f"gt_future_videos must be (B,3,T_f,H,W), got {tuple(gt_future_videos.shape)}"
            full_bcthw = torch.cat([videos_bcthw, gt_future_videos.to(device=device, dtype=self.vae_codec.dtype)], dim=2)  # (B,3,T_in+T_f,H,W)
            with torch.no_grad():
                gt_latents = self._encode_video_to_latents_norm(full_bcthw)  # (B,C,T_lat,H,W)
            exp_T_lat = (int(num_frames_out) - 1) // vae_t_scale + 1
            assert int(gt_latents.shape[2]) == exp_T_lat, f"[selfdistill] window geometry mismatch: encoded T_lat={int(gt_latents.shape[2])} != expected {exp_T_lat} from num_frames_out={int(num_frames_out)}"

        latents, cond_latents, cond_indicator = self.prepare_latents(video=videos_bcthw, num_frames_out=int(num_frames_out), precomputed_cond_latents=gt_latents)
        cond_count = int(cond_indicator[0, 0, :, 0, 0].sum().item())
        cond_mask_t = cond_indicator.to(transformer_dtype)
        cond = self._condition_kwargs(videos_bcthw=videos_bcthw, cond_latents=cond_latents, cond_indicator=cond_indicator, num_frames_out=int(num_frames_out), gt_latents=gt_latents)

        # ── forward (a): σ=1.0 extraction with the per-token map {cond=0, future=σ_max} ──
        distill = bool(self.training)
        tpf, n_all_tok = token_grid(latents, self.video_dit.patch_size)
        n_cond_tok = cond_count * tpf
        assert 0 < n_cond_tok < n_all_tok, f"[extractor] per-token map: cond tokens {n_cond_tok} / all {n_all_tok} — expected 0 < cond < all."
        max_t = torch.full((b,), float(self.scheduler.num_train_timesteps), device=device, dtype=torch.float32)   # σ_max = 1.0
        t_extract = self._per_token_timesteps(max_t, n_cond_tok=n_cond_tok, n_all_tok=n_all_tok)
        in_latents = (cond_mask_t * cond_latents + (1.0 - cond_mask_t) * latents).to(transformer_dtype)
        self._cached_hidden = None
        self._capture_hidden_enabled = True
        self._sd_student_hidden = None
        self._sd_student_capture = distill
        self._extract_exit_now = self._extract_early_exit
        try:
            with (torch.enable_grad() if distill else torch.no_grad()):
                self.video_dit(x=in_latents, timestep=t_extract, context=prompt_embeds, **cond)
        except _ExtractEarlyExit:
            pass
        finally:
            self._extract_exit_now = False
        self._capture_hidden_enabled = False
        self._sd_student_capture = False
        hidden = self._cached_hidden  # (B, S, D), detached
        self._cached_hidden = None
        assert hidden is not None, "No transformer hidden captured. Check extract_layer / that transformer.blocks[extract_layer] runs."
        feature = hidden[:, n_cond_tok:, :]   # action conditioning = the future tokens

        if not with_future:
            return feature

        student_future_tokens: Optional[torch.Tensor] = None
        teacher_future_tokens: Optional[torch.Tensor] = None
        with torch.enable_grad():
            # ── forward (b): video flow matching on the live DiT (future slots) ──
            x0_future = gt_latents[:, :, cond_count:].to(device=device, dtype=torch.float32)
            t_fm = self.scheduler.sample_training_t(b, device=device)
            noise = torch.randn_like(x0_future)
            xt_future = self.scheduler.add_noise(x0_future, noise, t_fm)   # (1-σ)x0 + σ·noise
            v_target = self.scheduler.training_target(x0_future, noise)    # noise - x0
            xt_full = latents.clone().to(torch.float32)
            xt_full[:, :, cond_count:] = xt_future
            in_latents_fm = (cond_mask_t * cond_latents + (1.0 - cond_mask_t) * xt_full.to(transformer_dtype)).to(transformer_dtype)
            fm_timestep = self._per_token_timesteps(t_fm, n_cond_tok=n_cond_tok, n_all_tok=n_all_tok)
            v_pred = self.video_dit(x=in_latents_fm, timestep=fm_timestep, context=prompt_embeds, **cond)
            v_pred_future = v_pred[:, :, cond_count:]
            # bell-weighted per-sample MSE (mid-σ bell over the shift-warped grid)
            per_sample_mse = ((v_pred_future.float() - v_target.float()) ** 2).mean(dim=tuple(range(1, v_pred_future.dim())))  # (B,)
            w = self.scheduler.training_weight(t_fm).to(per_sample_mse).reshape(-1)
            future_loss = (per_sample_mse * w).mean()

            # ── teacher (train only): teacher-forced forward over the same window ──
            if distill:
                with torch.no_grad():
                    hist_latents = self._encode_video_to_latents_norm(videos_bcthw)  # clean [history+current] latents
                assert int(hist_latents.shape[2]) == cond_count and cond_count >= 2, (
                    f"[selfdistill] history latents {int(hist_latents.shape[2])} slots != student cond slots {cond_count} (>= 2 required)"
                )
                teacher_future_tokens = self._sd_teacher_future_tokens(
                    prompt_embeds=prompt_embeds, hist_latents=hist_latents, latents=latents, gt_latents=gt_latents,
                    cond_count=cond_count, transformer_dtype=transformer_dtype, cond=cond,
                )
                # SelfDistill pair: student future tokens (grad-on, blocks[sd_student_layer] of forward (a)) ↔ teacher future tokens
                g_stu = self._sd_student_hidden
                self._sd_student_hidden = None
                assert g_stu is not None, "SelfDistill: blocks[sd_student_layer] grad-on capture missing — the hook did not fire on forward (a)."
                assert g_stu.ndim == 3, f"SelfDistill: capture must be (B, S, D), got {tuple(g_stu.shape)}"
                student_future_tokens = g_stu[:, n_cond_tok:, :]
                assert int(student_future_tokens.shape[1]) == int(teacher_future_tokens.shape[1]), f"SelfDistill: student future block {int(student_future_tokens.shape[1])} tokens vs teacher {int(teacher_future_tokens.shape[1])}"
        return feature, future_loss, student_future_tokens, teacher_future_tokens
