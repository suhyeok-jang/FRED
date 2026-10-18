"""WAMModel model config — the fields that decide what the model computes.

Every field is written to the checkpoint's ``config.json`` (``to_dict`` dumps all dataclass fields, never
the HF diff), so a checkpoint rebuilds the exact train-time model. Extra keys found in an older
checkpoint's config.json (removed experiment knobs) are accepted and ignored.
"""
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated

import tyro

from transformers import PretrainedConfig


def resolve_action_dit_dims(action_dit_size, action_dit_cfg, input_embedding_dim):
    """Map the action-head size preset to concrete DiT dims (idempotent).

    "DiT-B" (default) is a no-op: 16 layers, 12 heads × 64 = 768. "DiT-L" = the official DiT-L footprint
    (24 layers, 16 heads × 64 = 1024); output_dim / attention_head_dim / dropout stay untouched and
    ``input_embedding_dim`` follows the inner dim (the action/state encoders feed the DiT at that width).
    """
    size = str(action_dit_size)
    if size == "DiT-B":
        return action_dit_cfg, input_embedding_dim
    if size == "DiT-L":
        cfg = dict(action_dit_cfg)
        cfg["num_attention_heads"] = 16
        cfg["num_layers"] = 24
        return cfg, 1024
    raise ValueError(f"action_dit_size must be 'DiT-B' or 'DiT-L', got {size!r}")


RUNTIME_FIELDS = {"extract_early_exit", "compile_mode", "vae_encode_chunk", "text_embed_cache", "text_embed_require_hit", "load_text_encoder"}


# Per-backbone defaults of the fields that follow the video model (the paper's 5B and 14B recipes).
BACKBONE_DEFAULTS = {
    "wan22_5b": {"wan_feature_extraction_layer": 18, "action_dit_size": "DiT-B", "video_ema_sharded": False},
    "wan21_14b": {"wan_feature_extraction_layer": 24, "action_dit_size": "DiT-L", "video_ema_sharded": True},
}


@dataclass
class WAMConfig(PretrainedConfig):
    """WAM: Wan video DiT (world model) + cross-attention action DiT, flow-matching training with
    the SelfDistill term (see fred/model/wam)."""

    model_type: Annotated[str, tyro.conf.Suppress] = "WAM"

    # ── video backbone ──
    backbone_type: str = "wan22_5b"                # wan22_5b (Wan2.2-TI2V-5B) | wan21_14b (Wan2.1-I2V-14B)
    wan22_model_path: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    wan21_model_path: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    wan_feature_extraction_layer: int | None = None   # 0-indexed DiT block whose output feeds the action head; None = BACKBONE_DEFAULTS
    freeze_modules_pattern: str = "text_encoder,vae"

    # ── clip geometry (pixel frames before the stride) ──
    num_frames_in: int = 9                          # history + current
    num_frames_out: int = 25                        # history + current + future
    frame_downsample_factor: int = 2

    # ── losses ──
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    wan_fm_train_shift: float = 5.0                 # σ = s·u / (1 + (s − 1)·u)
    selfdistill_weight: float = 0.01                         # γ of the self-distillation term
    selfdistill_num_prototypes: int = 4096
    selfdistill_tau_student: float = 0.1
    selfdistill_tau_teacher: float = 0.04
    selfdistill_head_ema_momentum: float = 0.996
    selfdistill_teacher_forcing_sigma: float = 0.25                       # teacher-forcing σ on the EMA teacher's future slots
    selfdistill_student_layer: int | None = None             # None = wan_feature_extraction_layer
    selfdistill_teacher_fp32: bool = False
    video_ema_sharded: bool | None = None           # bf16 EMA copy + per-rank fp32 master shard; None = BACKBONE_DEFAULTS

    # ── runtime (not architecture) ──
    extract_early_exit: bool = True                 # stop forwards right after the tap block
    compile_mode: str | None = "default"          # torch.compile mode: None | default | reduce-overhead | max-autotune
    vae_encode_chunk: int = 0                       # clips per VAE encode (0 = whole batch)
    text_embed_cache: str | None = None             # scripts/precompute_text_embeds.py output
    text_embed_require_hit: bool = True             # every prompt must be cached (umT5 never loaded)
    load_text_encoder: bool = False
    wan21_clip_cond_frame: str = "current"               # current | first

    # ── action head (cross-attention DiT) ──
    action_dit_size: str | None = None              # DiT-B | DiT-L; None = BACKBONE_DEFAULTS
    action_dit_cfg: Annotated[dict, tyro.conf.Suppress] = field(default_factory=lambda: {
        "num_attention_heads": 12,
        "attention_head_dim": 64,
        "num_layers": 16,
        "output_dim": 2560,
        "dropout": 0.2,
        "final_dropout": True,
        "interleave_self_attention": True,
        "norm_type": "ada_norm",
        "positional_embeddings": None,
    })
    action_head_hidden_size: int = 2560             # state-encoder / action-decoder MLP width
    input_embedding_dim: int = 768
    add_pos_embed: bool = True
    max_seq_len: int = 1024
    max_state_dim: int = 64
    max_action_dim: int = 32
    action_horizon: int = 16
    action_valid_dim: Annotated[int | None, tyro.conf.Suppress] = None   # recorded at train for inference masking
    repeated_diffusion_steps: int = 4               # independent timesteps per sample
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    timestep_sample_scale: float = 0.999
    num_timestep_buckets: int = 1000

    def __init__(self, **kwargs):
        provided_keys = set(kwargs)
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        assert self.backbone_type in BACKBONE_DEFAULTS, f"backbone_type must be one of {sorted(BACKBONE_DEFAULTS)}, got {self.backbone_type!r}"
        for key, value in BACKBONE_DEFAULTS[self.backbone_type].items():
            if getattr(self, key) is None:
                setattr(self, key, value)
        if "action_dit_cfg" not in kwargs:   # the one field without a class-level default
            self.action_dit_cfg = self.__dataclass_fields__["action_dit_cfg"].default_factory()
        self.action_dit_cfg, self.input_embedding_dim = resolve_action_dit_dims(self.action_dit_size, self.action_dit_cfg, self.input_embedding_dim)
        # An HF-written config.json that lacks the geometry/loss fields would silently fall back to code defaults.
        critical_keys = {"num_frames_in", "num_frames_out", "frame_downsample_factor", "action_horizon", "max_action_dim", "action_dit_cfg"}
        hf_checkpoint_markers = {"architectures", "transformers_version", "dtype", "torch_dtype", "_name_or_path"}
        from_checkpoint = bool(provided_keys & hf_checkpoint_markers) or ("model_type" in provided_keys and len(provided_keys) <= 8)
        assert not from_checkpoint or critical_keys.issubset(provided_keys), (
            f"WAM checkpoint config.json is incomplete (missing {sorted(critical_keys - provided_keys)}); restore a full config.json."
        )

    def validate(self) -> "WAMConfig":
        """Cross-field constraints; called by WAMModel at construction and by the launcher after the CLI overrides."""
        assert self.backbone_type in ("wan22_5b", "wan21_14b"), (
            f"backbone_type must be 'wan22_5b' (Wan2.2-TI2V-5B) or 'wan21_14b' (Wan2.1-I2V-14B), got {self.backbone_type!r}"
        )
        fdf = self.frame_downsample_factor
        assert (self.num_frames_in - 1) % (4 * fdf) == 0 and (self.num_frames_out - 1) % (4 * fdf) == 0, (
            f"num_frames_in={self.num_frames_in}, num_frames_out={self.num_frames_out} must each satisfy (N - 1) % (4 * fdf) == 0 with fdf={fdf}"
        )
        assert (self.num_frames_out - self.num_frames_in) <= self.action_horizon, (
            f"num_frames_out - num_frames_in ({self.num_frames_out - self.num_frames_in}) must be <= action_horizon ({self.action_horizon})"
        )
        assert self.repeated_diffusion_steps >= 1, f"repeated_diffusion_steps={self.repeated_diffusion_steps}: must be >= 1"
        assert 0.0 <= self.selfdistill_teacher_forcing_sigma <= 1.0, f"selfdistill_teacher_forcing_sigma must be in [0, 1], got {self.selfdistill_teacher_forcing_sigma!r}"
        return self

    def to_dict(self) -> dict:
        """Full serialization (every dataclass field) for HF checkpoints."""
        cfg = super().to_dict()
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, Enum):
                value = value.value
            elif isinstance(value, Path):
                value = str(value)
            elif isinstance(value, tuple):
                value = list(value)
            cfg[name] = value
        cfg["model_type"] = self.model_type
        return cfg

    def to_diff_dict(self) -> dict:   # HF saves the diff against defaults by default; we always want the full config
        return self.to_dict()

    def to_filtered_json(self) -> str:
        """Architecture + recipe fields only (runtime fields dropped), for experiment_cfg/final_model_config.json."""
        cfg = {k: v for k, v in self.to_dict().items() if k not in RUNTIME_FIELDS}
        return json.dumps(cfg, indent=2, default=str)


