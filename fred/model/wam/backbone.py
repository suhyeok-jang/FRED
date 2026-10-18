"""Wan video-DiT backbone wrapper used by WAMModel.

The heavy lifting (VAE/text encoding, the student forward, the EMA teacher and the SelfDistill
captures) lives in the feature extractor; the backbone is the thin wrapper that

  - holds the extractor at ``self.extractor`` (state-dict prefix ``backbone.extractor.*``),
  - adapts the processor's collated batch (``cond_video``, ``future_video``, ``text_prompt``) to the
    extractor's signature (``prompts``, ``videos``, ``num_frames_out``, ``gt_future_videos``),
  - returns a ``BackboneOutput`` (feature, video FM loss, SelfDistill student/teacher future tokens).

``config.backbone_type`` selects the extractor:
  - ``wan22_5b``  → Wan2.2-TI2V-5B (``Wan22FeatureExtractor``), hidden 3072, 30 blocks, ``config.wan22_model_path``
  - ``wan21_14b`` → Wan2.1-I2V-14B-480P (``Wan21I2V14BFeatureExtractor``), hidden 5120, 40 blocks, ``config.wan21_model_path``
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from fred.model.wam.wan21_14b_extractor import Wan21I2V14BFeatureExtractor
from fred.model.wam.wan22_extractor import Wan22FeatureExtractor
from fred.utils.dist import is_global_zero


@dataclass
class BackboneOutput:
    vdm_embeds: torch.Tensor                            # (B, S_future, D) detached action-conditioning feature
    video_fm_loss: Optional[torch.Tensor] = None       # video flow-matching loss (training forward only)
    sd_student_future: Optional[torch.Tensor] = None       # (B, S_future, D) grad-on student tokens (train only)
    sd_teacher_future: Optional[torch.Tensor] = None       # (B, S_future, D) EMA-teacher tokens, fp32, no grad


# backbone_type → (extractor class, DiT block count, config field holding the model id/path)
_BACKBONES = {
    "wan22_5b": (Wan22FeatureExtractor, 30, "wan22_model_path"),
    "wan21_14b": (Wan21I2V14BFeatureExtractor, 40, "wan21_model_path"),
}


class WanBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        extractor_cls, n_blocks, path_field = _BACKBONES[config.backbone_type]
        model_id = getattr(config, path_field)
        layer = int(config.wan_feature_extraction_layer)
        assert 0 <= layer < n_blocks, f"wan_feature_extraction_layer={layer} out of range [0,{n_blocks}) for {config.backbone_type} ({n_blocks} DiT blocks)"
        if is_global_zero():
            print(f"[WanBackbone] {config.backbone_type}: loading from '{model_id}'; extract_layer=L{layer} ({layer}/{n_blocks} = {layer / n_blocks:.0%} depth)", flush=True)
        self.extractor = extractor_cls(model_id, extract_layer=layer, config=config)
        self._apply_freeze_policy()

    def _apply_freeze_policy(self) -> None:
        """Each comma-separated token of ``config.freeze_modules_pattern`` is a module path under ``self.extractor``
        (e.g. "text_encoder", "vae"); every one that resolves to a module gets ``requires_grad=False``
        (``text_encoder`` is None unless ``config.load_text_encoder``)."""
        frozen = []
        for token in [t.strip() for t in self.config.freeze_modules_pattern.split(",") if t.strip()]:
            mod = self.extractor
            for attr in token.split("."):
                mod = getattr(mod, attr)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = False
            frozen.append(token)
        if is_global_zero():
            print(f"[WanBackbone] froze modules under extractor: {frozen}", flush=True)

    def forward(self, backbone_input: dict) -> BackboneOutput:
        prompts = list(backbone_input["text_prompt"])
        cond_video: torch.Tensor = backbone_input["cond_video"]                    # (B, 3, T_cond, H, W) in [-1, 1]
        future_video: Optional[torch.Tensor] = backbone_input.get("future_video")   # (B, 3, T_f, H, W) or None
        fdf = self.config.frame_downsample_factor
        num_frames_out = (self.config.num_frames_out - 1) // fdf + 1
        if future_video is None:   # inference: feature extraction only
            hidden = self.extractor(prompts=prompts, videos=cond_video, num_frames_out=num_frames_out)
            return BackboneOutput(vdm_embeds=hidden)
        # training forward: the video FM term and the SelfDistill captures come from the extractor; the action-conditioning
        # feature is the detached hook capture, so the action loss never backprops into the video DiT.
        hidden, future_loss, student_future_tokens, teacher_future_tokens = self.extractor(prompts=prompts, videos=cond_video, num_frames_out=num_frames_out, gt_future_videos=future_video)
        return BackboneOutput(vdm_embeds=hidden, video_fm_loss=future_loss, sd_student_future=student_future_tokens, sd_teacher_future=teacher_future_tokens)
