"""Eval-time weights: a checkpoint carries the live modules AND their EMA copies; evaluation runs the EMA.

The model class only describes training (live modules + registered EMA copies). Here, before the model is
built, the checkpoint tensors are read on CPU and re-keyed so the EMA tensors land in the live slots and the
live tensors are dropped; after the model is built from that state dict the (now meaningless) EMA copies are
removed, leaving one set of weights in memory.
"""
import json
import os
import re

import torch
from safetensors.torch import load_file

VIDEO_EMA_PREFIX = "backbone.extractor.video_ema.model."
VIDEO_LIVE_PREFIX = "backbone.extractor.video_dit."
ACTION_EMA_PREFIX = "action_ema.model."
ACTION_LIVE_PREFIX = "action_head."
TRAIN_ONLY_PREFIXES = ("sd_student_head.", "sd_teacher_head.")


def checkpoint_shards(ckpt_dir: str) -> list[str]:
    idx = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        return sorted({os.path.join(ckpt_dir, f) for f in json.load(open(idx))["weight_map"].values()})
    single = os.path.join(ckpt_dir, "model.safetensors")
    assert os.path.isfile(single), f"{ckpt_dir}: no model.safetensors(.index.json)"
    return [single]


def is_dropped_at_eval(key: str, extract_layer: int) -> bool:
    if key.startswith("backbone.extractor.vae.decoder.") or key.startswith("backbone.extractor.vae.post_quant_conv."):
        return True
    m = re.match(r"backbone.extractor.video_dit\.dit\.blocks\.(\d+)\.", key)
    return bool(m) and int(m.group(1)) > extract_layer


def ema_state_dict(ckpt_dir: str, extract_layer: int) -> dict[str, torch.Tensor]:
    """Checkpoint tensors (CPU) with the EMA copies re-keyed onto the live modules, live tensors dropped."""
    out: dict[str, torch.Tensor] = {}
    n_video = n_action = n_live_dropped = 0
    for shard in checkpoint_shards(ckpt_dir):
        for k, v in load_file(shard, device="cpu").items():
            if k.startswith(VIDEO_EMA_PREFIX):
                nk = VIDEO_LIVE_PREFIX + k[len(VIDEO_EMA_PREFIX):]
                if is_dropped_at_eval(nk, extract_layer):
                    continue
                out[nk] = v
                n_video += 1
            elif k.startswith(ACTION_EMA_PREFIX):
                out[ACTION_LIVE_PREFIX + k[len(ACTION_EMA_PREFIX):]] = v
                n_action += 1
            elif k.startswith((VIDEO_LIVE_PREFIX, ACTION_LIVE_PREFIX)) or k.startswith(TRAIN_ONLY_PREFIXES):
                n_live_dropped += 1
            elif not is_dropped_at_eval(k, extract_layer):
                out[k] = v  # vae encoder, ...
    assert n_video > 0 and n_action > 0, (
        f"{ckpt_dir}: checkpoint carries no EMA weights (video {n_video}, action {n_action}) — eval runs the EMA only"
    )
    print(f"[EVAL-WEIGHTS] EMA -> live: video {n_video} tensors, action {n_action} tensors; "
          f"{n_live_dropped} live/training-only tensors dropped", flush=True)
    return out


def strip_for_eval(model: torch.nn.Module) -> None:
    """Remove what action generation never touches: the registered EMA copies (built by the training-shaped
    constructor), the SelfDistill heads, the video-DiT blocks above the extraction tap and the VAE decoder."""
    ext = model.backbone.extractor
    for owner, name in ((ext, "video_ema"), (model, "action_ema"), (model, "sd_student_head"), (model, "sd_teacher_head")):
        if name in owner._modules:
            del owner._modules[name]
        object.__setattr__(owner, name, None)
    ext.truncate_above_extraction()
    ext.vae.decoder = None
    ext.vae.post_quant_conv = None
