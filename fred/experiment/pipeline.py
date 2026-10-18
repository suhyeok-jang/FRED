"""Training pipeline: model (fresh or warm-started), processor and datasets from one run Config."""
import json
from pathlib import Path

import numpy as np
import torch
from termcolor import colored

from fred.configs.base_config import Config
from fred.data.dataset.factory import DatasetFactory
from fred.model.wam.fred_model import WAMModel
from fred.model.wam.processor import WAMProcessor
from fred.utils.dist import is_global_zero, rank_zero_print


def _to_lists(obj):
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_lists(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_lists(v) for v in obj]
    return obj


def _load_safetensors(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """Every tensor of a single-file or HF-sharded safetensors checkpoint, on CPU."""
    from safetensors import safe_open

    index = ckpt_path / "model.safetensors.index.json"
    if index.exists():
        with open(index) as f:
            shard_files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]
    assert (ckpt_path / shard_files[0]).exists(), f"no model.safetensors(.index.json) under {ckpt_path}"
    state_dict = {}
    for shard_file in shard_files:
        with safe_open(ckpt_path / shard_file, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def build_model(config: Config, save_cfg_dir: Path) -> WAMModel:
    """WAMModel from config; with ``training.base_model_path`` the checkpoint's tensors are loaded (shape-matched,
    strict=False) and the key coverage must reach ``training.warmstart_min_coverage``."""
    model = WAMModel(config.model)
    if config.training.base_model_path is not None:
        ckpt_path = Path(config.training.base_model_path)
        rank_zero_print(f"[pipeline] warm start from {ckpt_path}")
        state_dict = _load_safetensors(ckpt_path)

        # only shape-matched keys are loaded
        model_state = model.state_dict()
        filtered = {}
        mismatched = []
        for key, value in state_dict.items():
            if key in model_state and value.shape == model_state[key].shape:
                filtered[key] = value
            elif key in model_state:
                mismatched.append(f"  {key}: ckpt={list(value.shape)} vs model={list(model_state[key].shape)}")

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if mismatched:
            rank_zero_print(colored(f"[i] Shape-mismatched keys ({len(mismatched)}):", "yellow"))
            for m in mismatched[:10]:
                rank_zero_print(colored(m, "yellow"))
        if missing:
            rank_zero_print(f"[i] Missing keys: {len(missing)} (newly initialized)")
        if unexpected:
            rank_zero_print(f"[w] Unexpected keys: {len(unexpected)}")

        # strict=False would silently leave the model mostly random on key-prefix drift or a wrong-arch
        # checkpoint, so fail below training.warmstart_min_coverage (0.5); lower it for a deliberate
        # sub-module warm start.
        n_model = len(model_state)
        n_loaded = len(filtered)
        n_ckpt = len(state_dict)
        cov = (n_loaded / n_model) if n_model else 0.0
        ckpt_used = (n_loaded / n_ckpt) if n_ckpt else 0.0
        rank_zero_print(
            f"[i] warm-start key coverage: {n_loaded}/{n_model} model params loaded "
            f"({cov:.1%}); {n_loaded}/{n_ckpt} ckpt keys used ({ckpt_used:.1%}); "
            f"missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(mismatched)}"
        )
        min_cov = float(config.training.warmstart_min_coverage)
        assert not (n_model and cov < min_cov), (
            f"warm-start key coverage {cov:.1%} ({n_loaded}/{n_model}) is below the "
            f"warmstart_min_coverage={min_cov:.0%} floor — the checkpoint '{ckpt_path}' "
            f"populated <{min_cov:.0%} of the model (silent partial load: key-prefix drift, "
            f"wrong-arch ckpt, or the wrong backbone). Fix the ckpt/backbone, or set "
            f"training.warmstart_min_coverage lower for a deliberate sub-module warm-start."
        )
        rank_zero_print(colored("[i] Checkpoint loaded.", "green"))
    rank_zero_print(colored(f"\nWAMModel config: {model.config}", "yellow"))
    if is_global_zero():
        with open(save_cfg_dir / "final_model_config.json", "w") as f:
            f.write(model.config.to_filtered_json())
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank_zero_print(f"[pipeline] parameters: {total:,} total, {trainable:,} trainable ({100 * trainable / total:.2f}%)")
    return model


def build_processor(config: Config) -> WAMProcessor:
    m = config.model
    fdf = m.frame_downsample_factor
    downsampled_num_frames_in = (m.num_frames_in - 1) // fdf + 1
    downsampled_num_frames_out = (m.num_frames_out - 1) // fdf + 1
    d = config.data
    return WAMProcessor(
        modality_configs=d.modality_configs,
        downsampled_num_frames_in=downsampled_num_frames_in,
        downsampled_num_frames_out=downsampled_num_frames_out,
        frame_downsample_factor=fdf,
        max_state_dim=m.max_state_dim,
        max_action_dim=m.max_action_dim,
        max_action_horizon=m.action_horizon,
        use_percentiles=d.use_percentiles,
        apply_sincos_state_encoding=d.apply_sincos_state_encoding,
        use_relative_action=d.use_relative_action,
        image_resize_backend=d.image_resize_backend,
        multiview_layout=d.multiview_layout,
        video_resolution=d.video_resolution,
        video_resolution_hw=d.video_resolution_hw,
        video_crop_aspect=d.video_crop_aspect,
        image_aug=d.image_aug,
        image_aug_rotate_deg=d.image_aug_rotate_deg,
        image_aug_brightness=d.image_aug_brightness,
        image_aug_contrast=d.image_aug_contrast,
        image_aug_saturation=d.image_aug_saturation,
        image_aug_hue=d.image_aug_hue,
        text_dropout=d.text_dropout,
        formalize_language=d.formalize_language,
        view_mask_probs=d.view_mask_probs,
        batch_group_fractions=d.batch_group_fractions,
    )


def build_dataset(config: Config, processor: WAMProcessor, save_cfg_dir: Path):
    """The training mixture with the processor attached; the merged statistics are written next to the config."""
    train_dataset = DatasetFactory(config=config).build(processor=processor)
    if is_global_zero():
        with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
            json.dump(_to_lists(train_dataset.get_dataset_statistics()), f, indent=2)
    return train_dataset
