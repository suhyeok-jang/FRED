"""WAMModel processor + collator: state/action normalization, per-view resize + augmentation,
multi-view canvas layout, cond/future frame split, text normalization / caption dropout.

Persisted to ``processor_config.json`` (eval rebuilds the exact train-time geometry from it):
frame geometry, action/state dims, multiview_layout, video_resolution / video_resolution_hw / video_crop_aspect,
image_aug / image_aug_rotate_deg / colour jitter, image_resize_backend, use_percentiles, use_relative_action /
text_dropout, formalize_language.
"""
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature

from fred.data.dataset.robot_human_mix import parse_batch_group_fractions
from fred.data.image_augmentations import apply_aug_eval, apply_aug_train, apply_resize, build_view_resize, sample_aug_params
from fred.data.state_action_processor import StateActionProcessor
from fred.data.types import ModalityConfig, StepData
from fred.data.utils import parse_modality_configs, to_json_serializable
from fred.utils.dist import rank_zero_print


def normalize_language(text: str) -> str:
    """Lower-case and strip punctuation (data.formalize_language); the text-embed cache is keyed by this form."""
    return re.sub(r"[^\w\s]", "", text.lower())



def parse_view_mask_probs(s: str | None) -> tuple[float, float, float] | None:
    """'p_both,p_mask_first,p_mask_second' → tuple, or None when unset/empty (masking off)."""
    if s is None or not s.strip():
        return None
    parts = [float(x) for x in s.strip().split(",")]
    assert len(parts) == 3, f"data.view_mask_probs needs 3 comma-separated probs (both,left,right), got {s!r}"
    assert not (any(p < 0 for p in parts) or abs(sum(parts) - 1.0) > 1e-6), f"data.view_mask_probs must be non-negative and sum to 1, got {parts}"
    return (parts[0], parts[1], parts[2])


def draw_view_mask(
    probs: tuple[float, float, float] | None,
    image_keys: list[str],
    present_views: list[str],
    u: float,
) -> list[str]:
    """Return the list of views (subset of image_keys) to render BLACK for this sample.

    u ∈ [0,1) is the uniform draw (injected for testability). Rules:
      - views not in `present_views` are always black (regardless of probs / u);
      - probs None (masking off) → only the missing views;
      - a draw that would black the ONLY present view is ignored (never a fully black sample);
      - the draw covers image_keys[0] (p[1]) and image_keys[1] (p[2]); >2 views: extra views
        are never masked by the draw.
    """
    missing = [k for k in image_keys if k not in present_views]
    if probs is None:
        return missing
    p_both, p_first, _p_second = probs
    if u < p_both:
        chosen = None
    elif u < p_both + p_first:
        chosen = image_keys[0] if len(image_keys) >= 1 else None
    else:
        chosen = image_keys[1] if len(image_keys) >= 2 else None
    out = list(missing)
    if chosen is not None and chosen not in out:
        remaining = [k for k in image_keys if k not in out and k != chosen]
        if remaining:  # never black the last present view
            out.append(chosen)
    return [k for k in image_keys if k in out]  # canonical order


class WAMCollator:
    """Collate samples into batched tensors.

    text_prompt: list[str]; cond_video / future_video: [B, 3, T, H, W] in [-1, 1]; state, action,
    action_mask: stacked. ``human_frac`` (data.batch_group_fractions): assert the exact per-batch robot/human split (OpenArm co-training)."""

    def __init__(self, human_frac: float | None = None):
        self.human_frac = human_frac

    def __call__(self, features: list[dict[str, Any]]) -> BatchFeature:
        if self.human_frac is not None and features and ("is_human" in features[0]):
            n_human = sum(int(_e["is_human"]) for _e in features)
            want_h = self.human_frac * len(features)
            assert abs(n_human - want_h) <= 1e-6, f"[batch_group_fractions] inexact split: rows={len(features)} human={n_human} (expected human={want_h:g})"
        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))
        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            if key == "text_prompt":
                batch[key] = values
            elif isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            elif isinstance(values[0], np.ndarray):
                batch[key] = torch.from_numpy(np.stack(values))
            elif isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values)
            else:
                batch[key] = values
        return BatchFeature(data={"inputs": batch})


class WAMProcessor(ProcessorMixin):
    """See the module docstring."""

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics=None,
        use_percentiles: bool = False,
        downsampled_num_frames_in: int = 1,
        downsampled_num_frames_out: int = 17,
        frame_downsample_factor: int = 1,
        max_state_dim: int = 64,
        max_action_dim: int = 64,
        max_action_horizon: int = 16,
        formalize_language: bool = True,
        image_resize_backend: str = "cv2_linear",
        multiview_layout: str = "hconcat",
        video_resolution: int = 224,
        image_aug: bool = False,
        image_aug_rotate_deg: float = 0.0,
        image_aug_brightness: float = 0.2,
        image_aug_contrast: float = 0.2,
        image_aug_saturation: float = 0.2,
        image_aug_hue: float = 0.05,
        apply_sincos_state_encoding: bool = False,
        video_resolution_hw: str | None = None,
        video_crop_aspect: bool = False,
        text_dropout: float = 0.0,
        use_relative_action: bool = False,
        view_mask_probs: str | None = None,
        batch_group_fractions: str | None = None,
    ):
        self.modality_configs = parse_modality_configs(modality_configs)
        self.batch_group_fractions = parse_batch_group_fractions(batch_group_fractions)
        self.view_mask_probs = parse_view_mask_probs(view_mask_probs)

        # ── state/action normalization: min/max or q01/q99 bounds → [-1, 1]; RELATIVE-declared action keys
        #    are converted to action − state[anchor] and normalized on the relative stats ──
        self.state_action_processor = StateActionProcessor(
            modality_configs=modality_configs, statistics=statistics, use_percentiles=use_percentiles,
            use_relative_action=use_relative_action, apply_sincos_state_encoding=apply_sincos_state_encoding,
        )
        self.use_percentiles = use_percentiles
        self.use_relative_action = use_relative_action
        self.apply_sincos_state_encoding = bool(apply_sincos_state_encoding)

        # ── video geometry / augmentation ──
        self.multiview_layout = multiview_layout
        assert self.multiview_layout in ("hconcat", "tmosaic", "vconcat"), f"multiview_layout must be hconcat|tmosaic|vconcat, got {self.multiview_layout!r}"
        self.image_aug = bool(image_aug)
        self.image_aug_rotate_deg = float(image_aug_rotate_deg)
        self.image_aug_jitter = (float(image_aug_brightness), float(image_aug_contrast), float(image_aug_saturation), float(image_aug_hue))
        assert all(v >= 0 for v in self.image_aug_jitter) and self.image_aug_jitter[3] <= 0.5, f"colour jitter out of range: {self.image_aug_jitter}"
        self.text_dropout = float(text_dropout)
        if self.image_aug:
            rank_zero_print(f"[WAMProcessor] augmentation: crop area 0.9 + rotate ±{self.image_aug_rotate_deg}° + jitter b/c/s/h = {'/'.join(str(v) for v in self.image_aug_jitter)}; eval = center crop")
        if self.text_dropout > 0.0:
            rank_zero_print(f"[text_dropout] train caption dropout p={self.text_dropout} (requires '' in the text cache).")

        # per-view cell: the square video_resolution, or the "HxW" rectangle video_resolution_hw (DROID 192x320, OpenArm
        # 192x256); the torch_bilinear backend and the aspect crop belong to the square / rectangle path respectively
        self.image_resize_backend = image_resize_backend
        self.video_resolution = video_resolution
        self.video_resolution_hw = None
        self.video_crop_aspect = bool(video_crop_aspect)
        if video_resolution_hw:
            _h, _w = (int(x) for x in str(video_resolution_hw).strip().lower().split("x"))
            self.video_resolution_hw = (_h, _w)
            assert image_resize_backend == "cv2_linear", f"video_resolution_hw uses the cv2_linear resize, got {image_resize_backend!r}"
            print(f"[WAMProcessor] per-view resize {_h}x{_w}{' (center-crop to aspect first)' if self.video_crop_aspect else ''}", flush=True)
        else:
            assert not self.video_crop_aspect, "video_crop_aspect needs video_resolution_hw"
        self.video_hw = self.video_resolution_hw or (video_resolution, video_resolution)
        self._view_resize = build_view_resize(*self.video_hw, resize_backend=image_resize_backend, crop_aspect=self.video_crop_aspect)

        self.downsampled_num_frames_in = downsampled_num_frames_in
        self.downsampled_num_frames_out = downsampled_num_frames_out
        self.frame_downsample_factor = frame_downsample_factor
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_action_horizon = max_action_horizon
        self.formalize_language = formalize_language
        self.collator = WAMCollator(
            human_frac=None if self.batch_group_fractions is None else self.batch_group_fractions["human"],
        )
        self.train()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs):
        """Rebuild the processor from a checkpoint directory (processor_config.json + experiment_cfg/). ``kwargs``
        are ignored."""
        ckpt_dir = Path(pretrained_model_name_or_path)
        with open(ckpt_dir / "processor_config.json", "r") as f:
            config = json.load(f)
        experiment_cfg_dir = ckpt_dir / "experiment_cfg"
        with open(experiment_cfg_dir / "config.yaml", "r") as f:
            full_config = yaml.unsafe_load(f)
        modality_configs = full_config.data.modality_configs
        statistics_file = experiment_cfg_dir / "dataset_statistics.json"
        statistics_fallback = ckpt_dir / "statistics.json"   # older checkpoints
        if statistics_file.exists():
            with open(statistics_file, "r") as f:
                statistics = json.load(f)
        elif statistics_fallback.exists():
            with open(statistics_fallback, "r") as f:
                statistics = json.load(f)
            rank_zero_print(f"[WAMProcessor] experiment_cfg/dataset_statistics.json absent; loaded statistics from {statistics_fallback}.")
        else:
            raise FileNotFoundError(f"Normalization statistics not found for checkpoint {ckpt_dir}: neither {statistics_file} nor {statistics_fallback} exists.")
        frame_downsample_factor = config.get("frame_downsample_factor", 1)
        downsampled_num_frames_in = config.get("downsampled_num_frames_in")
        if downsampled_num_frames_in is None:   # older checkpoints: derive from the model geometry
            downsampled_num_frames_in = (full_config.model.num_frames_in - 1) // frame_downsample_factor + 1
        inference_video_delta_indices = list(range(-(downsampled_num_frames_in - 1) * frame_downsample_factor, 1, frame_downsample_factor))
        for embodiment_cfg in modality_configs.values():
            if "video" in embodiment_cfg:
                embodiment_cfg["video"].delta_indices = inference_video_delta_indices
        return cls(
            modality_configs=modality_configs,
            statistics=statistics,
            downsampled_num_frames_in=downsampled_num_frames_in,
            downsampled_num_frames_out=config.get("downsampled_num_frames_out", 17),
            frame_downsample_factor=frame_downsample_factor,
            max_state_dim=config.get("max_state_dim", 64),
            max_action_dim=config.get("max_action_dim", 64),
            max_action_horizon=config.get("max_action_horizon", 16),
            formalize_language=config.get("formalize_language", True),
            use_percentiles=config.get("use_percentiles", False),
            image_resize_backend=config.get("image_resize_backend", "cv2_linear"),
            multiview_layout=config.get("multiview_layout", "hconcat"),
            video_resolution=config.get("video_resolution", 224),
            image_aug=config.get("image_aug", False),
            image_aug_rotate_deg=config.get("image_aug_rotate_deg", 0.0),
            image_aug_brightness=config.get("image_aug_brightness", 0.2),
            image_aug_contrast=config.get("image_aug_contrast", 0.2),
            image_aug_saturation=config.get("image_aug_saturation", 0.2),
            image_aug_hue=config.get("image_aug_hue", 0.05),
            apply_sincos_state_encoding=config.get("apply_sincos_state_encoding", False),
            video_resolution_hw=config.get("video_resolution_hw", config.get("video_res_hw")),   # old key: video_res_hw
            video_crop_aspect=config.get("video_crop_aspect", False),
            text_dropout=config.get("text_dropout", 0.0),
            use_relative_action=config.get("use_relative_action", False),
        )

    def save_pretrained(self, save_directory, **kwargs):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        config = {
            "processor_class": "WAMProcessor",
            "downsampled_num_frames_in": self.downsampled_num_frames_in,
            "downsampled_num_frames_out": self.downsampled_num_frames_out,
            "frame_downsample_factor": self.frame_downsample_factor,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "max_action_horizon": self.max_action_horizon,
            "formalize_language": self.formalize_language,
            "use_percentiles": self.use_percentiles,
            "image_resize_backend": self.image_resize_backend,
            "multiview_layout": self.multiview_layout,
            "video_resolution": self.video_resolution,
            "use_relative_action": self.use_relative_action,
            "apply_sincos_state_encoding": self.apply_sincos_state_encoding,
        }
        if self.image_aug:
            config["image_aug"] = True
            config["image_aug_rotate_deg"] = float(self.image_aug_rotate_deg)
            config["image_aug_brightness"], config["image_aug_contrast"], config["image_aug_saturation"], config["image_aug_hue"] = self.image_aug_jitter
        if self.video_resolution_hw:
            config["video_resolution_hw"] = f"{self.video_resolution_hw[0]}x{self.video_resolution_hw[1]}"
            if self.video_crop_aspect:
                config["video_crop_aspect"] = True
        if self.text_dropout > 0.0:
            config["text_dropout"] = self.text_dropout
        with open(save_directory / "processor_config.json", "w") as f:
            json.dump(config, f, indent=2)
        with open(save_directory / "statistics.json", "w") as f:
            json.dump(to_json_serializable(self.state_action_processor.statistics), f, indent=2)

    def train(self):
        self.training = True
        self.state_action_processor.train()

    def eval(self):
        self.training = False
        self.state_action_processor.eval()

    def set_statistics(self, statistics):
        self.state_action_processor.set_statistics(statistics)

    def decode_action(self, action, embodiment_tag, state=None):
        """Split the padded (..., max_action_horizon, max_action_dim) prediction into joint groups and undo the
        normalization (+ relative -> absolute with the raw ``state`` anchor)."""
        out_dict = {}
        start_idx = 0
        joint_groups = self.modality_configs[embodiment_tag.value]["action"].modality_keys
        action_horizon = len(self.modality_configs[embodiment_tag.value]["action"].delta_indices)
        for key in joint_groups:
            joint_dim = self.state_action_processor.norm_params[embodiment_tag.value]["action"][key]["dim"]
            out_dict[key] = action[..., :action_horizon, start_idx:start_idx + joint_dim]
            start_idx += joint_dim
        return self.state_action_processor.unapply_action(out_dict, embodiment_tag.value, state=state)

    # ── sample ───────────────────────────────────────────────────────────────────

    def __call__(self, step: StepData):
        embodiment_tag = step.embodiment
        normalized_states, normalized_actions = self.state_action_processor.apply(
            state=step.states, action=step.actions, embodiment_tag=embodiment_tag.value,
        )
        if normalized_actions:
            action_keys = self.modality_configs[embodiment_tag.value]["action"].modality_keys
            normalized_actions = torch.cat([torch.from_numpy(normalized_actions[key]) for key in action_keys], dim=-1)
            action_dim = normalized_actions.shape[1]
            normalized_actions = torch.cat([normalized_actions, torch.zeros(normalized_actions.shape[0], self.max_action_dim - action_dim)], dim=-1)
            action_horizon = normalized_actions.shape[0]
            normalized_actions = torch.cat([normalized_actions, torch.zeros(self.max_action_horizon - action_horizon, self.max_action_dim)], dim=0)
            action_mask = torch.ones_like(normalized_actions)
            action_mask[action_horizon:] = 0
            action_mask[:, action_dim:] = 0
        else:
            assert not self.training, "Action is required in training mode"
            normalized_actions = None
            action_mask = None
        state_keys = self.modality_configs[embodiment_tag.value]["state"].modality_keys
        normalized_states = torch.cat([torch.from_numpy(normalized_states[key]) for key in state_keys], dim=-1)
        normalized_states = torch.cat([normalized_states, torch.zeros(normalized_states.shape[0], self.max_state_dim - normalized_states.shape[1])], dim=-1)

        language = normalize_language(step.text) if self.formalize_language else step.text
        if self.training and self.text_dropout > 0.0 and random.random() < self.text_dropout:
            language = ""

        image_keys = self.modality_configs[embodiment_tag.value]["video"].modality_keys
        # OpenArm robot+human: per-sample view masking (missing views are always black)
        vm_probs = self.view_mask_probs if self.training else None
        present_views = [k for k in image_keys if k in step.images]
        masked_views = draw_view_mask(vm_probs, list(image_keys), present_views, random.random() if vm_probs is not None else 0.0)

        cond_video, future_video = self._process_video_frames(image_keys, step.images, mask_views=masked_views)

        result = {"text_prompt": language, "cond_video": cond_video, "state": normalized_states.to(torch.get_default_dtype())}
        if future_video is not None:
            result["future_video"] = future_video
        if normalized_actions is not None:
            result["action"] = normalized_actions.to(torch.get_default_dtype())
        if action_mask is not None:
            result["action_mask"] = action_mask
        return result

    def _process_video_frames(self, image_keys: list[str], images: dict, mask_views: list[str] | None = None):
        """Per view: resize → (train) one augmentation param set shared by all views/frames, (eval) center
        crop; masked / missing views → black; canvas layout; [-1, 1]; cond/future split."""
        resized_views = {}
        missing_views = [v for v in image_keys if v not in images]   # human datasets lacking a camera
        for view in image_keys:
            if view in missing_views and mask_views and view in mask_views:
                continue
            assert view in images, f"{view} not in {images}"
            resized_views[view] = apply_resize(self._view_resize, images[view])  # [T, C, H, W] uint8
        for view in missing_views:
            resized_views[view] = torch.zeros_like(next(iter(resized_views.values())))
        if self.image_aug and self.training:
            params = sample_aug_params(*self.video_hw, rotate_deg=self.image_aug_rotate_deg, jitter=self.image_aug_jitter)
            augmented_views = {view: apply_aug_train(resized_views[view], params) for view in image_keys}
        elif self.image_aug:
            augmented_views = {view: apply_aug_eval(resized_views[view]) for view in image_keys}
        else:
            augmented_views = resized_views
        if mask_views:
            for view in mask_views:
                augmented_views[view] = torch.zeros_like(augmented_views[view])
        return self._views_to_cond_future(augmented_views, image_keys)

    def _views_to_cond_future(self, augmented_views: dict, image_keys: list[str]):
        """Canvas layout (tmosaic: wrist on top, left|right half-res below; hconcat: side by side; vconcat:
        stacked) → [-1, 1] float [3, T, H, W] → cond/future split."""
        T = augmented_views[image_keys[0]].shape[0]
        if self.multiview_layout == "tmosaic":
            _left, _right, _wrist = (augmented_views[image_keys[i]] for i in (0, 1, 2))
            _, _, _Hv, _Wv = _wrist.shape
            ls = F.interpolate(_left, size=(_Hv // 2, _Wv // 2), mode="bilinear", align_corners=False)
            rs = F.interpolate(_right, size=(_Hv // 2, _Wv // 2), mode="bilinear", align_corners=False)
            frames_concat = torch.cat([_wrist, torch.cat([ls, rs], dim=-1)], dim=-2)   # [T, C, 3H/2, W]
        elif self.multiview_layout == "vconcat":
            frames_concat = torch.cat([augmented_views[view] for view in image_keys], dim=-2)
        else:
            frames_concat = torch.cat([augmented_views[view] for view in image_keys], dim=-1)   # [T, C, H, V*W]
        current_frame_idx = self.downsampled_num_frames_in - 1
        assert T > current_frame_idx, f"Need at least {self.downsampled_num_frames_in} frames; got {T}."
        assert frames_concat.dtype == torch.uint8, f"expected uint8 frames, got {frames_concat.dtype}"
        frames_float = (frames_concat.float() / 127.5 - 1.0).permute(1, 0, 2, 3)  # [3, T, H, W]
        cond_video = frames_float[:, :self.downsampled_num_frames_in]
        future_video = None
        if T >= self.downsampled_num_frames_out:
            future_video = frames_float[:, self.downsampled_num_frames_in:self.downsampled_num_frames_out]
        elif self.training:
            rank_zero_print(f"[WARNING] Not enough frames ({T}) for downsampled_num_frames_out ({self.downsampled_num_frames_out})")
        return cond_video, future_video


