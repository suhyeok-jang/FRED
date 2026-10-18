"""fred policy for inference.

- FredPolicy: runs a checkpoint on nested observations (video / state / language dicts)
- FredSimPolicyWrapper: adapts the flat 'video.<cam>' / 'state.<name>' / 'action.<name>' format of the sim envs
"""

import gc
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
import numpy as np
import torch

from fred.configs.model.wam import WAMConfig
from fred.data.embodiment_tags import EmbodimentTag
from fred.data.types import ModalityConfig, StepData
from fred.model.wam.fred_model import WAMModel
from fred.model.wam.processor import WAMProcessor

from .eval_weights import ema_state_dict, is_dropped_at_eval, strip_for_eval

from .policy import BasePolicy
from .utils import check_arrays, to_dtype


STRIPPED_AT_EVAL = ("backbone.extractor.video_ema.", "action_ema.", "sd_student_head.", "sd_teacher_head.")   # removed by strip_for_eval


class FredPolicy(BasePolicy):
    """Inference policy for a fred checkpoint: validate the observation, run it through the checkpoint's
    processor and model, decode the actions.

    Observations carry video, state and language modalities; actions follow the model's modality config.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
        text_embed_cache: str | None = None,
        text_embed_require_hit: bool = True,
        action_seed: int | None = None,
    ):
        """``model_path``: a checkpoint dir or an HF-hub repo id already in the local cache (never downloads).
        ``text_embed_cache``: umT5 embeddings for the eval prompts (overrides the checkpoint's training-time
        path); ``action_seed`` fixes the action-noise generator (paired evals)."""
        super().__init__(strict=strict)
        ckpt_dir = str(model_path) if Path(model_path).is_dir() else snapshot_download(model_path, local_files_only=True)

        # eval weights: the EMA copies become the live modules (see eval_weights.py)
        cfg = WAMConfig.from_pretrained(ckpt_dir)
        cfg.text_embed_cache = text_embed_cache
        cfg.text_embed_require_hit = bool(text_embed_require_hit)
        cfg.load_text_encoder = not text_embed_require_hit
        layer = int(cfg.wan_feature_extraction_layer)
        model = WAMModel(cfg)   # built like training (pretrained Wan weights, EMA copies); the checkpoint then overwrites it
        missing, unexpected = model.load_state_dict(ema_state_dict(ckpt_dir, layer), strict=False)
        assert not unexpected, f"{ckpt_dir}: tensors the model has no slot for: {unexpected[:8]}"
        not_loaded = [k for k in missing if not (k.startswith(STRIPPED_AT_EVAL) or is_dropped_at_eval(k, layer))]
        assert not not_loaded, f"{ckpt_dir}: tensors missing from the checkpoint: {not_loaded[:8]}"
        strip_for_eval(model)
        self.model = model.to(device=device, dtype=torch.bfloat16).eval()
        # the removed modules (built on the GPU, ~21 GB for wan22_5b) sit in reference cycles until a collection runs
        gc.collect()
        torch.cuda.empty_cache()

        self._action_generator = None
        if action_seed is not None:
            self._action_generator = torch.Generator(device=self.model.device).manual_seed(int(action_seed))
            print(f"[FredPolicy] action noise seeded: seed={int(action_seed)}", flush=True)

        self.processor = WAMProcessor.from_pretrained(ckpt_dir)
        self.processor.eval()

        self.embodiment_tag = embodiment_tag
        self.modality_configs = self.processor.modality_configs[self.embodiment_tag.value]
        self.collate_fn = self.processor.collator

        language_keys = self.modality_configs["language"].modality_keys
        assert len(self.modality_configs["language"].delta_indices) == 1, "Only one language delta index is supported"
        # several keys are alternative instructions of one episode (DROID has three; training samples one per
        # step), so inference reads the first
        self.language_key = language_keys[0]

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Split a batched observation (B, ...) into a list of B single observations."""
        batch_size = next(iter(value["video"].values())).shape[0]
        return [
            {modality: {k: v[i] for k, v in value[modality].items()} for modality in ("video", "state", "language")}
            for i in range(batch_size)
        ]

    def _to_step_data(self, observation: dict[str, Any]) -> StepData:
        """Single observation dict -> StepData for the processor."""
        return StepData(
            images=observation["video"],
            states=observation["state"],
            actions={},  # No ground truth actions during inference
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Assert the nested observation layout (T = the modality's horizon from the modality config):
            - video: dict[str, np.ndarray[np.uint8, (B, T, H, W, 3)]]
            - state: dict[str, np.ndarray[np.float32, (B, T, D)]]
            - language: dict[str, list[list[str]]] of shape (B, T), T == 1, under the first language key
        """
        for modality in ("video", "state", "language"):
            assert modality in observation, f"Observation must contain a '{modality}' key"
            assert isinstance(observation[modality], dict), (
                f"Observation '{modality}' must be a dictionary. Got {type(observation[modality])}"
            )
        video, state = (self.modality_configs[m] for m in ("video", "state"))
        bs = check_arrays(
            observation["video"], video.modality_keys, len(video.delta_indices), np.uint8, 5, "Video", channels=3
        )
        check_arrays(observation["state"], state.modality_keys, len(state.delta_indices), np.float32, 3, "State", bs)
        key = self.language_key
        assert key in observation["language"], f"Language key '{key}' must be in observation"
        batched_language = observation["language"][key]
        assert isinstance(batched_language, list), f"Language key '{key}' must be a list. Got {type(batched_language)}"
        assert len(batched_language) == bs, f"Language key '{key}' must have batch size {bs}. Got {len(batched_language)}"
        for item in batched_language:
            # exactly one instruction per timestep
            assert isinstance(item, list) and len(item) == 1 and isinstance(item[0], str), (
                f"Language batch item must be a one-string list. Got {item!r}"
            )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Unbatch -> per-sample processor -> collate -> model.get_action -> decode to physical units."""
        states = []  # dict[str, np.ndarray[np.float32, (T, D)]] per sample
        processed_inputs = []
        for obs in self._unbatch_observation(observation):
            step_data = self._to_step_data(obs)
            states.append(step_data.states)
            processed_inputs.append(self.processor(step_data))
        inputs = to_dtype(self.collate_fn(processed_inputs)["inputs"], dtype=torch.bfloat16)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model_pred = self.model.get_action(inputs, generator=self._action_generator)
        normalized_action = model_pred["action_pred"].float()

        # decode from normalized space back to physical units
        batched_states = {
            k: np.stack([s[k] for s in states], axis=0) for k in self.modality_configs["state"].modality_keys
        }  # (B, T, D)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )
        return {key: value.astype(np.float32) for key, value in unnormalized_action.items()}, {}

    def check_action(self, action: dict[str, Any]) -> None:
        """Assert every action key of the modality config is an np.float32 array of shape (B, T, D), T = action horizon."""
        cfg = self.modality_configs["action"]
        check_arrays(action, cfg.modality_keys, len(cfg.delta_indices), np.float32, 3, "Action")

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_configs

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """The policy is stateless; nothing to reset."""
        return {}


class FredSimPolicyWrapper(BasePolicy):
    """Adapter between the flat observation/action format of the bundled sim environments and FredPolicy's
    nested format:

    - 'video.<cam>' -> observation['video']['<cam>'] (env camera names aliased to the policy's view keys,
      see VIDEO_KEY_ALIASES), 'state.<name>' -> observation['state']['<name>']
    - the first language key of the modality config holds a (B,) tuple/list of instructions
    - action['<name>'] -> 'action.<name>'
    """

    # policy view key -> raw env camera names that serve it (RoboCasa kitchen / GR-1 tabletop)
    VIDEO_KEY_ALIASES = {
        "ego_view": ("ego_view_bg_crop_pad_res256_freq20", "ego_view_pad_res256_freq20"),
        "left_view": ("res256_image_side_0",),
        "right_view": ("res256_image_side_1",),
        "wrist_view": ("res256_image_wrist_0",),
    }

    def __init__(self, policy: FredPolicy, *, strict: bool = True):
        super().__init__(strict=strict)
        self.policy = policy

    def _remap_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Copy of the flat observation with the env's camera names aliased to the policy's video keys."""
        obs = dict(observation)
        for key in self.policy.modality_configs["video"].modality_keys:
            if f"video.{key}" in obs:
                continue
            for alias in self.VIDEO_KEY_ALIASES.get(key, ()):
                if f"video.{alias}" in obs:
                    obs[f"video.{key}"] = obs[f"video.{alias}"]
                    break
        return obs

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Assert the flat sim observation layout (after camera-key aliasing):
            - 'video.<name>': np.ndarray[np.uint8, (B, T, H, W, 3)]
            - 'state.<name>': np.ndarray[np.float32, (B, T, D)]
            - language: tuple[str] | list[str] of length B under the modality config's first language key
        """
        obs = self._remap_observation(observation)
        video, state = (self.policy.modality_configs[m] for m in ("video", "state"))
        bs = check_arrays(
            obs, [f"video.{k}" for k in video.modality_keys], len(video.delta_indices), np.uint8, 5, "Video", channels=3
        )
        check_arrays(obs, [f"state.{k}" for k in state.modality_keys], len(state.delta_indices), np.float32, 3, "State", bs)
        key = self.policy.language_key
        assert key in obs, f"Language key '{key}' must be in observation"
        batched_language = obs[key]
        assert isinstance(batched_language, (tuple, list)), (
            f"Language key '{key}' must be a tuple or list. Got {type(batched_language)}"
        )
        assert len(batched_language) == bs, f"Language key '{key}' must have batch size {bs}. Got {len(batched_language)}"
        assert all(isinstance(item, str) for item in batched_language), "Language batch items must be strings"

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Flat sim observation -> nested FredPolicy observation -> actions -> flat 'action.<name>' keys."""
        obs = self._remap_observation(observation)
        configs = self.policy.modality_configs
        language_key = self.policy.language_key
        new_obs = {
            "video": {key: obs[f"video.{key}"] for key in configs["video"].modality_keys},
            "state": {key: obs[f"state.{key}"] for key in configs["state"].modality_keys},
            # tuple[str] | list[str] (B,) -> list[list[str]] (B, 1): one string per timestep
            "language": {language_key: [[str(item)] for item in obs[language_key]]},
        }
        action, info = self.policy.get_action(new_obs, options)
        return {f"action.{key}": value for key, value in action.items()}, info

    def check_action(self, action: dict[str, Any]) -> None:
        """Assert every 'action.<name>' key is an np.float32 array of shape (B, T, D), T = action horizon."""
        cfg = self.policy.modality_configs["action"]
        check_arrays(action, [f"action.{k}" for k in cfg.modality_keys], len(cfg.delta_indices), np.float32, 3, "Action")

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.policy.get_modality_config()

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.policy.reset(options)
