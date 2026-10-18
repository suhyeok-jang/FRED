"""State / action normalization and the absolute <-> relative action conversion (per modality config)."""

from copy import deepcopy

import numpy as np

from fred.data.types import ActionRepresentation, ModalityConfig
from fred.data.utils import (
    apply_sincos_encoding,
    normalize_values_minmax,
    parse_modality_configs,
    unnormalize_values_minmax,
)


class StateActionProcessor:
    """Normalizes states (min/max or q01/q99 to [-1, 1], optional sin/cos encoding) and actions, and converts
    RELATIVE action keys to/from absolute using the current state as anchor (data.use_relative_action)."""

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
        use_percentiles: bool = False,
        apply_sincos_state_encoding: bool = False,
        use_relative_action: bool = False,
    ):
        """
        Args:
            modality_configs: {embodiment_tag: {"state" | "action": ModalityConfig}}
            statistics: {embodiment_tag: {"state" | "action" | "relative_action": {joint_group: {min, max, mean,
                std, q01, q99}}}} (relative_action values are (horizon, D) tables)
            use_percentiles: q01/q99 instead of min/max as normalization bounds
            apply_sincos_state_encoding: sin/cos-encode the state keys listed in sincos_encoding_keys
            use_relative_action: convert RELATIVE action keys to action − current state
        """
        self.modality_configs = parse_modality_configs(modality_configs)
        self.statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
        self.use_percentiles = use_percentiles
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action

        # norm_params[embodiment_tag][modality][joint_group] -> {min, max, dim}
        self.norm_params: dict[str, dict[str, dict[str, dict]]] = {}

        if statistics is not None:
            self.set_statistics(statistics)

        self.train()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
    ) -> None:
        """Install (merged) dataset statistics and recompute the normalization parameters."""
        for key in statistics:
            self.statistics[key] = deepcopy(statistics[key])
        self._compute_normalization_parameters()

    def _relative_keys(self, embodiment_tag: str) -> list[str]:
        """Action keys converted to action − state anchor (RELATIVE rep and use_relative_action)."""
        if not self.use_relative_action:
            return []
        config = self.modality_configs[embodiment_tag]["action"]
        if config.action_configs is None:
            return []
        return [
            key
            for key, action_config in zip(config.modality_keys, config.action_configs)
            if action_config.representation == ActionRepresentation.RELATIVE
        ]

    def _compute_normalization_parameters(self) -> None:
        """Bounds per joint group (q01/q99 or min/max); relative action keys take the (horizon, D) relative_action
        table instead."""
        lo, hi = ("q01", "q99") if self.use_percentiles else ("min", "max")
        for embodiment_tag, statistics in self.statistics.items():
            params = {}
            for modality in ["state", "action"]:
                if modality not in statistics:
                    continue
                params[modality] = {
                    joint_group: {"min": np.array(stats[lo]), "max": np.array(stats[hi]), "dim": len(stats["min"])}
                    for joint_group, stats in statistics[modality].items()
                }
            for key in self._relative_keys(embodiment_tag):
                assert "relative_action" in statistics and key in statistics["relative_action"], (
                    f"relative_action statistics for '{key}' ({embodiment_tag}) missing — fred/data/stats.py writes them"
                )
                stats = statistics["relative_action"][key]
                assert lo in stats and hi in stats, (
                    f"relative_action stats for '{key}' ({embodiment_tag}) carry no {lo}/{hi} — rebuild them"
                )
                params["action"][key] = {"min": np.asarray(stats[lo]), "max": np.asarray(stats[hi]), "dim": params["action"][key]["dim"]}
            self.norm_params[embodiment_tag] = params

    def apply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """Per joint group: sin/cos encoding ((..., D) -> (..., 2D)) or min/max to [-1, 1] (clipped)."""
        state_config = self.modality_configs[embodiment_tag]["state"]
        sin_cos_keys = (state_config.sincos_encoding_keys or []) if self.apply_sincos_state_encoding else []
        normalized_values = {}
        for joint_group in state_config.modality_keys:
            assert joint_group in state, f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
            if joint_group in sin_cos_keys:
                normalized_values[joint_group] = apply_sincos_encoding(state[joint_group])
            else:
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized_values[joint_group] = np.clip(normalize_values_minmax(state[joint_group], params), -1.0, 1.0)
        return normalized_values

    def apply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Relative keys become action − state[key][-1] (needs the raw ``state``), then every group is min/max
        normalized to [-1, 1] (clipped). Shapes per group: (T, D)."""
        action = dict(action)
        for key in self._relative_keys(embodiment_tag):
            assert state is not None and key in state, (
                f"Reference state '{key}' required for the relative action key '{key}' in embodiment '{embodiment_tag}'"
            )
            reference_state = np.asarray(state[key][-1], dtype=np.float64)   # current-frame anchor
            values = np.asarray(action[key], dtype=np.float64)
            assert values.ndim == 2 and reference_state.shape == values.shape[1:], (
                f"Expected action (T, D) and state (D,), got {values.shape} and {reference_state.shape}"
            )
            action[key] = values - reference_state

        normalized_values = {}
        for joint_group in self.modality_configs[embodiment_tag]["action"].modality_keys:
            assert joint_group in action, f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
            params = self.norm_params[embodiment_tag]["action"][joint_group]
            normalized_values[joint_group] = np.clip(normalize_values_minmax(action[joint_group], params), -1.0, 1.0)
        return normalized_values

    def unapply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Inverse of apply_action: denormalize, then add the raw state anchor state[key][..., -1, :] back to the
        relative keys. Actions are (T, D) or batched (B, T, D) with (T_state, D) / (B, T_state, D) states."""
        unnormalized_values = {}
        for joint_group in self.modality_configs[embodiment_tag]["action"].modality_keys:
            assert joint_group in action, f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
            params = self.norm_params[embodiment_tag]["action"][joint_group]
            unnormalized_values[joint_group] = unnormalize_values_minmax(action[joint_group], params)

        for key in self._relative_keys(embodiment_tag):
            assert state is not None and key in state, (
                f"Reference state '{key}' required for the relative action key '{key}' in embodiment '{embodiment_tag}'"
            )
            relative = np.asarray(unnormalized_values[key], dtype=np.float64)
            reference_state = np.asarray(state[key], dtype=np.float64)[..., -1, :]   # (D,) or (B, D)
            assert relative.ndim in (2, 3) and reference_state.shape == relative.shape[:-2] + relative.shape[-1:], (
                f"State anchor {reference_state.shape} does not match action {relative.shape}"
            )
            unnormalized_values[key] = relative + reference_state[..., None, :]
        return unnormalized_values

    def apply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """apply_state + apply_action (the raw state is the relative anchor); an empty action is only allowed in
        eval mode."""
        processed_state = self.apply_state(state, embodiment_tag)
        if action:
            processed_action = self.apply_action(action, embodiment_tag, state=state)
        else:
            assert not self.training, "Action is required in training mode"
            processed_action = {}
        return processed_state, processed_action
