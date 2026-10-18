from dataclasses import dataclass
from enum import Enum

import numpy as np

from fred.data.embodiment_tags import EmbodimentTag


class ActionRepresentation(Enum):
    RELATIVE = "relative"
    DELTA = "delta"
    ABSOLUTE = "absolute"


class ActionType(Enum):
    EEF = "eef"
    NON_EEF = "non_eef"


class ActionFormat(Enum):
    DEFAULT = "default"


@dataclass
class StepData:
    """One raw step as returned by the datasets (observations + action chunk), consumed by WAMProcessor."""

    images: dict[str, list[np.ndarray]]  # view_name -> list[np.ndarray] (for temporal stacking)
    states: dict[str, np.ndarray]  # state_name -> np.ndarray (horizon, dim)
    actions: dict[str, np.ndarray]  # action_name -> np.ndarray (horizon, dim) for action chunk
    text: str | None = None  # Optional task description or instruction
    embodiment: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT  # modality-config / statistics key


@dataclass
class ActionConfig:
    representation: ActionRepresentation
    type: ActionType
    format: ActionFormat


@dataclass
class ModalityConfig:
    """Which delta indices to sample relative to the base index and which keys to load for one modality (video,
    state, action, language)."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""
    sincos_encoding_keys: list[str] | None = None
    """Optional list of state keys to sin/cos-encode (with data.apply_sincos_state_encoding); the rest are min/max normalized."""
    action_configs: list[ActionConfig] | None = None

    def __post_init__(self):
        """Parse dict-form action_configs (from yaml/json) into ActionConfig and require one per modality key."""
        if self.action_configs is not None:
            assert len(self.action_configs) == len(self.modality_keys), (
                f"Number of action configs ({len(self.action_configs)}) must match number of modality keys ({len(self.modality_keys)})"
            )
            parsed_action_configs = []
            for action_config in self.action_configs:
                if isinstance(action_config, dict):
                    action_config = ActionConfig(
                        representation=ActionRepresentation[action_config["representation"]],
                        type=ActionType[action_config["type"]],
                        format=ActionFormat[action_config["format"]],
                    )
                parsed_action_configs.append(action_config)
            self.action_configs = parsed_action_configs
