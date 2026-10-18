from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np

from fred.data.types import ModalityConfig


def apply_sincos_encoding(values: np.ndarray) -> np.ndarray:
    """(..., D) -> (..., 2D): [sin(v_1..v_D), cos(v_1..v_D)] on the last axis (doubles the dimension)."""
    sin_values = np.sin(values)
    cos_values = np.cos(values)
    return np.concatenate([sin_values, cos_values], axis=-1)


def nested_dict_to_numpy(data):
    """Recursively convert the leaf lists of a nested dict to numpy arrays."""
    if isinstance(data, dict):
        return {key: nested_dict_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return np.array(data)
    else:
        return data


def normalize_values_minmax(values, params):
    """Linear map from [min, max] to [-1, 1] over the last axis of (T, D) / (B, T, D) values; bounds are
    (D,) or per-step (T, D) (relative-action stats). Features with min == max map to 0."""
    min_vals = params["min"]
    max_vals = params["max"]
    normalized = np.zeros_like(values)

    mask = ~np.isclose(max_vals, min_vals)

    normalized[..., mask] = (values[..., mask] - min_vals[..., mask]) / (
        max_vals[..., mask] - min_vals[..., mask]
    )
    normalized[..., mask] = 2 * normalized[..., mask] - 1

    return normalized


def unnormalize_values_minmax(normalized_values, params):
    """Inverse of normalize_values_minmax; the input is clipped to [-1, 1] first."""

    min_vals = params["min"]
    max_vals = params["max"]
    range_vals = max_vals - min_vals

    unnormalized = (np.clip(normalized_values, -1.0, 1.0) + 1.0) / 2.0 * range_vals + min_vals
    return unnormalized


def to_json_serializable(obj: Any) -> Any:
    """Recursively convert dataclasses, numpy arrays/scalars, sets and enums (by name) to JSON-serializable
    values; any other type falls back to str()."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_json_serializable(asdict(obj))
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, Enum):
        return obj.name
    else:
        return str(obj)


def parse_modality_configs(
    modality_configs: dict[str, dict[str, ModalityConfig]],
) -> dict[str, dict[str, ModalityConfig]]:
    parsed_modality_configs = {}
    for embodiment_tag, modality_config in modality_configs.items():
        parsed_modality_configs[embodiment_tag] = {}
        for modality, config in modality_config.items():
            if isinstance(config, dict):
                parsed_modality_configs[embodiment_tag][modality] = ModalityConfig(**config)
            else:
                parsed_modality_configs[embodiment_tag][modality] = config
    return parsed_modality_configs
