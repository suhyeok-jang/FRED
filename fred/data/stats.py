#!/usr/bin/env python
"""Dataset statistics for LeRobot datasets: meta/stats.json (per-column mean/std/min/max/q01/q99 of every float
feature) and meta/relative_stats.json (the same stats of each RELATIVE action key, taken relative to the
current state, shape (horizon, D)).

Usage: python fred/data/stats.py <dataset_path> <embodiment_tag>
The tag's modality config must be registered in MODALITY_CONFIGS (import one of fred/configs/data/*_config.py).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from fred.configs.data.embodiment_configs import MODALITY_CONFIGS
from fred.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from fred.data.types import ActionRepresentation, ActionType, EmbodimentTag, ModalityConfig
from fred.data.utils import to_json_serializable


LEROBOT_PARQUET_GLOB = "data/*/*.parquet"
LEROBOT_INFO_PATH = "meta/info.json"
LEROBOT_STATS_PATH = "meta/stats.json"
LEROBOT_RELATIVE_STATS_PATH = "meta/relative_stats.json"


def calculate_dataset_statistics(
    parquet_paths: list[Path], features: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """Per-column mean/std/min/max/q01/q99 over all rows of the parquet files (all columns when ``features`` is
    None)."""
    all_low_dim_data_list = []
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        all_low_dim_data_list.append(pd.read_parquet(parquet_path))
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    dataset_statistics = {}
    if features is None:
        features = list(all_low_dim_data.columns)
    for le_modality in features:
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = dict(
            mean=np.mean(np_data, axis=0).tolist(),
            std=np.std(np_data, axis=0).tolist(),
            min=np.min(np_data, axis=0).tolist(),
            max=np.max(np_data, axis=0).tolist(),
            q01=np.quantile(np_data, 0.01, axis=0).tolist(),
            q99=np.quantile(np_data, 0.99, axis=0).tolist(),
        )
    return dataset_statistics


def check_stats_validity(dataset_path: Path | str, features: list[str]):
    stats_path = Path(dataset_path) / LEROBOT_STATS_PATH
    if not stats_path.exists():
        return False
    with open(stats_path, "r") as f:
        stats = json.load(f)
    for feature in features:
        if feature not in stats:
            return False
        if not isinstance(stats[feature], dict):
            return False
        for stat in ["mean", "std", "min", "max", "q01", "q99"]:
            if stat not in stats[feature]:
                return False
    return True


def generate_stats(dataset_path: Path | str):
    dataset_path = Path(dataset_path)
    print(f"Generating stats for {str(dataset_path)}")
    lowdim_features = []
    with open(dataset_path / LEROBOT_INFO_PATH, "r") as f:
        le_features = json.load(f)["features"]
    for feature in le_features:
        if "float" in le_features[feature]["dtype"]:
            lowdim_features.append(feature)
    if check_stats_validity(dataset_path, lowdim_features):
        return

    parquet_files = list(dataset_path.glob(LEROBOT_PARQUET_GLOB))
    stats = calculate_dataset_statistics(parquet_files, lowdim_features)
    stats_path = dataset_path / LEROBOT_STATS_PATH
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)


class RelativeActionLoader:
    """Per-episode relative action chunks (action − state at the chunk's first index) of one NON_EEF action key."""

    def __init__(self, dataset_path: Path | str, embodiment_tag: EmbodimentTag, action_key: str):
        self.dataset_path = Path(dataset_path)
        self.action_key = action_key
        config = MODALITY_CONFIGS[embodiment_tag.value]
        assert action_key in config["action"].modality_keys and action_key in config["state"].modality_keys
        assert config["action"].action_configs is not None, config["action"]
        action_config = config["action"].action_configs[config["action"].modality_keys.index(action_key)]
        assert action_config.type == ActionType.NON_EEF, f"relative stats need a NON_EEF key, got {action_config}"
        self.modality_configs = {
            "action": ModalityConfig(delta_indices=config["action"].delta_indices, modality_keys=[action_key]),
            "state": ModalityConfig(delta_indices=config["state"].delta_indices, modality_keys=[action_key]),
        }
        # the relative anchor is the state at the first action index
        assert config["state"].delta_indices[-1] == config["action"].delta_indices[0]
        self.loader = LeRobotEpisodeLoader(dataset_path, self.modality_configs)

    def load_relative_actions(self, trajectory_id: int) -> list[np.ndarray]:
        """(horizon, D) float32 relative chunks for every usable step of the episode."""
        df = self.loader[trajectory_id]
        state_data = df[f"state.{self.action_key}"].values  # (episode_length,) of (D,)
        action_data = df[f"action.{self.action_key}"].values
        trajectories = []
        usable_length = len(df) - self.modality_configs["action"].delta_indices[-1]
        action_delta_indices = np.array(self.modality_configs["action"].delta_indices)
        for i in range(usable_length):
            state_ind = self.modality_configs["state"].delta_indices[-1] + i
            last_state = np.asarray(state_data[state_ind], dtype=np.float64)
            actions = np.stack(list(action_data[action_delta_indices + i])).astype(np.float64)
            trajectories.append((actions - last_state).astype(np.float32))
        return trajectories

    def __len__(self) -> int:
        return len(self.loader)


def calculate_stats_for_key(
    dataset_path: Path | str,
    embodiment_tag: EmbodimentTag,
    group_key: str,
    max_episodes: int = -1,
) -> dict:
    loader = RelativeActionLoader(dataset_path, embodiment_tag, group_key)
    trajectories = []
    for episode_id in tqdm(range(len(loader)), desc=f"Loading trajectories for key {group_key}"):
        if max_episodes != -1 and episode_id >= max_episodes:
            break
        trajectories.extend(loader.load_relative_actions(episode_id))
    return {
        "max": np.max(trajectories, axis=0),
        "min": np.min(trajectories, axis=0),
        "q01": np.quantile(trajectories, 0.01, axis=0),
        "q99": np.quantile(trajectories, 0.99, axis=0),
        "mean": np.mean(trajectories, axis=0),
        "std": np.std(trajectories, axis=0),
    }


def generate_rel_stats(dataset_path: Path | str, embodiment_tag: EmbodimentTag) -> None:
    dataset_path = Path(dataset_path)
    action_config = MODALITY_CONFIGS[embodiment_tag.value]["action"]
    if action_config.action_configs is None:
        return
    action_keys = [
        key
        for key, action_config in zip(action_config.modality_keys, action_config.action_configs)
        if action_config.representation == ActionRepresentation.RELATIVE
    ]
    if not action_keys:
        # nothing to compute: return before touching meta/ so read-only shared datasets never need write access
        return
    stats_path = Path(dataset_path) / LEROBOT_RELATIVE_STATS_PATH
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        stats = {}
    computed_any = False
    for action_key in sorted(action_keys):
        if action_key in stats:
            continue
        print(f"Generating relative stats for {dataset_path} {embodiment_tag} {action_key}")
        stats[action_key] = calculate_stats_for_key(dataset_path, embodiment_tag, action_key)
        computed_any = True
    if not computed_any:
        # everything already persisted: skip the byte-identical rewrite (read-only shared datasets)
        return
    with open(stats_path, "w") as f:
        json.dump(to_json_serializable(dict(stats)), f, indent=4)


def main(dataset_path: Path | str, modality_config_path: str, embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT):
    """Write <dataset>/meta/stats.json and relative_stats.json for the modality config registered by the given file."""
    from fred.configs.data.embodiment_configs import load_modality_config

    load_modality_config(modality_config_path)
    generate_stats(dataset_path)
    generate_rel_stats(dataset_path, embodiment_tag)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
