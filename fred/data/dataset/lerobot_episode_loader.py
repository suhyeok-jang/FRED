#!/usr/bin/env python
"""Episode-level loader for LeRobot-format datasets: parses meta/ (info, episodes, tasks, modality, stats),
decodes video, and returns one episode as a DataFrame with video.<key> / state.<key> / action.<key> /
language.<key> columns (ShardedSingleStepDataset slices it into steps)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fred.data.types import ModalityConfig

from .robot_human_mix import read_action_groups_marker, read_human_marker


# LeRobot standard metadata filenames
LEROBOT_META_DIR_NAME = "meta"
LEROBOT_INFO_FILENAME = "info.json"
LEROBOT_EPISODES_FILENAME = "episodes.jsonl"
LEROBOT_TASKS_FILENAME = "tasks.jsonl"
LEROBOT_MODALITY_FILENAME = "modality.json"
LEROBOT_STATS_FILE_NAME = "stats.json"
LEROBOT_RELATIVE_STATS_FILE_NAME = "relative_stats.json"

ALLOWED_MODALITIES = ["video", "state", "action", "language"]
DEFAULT_COLUMN_NAMES = {
    "state": "observation.state",
    "action": "action",
}


def get_frames_by_indices(video_path: str, indices: list[int] | np.ndarray) -> np.ndarray:
    """Decode the given frame indices with torchcodec (seek-based; one decoder per call). Returns [N, H, W, C] uint8."""
    from torchcodec.decoders import VideoDecoder  # noqa: PLC0415

    decoder = VideoDecoder(video_path, device="cpu", dimension_order="NHWC")
    try:
        return decoder.get_frames_at(indices=list(indices)).data.numpy()
    finally:
        del decoder


class LeRobotEpisodeLoader:
    """Loads one LeRobot episode at a time (``loader[i]`` -> DataFrame) for the keys named in ``modality_configs``
    (video / state / action / language) and exposes the sliced per-key statistics for normalization.

    Example:
        >>> loader = LeRobotEpisodeLoader(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(delta_indices=list(range(16)), modality_keys=["joint_velocities"]),
        ...     },
        ... )
        >>> episode_data = loader[0]
    """

    def __init__(
        self,
        dataset_path: str | Path,
        modality_configs: dict[str, ModalityConfig],
        stats_file: str = LEROBOT_STATS_FILE_NAME,
        relative_stats_file: str = LEROBOT_RELATIVE_STATS_FILE_NAME,
        human_datasets: bool = False,
        dataset_action_groups: bool = False,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.stats_file, self.relative_stats_file = stats_file, relative_stats_file
        self.human_datasets, self.dataset_action_groups = bool(human_datasets), bool(dataset_action_groups)

        assert self.dataset_path.is_dir(), f"Dataset path does not exist: {self.dataset_path}"

        self._load_metadata()
        # language: >= 1 key (several keys are allowed, e.g. DROID's three instructions, and extract_step_data
        # picks one per sample) and delta_indices == [0]
        for modality, config in modality_configs.items():
            assert modality in ALLOWED_MODALITIES, f"Invalid modality: {modality}"
            if modality == "language":
                assert len(config.modality_keys) >= 1, "Language modality must have at least one key"
                assert config.delta_indices == [0], "Only single timestep is supported for language modality"
        self.modality_configs = modality_configs
        self.episode_lengths = [int(ep_meta["length"]) for ep_meta in self.episodes_metadata]

    def _load_metadata(self) -> None:
        """Read meta/: info.json (file patterns), episodes.jsonl, tasks.jsonl, modality.json (key -> column slice),
        stats.json (+ relative_stats.json when present), and the human / action-group markers."""
        meta_dir = self.dataset_path / LEROBOT_META_DIR_NAME

        with open(meta_dir / LEROBOT_INFO_FILENAME, "r") as f:
            self.info_meta = json.load(f)

        with open(meta_dir / LEROBOT_EPISODES_FILENAME, "r") as f:
            self.episodes_metadata = [json.loads(line) for line in f]

        with open(meta_dir / LEROBOT_TASKS_FILENAME, "r") as f:
            tasks_data = [json.loads(line) for line in f]
            self.tasks_map = {task["task_index"]: task["task"] for task in tasks_data}

        with open(meta_dir / LEROBOT_MODALITY_FILENAME, "r") as f:
            self.modality_meta = json.load(f)

        stats_path = meta_dir / self.stats_file
        assert stats_path.exists(), f"{stats_path} does not exist for {self.dataset_path}, please use fred/data/stats.py to generate it"
        with open(stats_path, "r") as f:
            self.stats = json.load(f)

        relative_stats_path = meta_dir / self.relative_stats_file
        if relative_stats_path.exists():
            with open(relative_stats_path, "r") as f:
                self.stats["relative_action"] = json.load(f)

        # human datasets (data.human_datasets): meta/wam_human.json marks a video-only member (no action loss,
        # zero state, missing views black; see robot_human_mix.py). Without the flag such a dataset is refused
        # rather than silently trained on its zero-filled actions.
        marked_human = read_human_marker(self.dataset_path)
        self.is_human = bool(marked_human and self.human_datasets)
        assert not (marked_human and not self.human_datasets), (
            f"{self.dataset_path} carries meta/wam_human.json (human, video-only dataset) but "
            f"data.human_datasets is off — pass --data.human-datasets or drop the dataset."
        )

        # action groups (data.dataset_action_groups): meta/wam_action_groups.json lists the groups the dataset
        # really supplies; the rest are zero-filled and get masked per sample in sharded_single_step_dataset.py.
        # Without the flag such a dataset is refused rather than trained to predict zeros.
        marked_groups = read_action_groups_marker(self.dataset_path)
        self.action_valid_groups = (
            list(marked_groups) if (marked_groups and self.dataset_action_groups) else None
        )
        assert not (marked_groups and not self.dataset_action_groups), (
            f"{self.dataset_path} carries meta/wam_action_groups.json (valid action groups "
            f"{marked_groups}; every other group is zero-filled) but data.dataset_action_groups is "
            f"off — pass --data.dataset-action-groups or drop the dataset. Training it as-is "
            f"would teach the model to predict zeros for the groups it does not have."
        )

        self.feature_config = self.info_meta["features"]
        self.data_path_pattern = self.info_meta["data_path"]
        self.video_path_pattern = self.info_meta.get("video_path")
        self.chunk_size = self.info_meta["chunks_size"]

    def __len__(self) -> int:
        """Return number of episodes in dataset."""
        return len(self.episodes_metadata)

    def _extract_joint_groups(
        self,
        df: pd.DataFrame,
        joint_groups: list[str],
        modality_type: str,
    ) -> pd.DataFrame:
        """One column per joint group, sliced [start:end] out of the raw state/action column per modality.json."""
        modality_info = self.modality_meta[modality_type]
        joint_data = pd.DataFrame()

        for group_name in joint_groups:
            assert group_name in modality_info, (
                f"{self.dataset_path}: joint group '{group_name}' not in {modality_type} of modality.json "
                f"(available: {list(modality_info)})"
            )
            group_info = modality_info[group_name]
            start_idx = group_info["start"]
            end_idx = group_info["end"]
            original_key = group_info.get("original_key", DEFAULT_COLUMN_NAMES[modality_type])
            if isinstance(df[original_key].iloc[0], np.ndarray):
                joint_data[group_name] = df[original_key].map(lambda x: x[start_idx:end_idx])
            else:
                joint_data[group_name] = df[original_key]  # for strings and scalars

        return joint_data

    def _load_parquet_data(self, episode_index: int) -> pd.DataFrame:
        """Episode parquet -> DataFrame with language.<key> (task index -> string) and the state/action joint-group
        columns."""
        chunk_idx = episode_index // self.chunk_size
        parquet_filename = self.data_path_pattern.format(
            episode_chunk=chunk_idx, episode_index=episode_index
        )
        parquet_path = self.dataset_path / parquet_filename
        original_df = pd.read_parquet(parquet_path)
        loaded_df = pd.DataFrame()

        if "language" in self.modality_configs:
            for key in self.modality_configs["language"].modality_keys:
                assert key.startswith("annotation.")
                subkey = key.replace("annotation.", "")
                assert subkey in self.modality_meta["annotation"], (
                    f"Dataset: {self.dataset_path} parquet {parquet_filename}; Key {subkey} not found in language modality: {self.modality_meta['annotation'].keys()=}"
                )
                original_key = self.modality_meta["annotation"][subkey].get("original_key", key)
                loaded_df[f"language.{key}"] = original_df[original_key].apply(
                    lambda x: self.tasks_map[x]
                )

        for modality_type in ["state", "action"]:
            if modality_type not in self.modality_configs:
                continue
            joint_groups_df = self._extract_joint_groups(
                original_df, self.modality_configs[modality_type].modality_keys, modality_type
            )
            for joint_group in joint_groups_df.columns:
                loaded_df[f"{modality_type}.{joint_group}"] = joint_groups_df[joint_group]

        return loaded_df

    def _load_video_data(self, episode_index: int, indices: np.ndarray) -> dict[str, np.ndarray]:
        """Decode the given frame indices of every configured view: {view: (N, H, W, C) uint8}."""
        video_data = {}

        if not self.video_path_pattern or "video" not in self.modality_configs:
            return video_data

        chunk_idx = episode_index // self.chunk_size
        for image_key in self.modality_configs["video"].modality_keys:
            # human datasets: a configured view the dataset lacks is skipped; the processor renders it black
            if self.is_human and image_key not in self.modality_meta["video"]:
                continue
            original_key = self.modality_meta["video"][image_key].get(
                "original_key", f"observation.images.{image_key}"
            )
            assert original_key in self.feature_config, (
                f"Original key {original_key} not found in feature config"
            )

            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_idx, video_key=original_key, episode_index=episode_index
            )
            video_path = self.dataset_path / video_filename

            video_data[image_key] = get_frames_by_indices(str(video_path), indices)

        return video_data

    def get_dataset_statistics(self) -> dict[str, Any]:
        """{state|action: {joint_group: {mean|std|min|max|q01|q99: values}}}, sliced per group out of the raw
        column stats, plus "relative_action" verbatim when relative_stats.json exists."""
        stats: dict[str, Any] = {}
        for modality in ["state", "action"]:
            stats[modality] = {}
            for joint_key in self.modality_configs[modality].modality_keys:
                meta = self.modality_meta[modality][joint_key]
                stats_key = meta["original_key"] if meta.get("original_key") is not None else DEFAULT_COLUMN_NAMES[modality]
                start_idx, end_idx = meta["start"], meta["end"]
                stats[modality][joint_key] = {
                    stat_type: values[start_idx:end_idx] for stat_type, values in self.stats[stats_key].items()
                }
        if "relative_action" in self.stats:
            stats["relative_action"] = self.stats["relative_action"]
        return stats

    def __getitem__(self, idx: int) -> pd.DataFrame:
        """Whole episode as a DataFrame (parquet columns + one decoded uint8 frame per row per view)."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of bounds")

        episode_meta = self.episodes_metadata[idx]
        episode_id = episode_meta["episode_index"]
        nominal_length = episode_meta["length"]

        df = self._load_parquet_data(episode_id)

        # the parquet may be shorter than the nominal length
        actual_length = min(len(df), nominal_length)
        df = df.iloc[:actual_length]

        video_data = self._load_video_data(episode_id, np.arange(actual_length))

        for key in video_data.keys():
            assert len(video_data[key]) == len(df), (
                f"Video data for {key} has length {len(video_data[key])} but dataframe has length {len(df)}"
            )
            df[f"video.{key}"] = [frame for frame in video_data[key]]

        return df
