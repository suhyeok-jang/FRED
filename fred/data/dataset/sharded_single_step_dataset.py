from pathlib import Path
import random

import numpy as np
import pandas as pd

from fred.data.types import ActionRepresentation, EmbodimentTag, ModalityConfig, StepData

from .lerobot_episode_loader import LeRobotEpisodeLoader
from .robot_human_mix import action_group_column_mask


def extract_step_data(
    episode_data: pd.DataFrame,
    step_index: int,
    modality_configs: dict[str, ModalityConfig],
    embodiment_tag: EmbodimentTag,
    allow_missing_video: bool = False,
) -> StepData:
    """One step's modalities at step_index ± delta_indices. Indices past the episode edges are clamped to the first/last
    row (RELATIVE/DELTA action keys get zeros there instead of a repeated delta)."""
    step_data = {}

    for modality, config in modality_configs.items():
        step_data[modality] = {}
        raw_indices = [step_index + delta_index for delta_index in config.delta_indices]
        max_idx = len(episode_data) - 1

        out_of_bounds = [idx < 0 or idx > max_idx for idx in raw_indices]

        indices_to_load = [max(0, min(idx, max_idx)) for idx in raw_indices]
        # RELATIVE/DELTA action keys: an out-of-bounds (clamped) step gets zeros instead of a repeated delta
        relative_keys = set()
        if config.action_configs is not None:
            relative_keys = {
                key for key, action_config in zip(config.modality_keys, config.action_configs)
                if action_config.representation in (ActionRepresentation.RELATIVE, ActionRepresentation.DELTA)
            }

        for key in config.modality_keys:
            if f"{modality}.{key}" in episode_data.columns:
                modality_data = episode_data[f"{modality}.{key}"].iloc[indices_to_load]
            elif modality == "video" and allow_missing_video:
                # human datasets: a view absent from the dataset is omitted; the processor renders it black
                continue
            else:
                raise KeyError(
                    f"{modality}.{key} not found in episode data, available keys: {episode_data.columns}"
                )

            if modality in ["state", "action"]:
                is_relative = key in relative_keys
                arrays = []
                for j in range(len(modality_data)):
                    val = np.array(modality_data.iloc[j]).astype(np.float32)
                    if out_of_bounds[j] and is_relative:
                        val = np.zeros_like(val)
                    arrays.append(val)
                step_data[modality][key] = np.vstack(arrays)
            else:
                step_data[modality][key] = modality_data.tolist()

    video_data = step_data.get("video", {})
    state_data = step_data.get("state", {})
    action_data = step_data.get("action", {})
    language_data = step_data.get("language", {})
    assert len(language_data) >= 1, f"Expected >=1 language, got {len(language_data)}"
    if len(language_data) == 1:
        text = language_data[list(language_data.keys())[0]][0]
    else:
        # several instruction keys (DROID): pick one at random per sample
        text = language_data[random.choice(sorted(language_data.keys()))][0]

    step_data = StepData(
        images=video_data,
        states=state_data,
        actions=action_data,
        text=text,
        embodiment=embodiment_tag,
    )
    return step_data


class ShardedSingleStepDataset:
    """One LeRobot dataset pre-split into balanced shards of ~shard_size timesteps (each shard mixes steps from many
    episodes); get_shard(i) loads its episodes and returns the processed steps.

    Args:
        dataset_path: LeRobot dataset root
        embodiment_tag: modality-config / statistics key
        modality_configs: per-modality keys and delta_indices
        shard_size: target timesteps per shard
        episode_sampling_rate: each episode is split into 1/rate interleaved step groups spread over shards
        seed: shard assignment seed

    Example:
        >>> dataset = ShardedSingleStepDataset(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(
        ...             delta_indices=list(range(8)), modality_keys=["joint_velocities"]
        ...         ),
        ...     },
        ...     shard_size=1024,
        ...     episode_sampling_rate=0.1,
        ... )
        >>> shard_data = dataset.get_shard(0)  # Get first shard of processed timesteps
    """

    def __init__(
        self,
        dataset_path: str | Path,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        stats_file: str = "stats.json",
        relative_stats_file: str = "relative_stats.json",
        human_datasets: bool = False,
        dataset_action_groups: bool = False,
        shard_size: int = 2**10,  # 1024 steps
        episode_sampling_rate: float = 0.1,
        seed: int = 42,
    ):
        """Initialize single-step dataset with sharding configuration."""
        self.dataset_path = dataset_path
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.shard_size = shard_size
        self.episode_sampling_rate = episode_sampling_rate
        self.seed = seed
        self.processor = None
        self.rng = np.random.default_rng(seed)
        action_delta_indices = modality_configs["action"].delta_indices
        self.action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1

        self.episode_loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            stats_file=stats_file,
            relative_stats_file=relative_stats_file,
            human_datasets=human_datasets,
            dataset_action_groups=dataset_action_groups,
        )

        # human datasets: loader-derived flag (meta/wam_human.json); see get_datapoint
        self.is_human = self.episode_loader.is_human
        if self.is_human:
            print(f"[human_datasets] {self.dataset_path}: HUMAN (video-only) dataset — rows get "
                  f"is_human=1, state=0, action_mask=0; missing views rendered black.", flush=True)

        # action groups this dataset actually supplies (meta/wam_action_groups.json, None = all). The per-column
        # 0/1 mask is built lazily in get_datapoint (it needs max_action_dim, which only the processor knows).
        self.action_valid_groups = self.episode_loader.action_valid_groups
        self._action_col_mask = None       # torch tensor (D,), cached
        self._action_col_mask_dim = None
        if self.action_valid_groups:
            print(f"[dataset_action_groups] {self.dataset_path}: action groups "
                  f"{self.action_valid_groups} are the ONLY ones in the action loss; every other group "
                  f"is masked out per sample (action_mask=0 and action=0 in those columns).", flush=True)

        self.shard_dataset()

    def shard_dataset(self):
        """Shuffle episodes, split each one's steps into 1/episode_sampling_rate interleaved groups and assign
        every group to the currently shortest shard (greedy balancing)."""
        shuffled_episode_indices = self.rng.permutation(len(self.episode_loader.episode_lengths))
        num_splits = int(1 / self.episode_sampling_rate)

        assert len(shuffled_episode_indices) > 0, (
            f"No valid trajectories found for dataset {self.dataset_path}"
        )

        total_steps = np.sum(
            [self.get_effective_episode_length(idx) for idx in shuffled_episode_indices]
        ).astype(int)
        num_shards = np.ceil(total_steps / self.shard_size).astype(int)

        sharded_episodes = [[] for _ in range(num_shards)]
        shard_lengths = np.zeros(num_shards, dtype=int)

        for ep_idx in shuffled_episode_indices:
            step_indices = np.arange(0, self.get_effective_episode_length(ep_idx))
            self.rng.shuffle(step_indices)
            for i in range(num_splits):
                split_step_indices = step_indices[i::num_splits]
                shard_index = np.argmin(shard_lengths)
                sharded_episodes[shard_index].append((ep_idx, split_step_indices))
                shard_lengths[shard_index] += len(split_step_indices)

        assert all(shard_lengths[i] > 0 for i in range(num_shards)), (
            "All shards must have length greater than 0"
        )

        print(f"Generated {num_shards} shards for dataset {self.dataset_path}")
        print(
            f"Total steps: {total_steps}, average shard length: {total_steps / num_shards}, shard length std: {np.std(shard_lengths)}"
        )
        self.sharded_episodes = sharded_episodes
        self.shard_lengths = shard_lengths

    def get_effective_episode_length(self, episode_index: int) -> int:
        """Get the effective episode length accounting for action horizon."""
        original_length = self.episode_loader.episode_lengths[episode_index]
        return max(0, original_length - self.action_horizon + 1)

    def __len__(self):
        """Return the number of shards in the dataset."""
        return len(self.shard_lengths)

    def get_datapoint(self, episode_data: pd.DataFrame, step_index: int) -> dict:
        """One processed step: extract_step_data -> processor, then the human / action-group masking."""
        assert self.processor is not None, "Processor must be set before getting datapoints"
        step_data = extract_step_data(
            episode_data, step_index, self.modality_configs, self.embodiment_tag,
            allow_missing_video=self.is_human,
        )
        result = self.processor(step_data)

        # human datasets: tag every sample with is_human (0/1) and, for human rows, zero the normalized state and
        # the action_mask so the action FM loss (squared.sum() / action_mask.sum()) never sees them and the
        # pad-noise valid-dim derivation (any over the batch) keeps coming from robot rows.
        if self.episode_loader.human_datasets:
            result["is_human"] = int(self.is_human)
            if self.is_human:
                result["state"] = result["state"] * 0
                if "action_mask" in result:
                    result["action_mask"] = result["action_mask"] * 0
                if "action" in result:
                    result["action"] = result["action"] * 0

        # action groups: zero the columns of the groups this dataset does not supply, in action_mask (dropped from
        # the FM loss) and in action (the noisy trajectory is exactly 0 there, like the
        # padding columns).
        if self.action_valid_groups and "action_mask" in result:
            am = result["action_mask"]
            dim = int(am.shape[-1])
            if self._action_col_mask is None or self._action_col_mask_dim != dim:
                spans = {
                    k: (v["start"], v["end"])
                    for k, v in self.episode_loader.modality_meta["action"].items()
                }
                cols = action_group_column_mask(
                    action_keys=list(self.modality_configs["action"].modality_keys),
                    group_spans=spans,
                    valid_groups=self.action_valid_groups,
                    total_dim=dim,
                )
                # new_tensor: same dtype/device as action_mask — this module does not import torch.
                self._action_col_mask = am.new_tensor(cols)
                self._action_col_mask_dim = dim
            colmask = self._action_col_mask
            result["action_mask"] = am * colmask
            if "action" in result:
                result["action"] = result["action"] * colmask.to(dtype=result["action"].dtype)

        return result

    def get_shard_length(self, idx: int) -> int:
        """Get the number of timesteps in a specific shard."""
        return self.shard_lengths[idx]

    def get_shard(self, idx: int) -> list:
        """All processed timesteps of shard ``idx`` (each episode is loaded once)."""
        episodes = self.sharded_episodes[idx]
        datapoints = []
        for ep_idx, step_indices in episodes:
            episode_data = self.episode_loader[ep_idx]
            for step_index in step_indices:
                datapoints.append(self.get_datapoint(episode_data, int(step_index)))
        return datapoints

    def set_processor(self, processor):
        self.processor = processor

    def get_dataset_statistics(self) -> dict:
        """Get dataset statistics from the underlying episode loader."""
        return self.episode_loader.get_dataset_statistics()
