import numpy as np
import torch

from fred.configs.base_config import Config
from fred.data.dataset.sharded_mixture_dataset import ExactRobotHumanMixDataset, ShardedMixtureDataset
from fred.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from fred.data.dataset.robot_human_mix import parse_batch_group_fractions
from fred.data.embodiment_tags import EmbodimentTag
from fred.data.stats import generate_rel_stats, generate_stats
from fred.utils.dist import barrier


class DatasetFactory:
    """Datasets from one run Config: every dataset becomes a ShardedSingleStepDataset (episodes pre-split into
    fixed-size shards, loaded in bulk and prefetched) and the mix is a ShardedMixtureDataset weighted by
    mix_ratio × relative length (ExactRobotHumanMixDataset when ``data.batch_group_fractions`` fixes the robot:human split)."""

    def __init__(self, config: Config):
        self.config = config

    def _ensure_stats(self, dataset_path: str, embodiment_tag: str) -> None:
        """Generate dataset statistics if not already present (rank-0 only, then barrier)."""
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            generate_stats(dataset_path)
            generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
        barrier()

    def build(self, processor) -> ShardedMixtureDataset:
        d = self.config.data
        all_datasets, all_weights = [], []
        for spec in d.datasets:
            datasets = []
            for dataset_path in spec.dataset_paths:
                self._ensure_stats(dataset_path, spec.embodiment_tag)
                datasets.append(ShardedSingleStepDataset(
                    dataset_path=dataset_path,
                    embodiment_tag=EmbodimentTag(spec.embodiment_tag),
                    modality_configs=d.modality_configs[spec.embodiment_tag],
                    stats_file=d.stats_file,
                    relative_stats_file=d.relative_stats_file,
                    human_datasets=d.human_datasets,
                    dataset_action_groups=d.dataset_action_groups,
                    shard_size=d.shard_size,
                    episode_sampling_rate=d.episode_sampling_rate,
                    seed=d.seed,
                ))
            lengths = np.array([len(ds) for ds in datasets], dtype=np.float64)
            for ds, rel in zip(datasets, lengths / lengths.sum()):
                all_datasets.append(ds)
                all_weights.append(rel * spec.mix_ratio)

        common = dict(datasets=all_datasets, weights=all_weights, processor=processor, seed=d.seed,
                      num_shards_per_epoch=d.num_shards_per_epoch, shard_prefetch_timeout_s=d.shard_prefetch_timeout_s)
        batch_group_fractions = parse_batch_group_fractions(d.batch_group_fractions)
        if batch_group_fractions is None:
            return ShardedMixtureDataset(**common)
        # exact per-batch robot:human split (OpenArm co-training)
        return ExactRobotHumanMixDataset(
            is_human=[ds.is_human for ds in all_datasets],
            per_device_batch_size=self.config.training.global_batch_size // self.config.training.num_gpus,
            human_frac=batch_group_fractions["human"], **common,
        )
