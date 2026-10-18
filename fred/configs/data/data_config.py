from dataclasses import dataclass, field
from typing import Annotated

import tyro

from fred.data.embodiment_tags import EmbodimentTag
from fred.data.types import ModalityConfig

from .embodiment_configs import MODALITY_CONFIGS


@dataclass
class SingleDatasetConfig:
    dataset_paths: list[str]
    embodiment_tag: str = EmbodimentTag.NEW_EMBODIMENT.value
    mix_ratio: float = 1.0


@dataclass
class DataConfig:
    """Datasets, sharded sampling, and the processor settings persisted with the checkpoint."""

    datasets: Annotated[list[SingleDatasetConfig], tyro.conf.Suppress] = field(default_factory=list)
    # the registry object itself: the launcher registers --modality-config-path into it and rewrites the video delta_indices
    modality_configs: Annotated[dict[str, dict[str, ModalityConfig]], tyro.conf.Suppress] = field(default_factory=lambda: MODALITY_CONFIGS)

    # sharded sampling
    shard_size: int = 2**10
    episode_sampling_rate: float = 0.1   # episodes split into 1/rate interleaved groups so one shard mixes many episodes
    num_shards_per_epoch: int = int(1e5)
    seed: int = 42

    # loading
    shard_prefetch_timeout_s: int = 1800    # prefetch deadline; fails before the NCCL watchdog would
    multiprocessing_context: str = "forkserver"
    prefetch_factor: int | None = None

    # state / action normalization
    use_percentiles: bool = False        # q01/q99 bounds instead of min/max
    use_relative_action: bool = False    # RELATIVE action keys become action − current state
    apply_sincos_state_encoding: bool = False
    stats_file: str = "stats.json"
    relative_stats_file: str = "relative_stats.json"

    # video
    video_resolution: int = 224
    video_resolution_hw: str | None = None   # "HxW" rectangle instead of the square
    video_crop_aspect: bool = False
    multiview_layout: str = "hconcat"    # hconcat | tmosaic | vconcat
    image_resize_backend: str = "cv2_linear"   # cv2_linear | torch_bilinear
    image_aug: bool = True
    image_aug_rotate_deg: float = 5.0
    image_aug_brightness: float = 0.2
    image_aug_contrast: float = 0.2
    image_aug_saturation: float = 0.2
    image_aug_hue: float = 0.05
    view_mask_probs: str | None = None   # "p_none,p_first,p_second"

    # language
    formalize_language: bool = True
    text_dropout: float = 0.1

    # robot + human co-training
    human_datasets: bool = False         # honour meta/wam_human.json
    batch_group_fractions: str | None = None         # "robot:0.5,human:0.5"
    dataset_action_groups: bool = False  # honour meta/wam_action_groups.json
