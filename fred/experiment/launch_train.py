"""Train WAMModel.

    python fred/experiment/launch_train.py --dataset-paths <dir> [<dir> ...] --modality-config-path fred/configs/data/<x>_config.py \
        [--dataset-mix-ratios ...] [--base-model-path <ckpt>] [--model.<field> ...] [--data.<field> ...] [--training.<field> ...]

The three groups are WAMConfig / DataConfig / TrainingConfig themselves (fred/configs); run_scripts/train/train_fred.sh
assembles the flags from its environment variables.
"""
import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from huggingface_hub import snapshot_download

from fred.configs.base_config import Config
from fred.configs.data.data_config import DataConfig, SingleDatasetConfig
from fred.configs.data.embodiment_configs import load_modality_config
from fred.configs.model.wam import WAMConfig
from fred.configs.training.training_config import TrainingConfig
from fred.data.embodiment_tags import EmbodimentTag
from fred.experiment.experiment import run

logger = logging.getLogger(__name__)

# WAMConfig fields that define the architecture: a warm start must agree with the checkpoint on these
MODEL_ARCHITECTURE_KEYS = [
    "backbone_type", "wan_feature_extraction_layer",
    "num_frames_in", "num_frames_out", "frame_downsample_factor",
    "action_head_hidden_size", "input_embedding_dim", "action_dit_size", "action_dit_cfg",
    "max_state_dim", "max_action_dim", "action_horizon", "selfdistill_num_prototypes",
]


@dataclass
class TrainCLI:
    dataset_paths: list[str]
    """LeRobot dataset roots (meta/, data/, videos/); one sharded dataset each."""
    modality_config_path: str
    """Python file registering the modality config (fred/configs/data/*_config.py)."""
    dataset_mix_ratios: list[float] | None = None
    """Shard-sampling weight per dataset (default: equal)."""
    base_model_path: str | None = None
    """Warm start: checkpoint directory or HF repo id whose weights initialise the model (step 0, fresh optimizer)."""
    allow_config_changes: bool = False
    """Let the CLI model fields differ from the warm-start checkpoint's architecture."""
    model: WAMConfig = field(default_factory=WAMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def resolve_warm_start(base_model_path: str, model: WAMConfig, allow_changes: bool) -> str:
    """Local dir or downloaded HF snapshot; refuses architecture fields that differ from its experiment_cfg/config.yaml."""
    if "/" in base_model_path and not Path(base_model_path).exists():
        base_model_path = snapshot_download(repo_id=base_model_path, repo_type="model", local_dir=None)
    exp_cfg = Path(base_model_path) / "experiment_cfg" / "config.yaml"
    if not exp_cfg.exists():
        logger.warning(f"{exp_cfg} does not exist; architecture not checked against the checkpoint")
        return base_model_path
    ckpt_model = Config().load(exp_cfg).model
    diffs = {k: (copy.deepcopy(getattr(ckpt_model, k, None)), getattr(model, k)) for k in MODEL_ARCHITECTURE_KEYS
             if getattr(ckpt_model, k, None) != getattr(model, k)}
    if diffs:
        msg = "model config differs from the checkpoint:\n" + "\n".join(f"  {k}: {old} -> {new}" for k, (old, new) in diffs.items())
        assert allow_changes, msg + "\nPass --allow-config-changes to override the checkpoint's architecture."
        logger.warning(msg)
    return base_model_path


def build_run_config(cli: TrainCLI) -> Config:
    """CLI groups -> the run Config: datasets, modality config registration, video delta indices, warm-start check."""
    load_modality_config(cli.modality_config_path)
    ratios = cli.dataset_mix_ratios if cli.dataset_mix_ratios is not None else [1.0] * len(cli.dataset_paths)
    assert len(ratios) == len(cli.dataset_paths), f"dataset_mix_ratios ({len(ratios)}) must match dataset_paths ({len(cli.dataset_paths)})"
    assert all(r > 0 for r in ratios), "every dataset_mix_ratio must be > 0"
    cli.data.datasets = [SingleDatasetConfig(dataset_paths=[p], mix_ratio=r, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT.value)
                         for p, r in zip(cli.dataset_paths, ratios)]

    m = cli.model
    video_delta_indices = list(range(-(m.num_frames_in - 1), m.num_frames_out - m.num_frames_in + 1, m.frame_downsample_factor))
    for modality_config in cli.data.modality_configs.values():
        if "video" in modality_config:
            modality_config["video"].delta_indices = video_delta_indices
    logger.info(f"[i] video delta_indices: {video_delta_indices}")
    m.validate()
    if cli.base_model_path:
        cli.training.base_model_path = resolve_warm_start(cli.base_model_path, m, cli.allow_config_changes)
    return Config(model=m, data=cli.data, training=cli.training)


if __name__ == "__main__":
    run(build_run_config(tyro.cli(TrainCLI, description=__doc__)))
