from dataclasses import dataclass, field
import json
from pathlib import Path
import yaml

from .data.data_config import DataConfig, SingleDatasetConfig
from .model.wam import WAMConfig
from .training.training_config import TrainingConfig


@dataclass
class Config:
    """Complete configuration: model + data + training. Saved as experiment_cfg/config.yaml (python-tagged yaml)."""

    model: WAMConfig = field(default_factory=WAMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def save(self, path: Path):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load a saved config.yaml (a Config object, or a plain {model, data, training} dict)."""
        data = yaml.load(Path(path).read_text(), Loader=yaml.Loader)
        if isinstance(data, Config):
            return data
        assert isinstance(data, dict), f"Invalid config file: {path}"
        config = cls()
        if "model" in data:
            config.model = WAMConfig(**data["model"])
        if "data" in data:
            config.data = DataConfig(**data["data"])
            config.data.datasets = [SingleDatasetConfig(**ds) if isinstance(ds, dict) else ds for ds in config.data.datasets]
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        return config

    def get_deepspeed_config(self) -> dict:
        """ZeRO-2 bf16 config (fred/configs/deepspeed/zero2_config.json) + optional universal-checkpoint load."""
        config = json.load(open(Path(__file__).parent / "deepspeed/zero2_config.json"))

        # universal checkpoint: rank-sharded ZeRO state cannot be reshaped, the universal format can (world-size change on resume)
        if self.training.deepspeed_load_universal:
            config.setdefault("checkpoint", {})["load_universal"] = True

        return config

    def validate(self):
        """Every dataset names a registered embodiment tag."""
        assert self.data.datasets, "no datasets configured"
        for d_cfg in self.data.datasets:
            assert d_cfg.embodiment_tag in self.data.modality_configs, (
                f"embodiment_tag {d_cfg.embodiment_tag!r} of dataset {d_cfg.dataset_paths} has no modality config"
            )
