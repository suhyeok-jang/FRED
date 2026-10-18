import importlib
import sys
from pathlib import Path

from fred.data.embodiment_tags import EmbodimentTag


MODALITY_CONFIGS: dict[str, dict] = {}   # filled by the modality config modules (fred/configs/data/*_config.py)


def register_modality_config(
    config: dict, embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
):
    assert embodiment_tag.value not in MODALITY_CONFIGS, (
        f"Embodiment tag {embodiment_tag} already registered"
    )
    MODALITY_CONFIGS[embodiment_tag.value] = config


def load_modality_config(modality_config_path: str):
    """Import a user-provided modality config Python file so its registrations take effect."""
    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"[i] Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")
