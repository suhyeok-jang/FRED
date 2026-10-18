from enum import Enum


class EmbodimentTag(Enum):
    """Key under which a modality config and its normalization statistics are registered. One dataset family per run,
    so every recipe (kitchen / GR-1 / LIBERO / DROID / OpenArm) uses the same tag."""

    NEW_EMBODIMENT = "new_embodiment"
