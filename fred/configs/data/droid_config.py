"""DROID modality config (DreamZero droid_relative recipe).

  - action: RELATIVE joint_position (7) + ABSOLUTE gripper_position (1); bounds are q01/q99 when
    data.use_percentiles is set (relative keys read meta/relative_stats.json)
  - language: one of the three instruction keys is picked at random per sample (extract_step_data)
  - view order [exterior_1 (left), exterior_2 (right), wrist] follows the tmosaic [left, right, wrist] convention
  - action horizon 24 (1.6 s at 15 fps); the video delta_indices are rewritten from the model's clip geometry
"""
from fred.configs.data.embodiment_configs import register_modality_config
from fred.data.embodiment_tags import EmbodimentTag
from fred.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

droid_dreamzero = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["joint_position", "gripper_position"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(24)),
        modality_keys=["joint_position", "gripper_position"],
        action_configs=[
            # joint_position (7): relative to the current state
            ActionConfig(
                representation=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper_position (1): absolute
            ActionConfig(
                representation=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.language.language_instruction",
            "annotation.language.language_instruction_2",
            "annotation.language.language_instruction_3",
        ],
    ),
}

register_modality_config(droid_dreamzero, EmbodimentTag.NEW_EMBODIMENT)
