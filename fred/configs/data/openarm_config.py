"""OpenArm 28-dim modality config (robot-only and robot+human runs).

neck 2 | left_arm 7 | right_arm 7 | left_hand 6 | right_hand 6 = 28. Arms are RELATIVE (action − current state of
the same key; needs data.use_relative_action, bounds from meta/relative_stats.json and q01/q99 when
data.use_percentiles), neck/hands ABSOLUTE. Action horizon 24 at frame stride 3. Video: camera_ego_left,
camera_ego_right (left first → top of the vconcat canvas). Human subsets are video-only (meta/wam_human.json +
data.human_datasets). meta/relative_stats.json must be the H=24 table ((24, 7) per arm key; (48, 7) raises).
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

_JOINT_ABS = ActionConfig(
    representation=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)
_JOINT_REL = ActionConfig(
    representation=ActionRepresentation.RELATIVE,   # action − state[key][-1] (current-frame anchor)
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

# modality.json order of the OpenArm subsets (start/end: 0-2 | 2-9 | 9-16 | 16-22 | 22-28)
_OPENARM28_KEYS = ["neck_joints", "left_arm_joints", "right_arm_joints", "left_hand_joints", "right_hand_joints"]

# which groups are RELATIVE: arms only (neck / hands stay ABSOLUTE)
_RELATIVE_KEYS = {"left_arm_joints", "right_arm_joints"}
_ACTION_CONFIGS = [_JOINT_REL if k in _RELATIVE_KEYS else _JOINT_ABS for k in _OPENARM28_KEYS]

openarm = {
    "video": ModalityConfig(
        # delta_indices overridden by launch_train.py (num_frames_in/out × frame_downsample_factor).
        delta_indices=[0],
        modality_keys=["camera_ego_left", "camera_ego_right"],   # LEFT first → top of the vconcat canvas
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=_OPENARM28_KEYS,   # == action keys (required: relative anchor = state of the same key)
    ),
    "action": ModalityConfig(
        delta_indices=list(range(24)),   # horizon 24 at stride 3
        modality_keys=_OPENARM28_KEYS,
        action_configs=_ACTION_CONFIGS,
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(openarm, EmbodimentTag.NEW_EMBODIMENT)
