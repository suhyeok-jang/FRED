"""LIBERO panda modality config (libero_delta LeRobot dataset; mirrors the DiT4DiT Libero4in1 recipe).

  - state[8]  = eef_pos_absolute[3] + eef_rot_absolute[3] + gripper_close[2]
  - action[7] = eef_pos_delta[3]    + eef_rot_delta[3]    + gripper_close[1]
  - video     = front_view + left_wrist_view, 256x256 @ 10 fps
  - the four LIBERO suites (spatial / object / goal / 10) merged into one dataset

Key names follow libero_delta's meta/modality.json, not the DiT4DiT flat naming (state.x/y/z/...); the per-axis
values are the same. All state keys are sin/cos-encoded, which also needs DataConfig.apply_sincos_state_encoding
(CLI --apply-sincos-state-encoding).
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


_STATE_KEYS = ["eef_pos_absolute", "eef_rot_absolute", "gripper_close"]
_ACTION_KEYS = ["eef_pos_delta", "eef_rot_delta", "gripper_close"]

# Data-side action chunk length. Must equal WAMConfig.action_horizon: --action-horizon only sets the model
# config and does not touch these delta_indices, so a mismatch trains the extra positions unsupervised
# (action_mask zeroed past action_horizon). 16 = the GR-1 geometry; the DiT4DiT LIBERO recipe used 8.
_ACTION_HORIZON = 16


libero_panda = {
    "video": ModalityConfig(
        # placeholder; launch_train.py rewrites video delta_indices from the model's clip geometry
        delta_indices=[0],
        modality_keys=["front_view", "left_wrist_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=_STATE_KEYS,
        sincos_encoding_keys=_STATE_KEYS,
    ),
    "action": ModalityConfig(
        delta_indices=list(range(_ACTION_HORIZON)),
        modality_keys=_ACTION_KEYS,
        action_configs=[
            ActionConfig(
                representation=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                representation=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                representation=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}


register_modality_config(libero_panda, EmbodimentTag.NEW_EMBODIMENT)
