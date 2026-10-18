"""LIBERO tasks as gymnasium envs (``libero_sim/<task>``), in the libero_delta key schema of
fred/configs/data/libero_config.py."""

import math
import os

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from libero.libero import benchmark

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from libero.libero.utils import get_libero_path  # noqa: E402
import numpy as np  # noqa: E402


def quat2axisangle(quat):
    """robosuite's quaternion (x, y, z, w) -> axis-angle (unit direction scaled by the angle in radians)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # (close to) a zero-degree rotation
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class LiberoEnv(gym.Env):
    """One LIBERO task (bddl file) as a gymnasium env.

    Keys follow the libero_delta dataset: video.front_view / video.left_wrist_view, grouped
    state.eef_pos_absolute / eef_rot_absolute / gripper_close, action.eef_pos_delta / eef_rot_delta /
    gripper_close. The gripper_close action is robosuite-native (-1 open / +1 close), so actions are passed
    straight through to OSC_POSE.
    """

    def __init__(self, task_bddl_file: str, task_description: str, num_steps_wait: int = 10):
        self._env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=256, camera_widths=256)
        self._task_description = task_description
        # OpenVLA/DiT4DiT convention: `num_steps_wait` zero-action steps at episode start so loose
        # objects settle before the policy acts (default 10).
        self._num_steps_wait = int(num_steps_wait)
        img = lambda: spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8)  # noqa: E731
        self.observation_space = spaces.Dict(
            {
                "video.front_view": img(),
                "video.left_wrist_view": img(),
                # dims 3/3/2 as in the ckpt's dataset_statistics.json; state gripper_close is the raw 2-finger
                # robot0_gripper_qpos (~[-0.042, 0.042]), not a binarized flag
                "state.eef_pos_absolute": spaces.Box(low=-1, high=1, shape=(3,)),
                "state.eef_rot_absolute": spaces.Box(low=-1, high=1, shape=(3,)),
                "state.gripper_close": spaces.Box(low=-1, high=1, shape=(2,)),
                "annotation.human.action.task_description": spaces.Text(max_length=512),
            }
        )
        self.action_space = spaces.Dict(
            {
                "action.eef_pos_delta": spaces.Box(low=-1, high=1, shape=(3,)),
                "action.eef_rot_delta": spaces.Box(low=-1, high=1, shape=(3,)),
                "action.gripper_close": spaces.Box(low=-1, high=1, shape=(1,)),
            }
        )

    def close(self):
        self._env.close()

    def _process_observation(self, obs):
        # the [::-1, ::-1] flip undoes robosuite's upside-down offscreen render
        return {
            "video.front_view": obs["agentview_image"][::-1, ::-1],
            "video.left_wrist_view": obs["robot0_eye_in_hand_image"][::-1, ::-1],
            "state.eef_pos_absolute": list(obs["robot0_eef_pos"]),
            "state.eef_rot_absolute": list(quat2axisangle(obs["robot0_eef_quat"])),
            "state.gripper_close": obs["robot0_gripper_qpos"],
            "annotation.human.action.task_description": self._task_description,
        }

    def reset(self, seed=None, options=None):
        observation = self._env.reset()
        # Settling warmup: zero motion + gripper open for `_num_steps_wait` ticks on the underlying
        # robosuite env (bypassing our step(), so the wrappers' step counts start at the post-warmup
        # obs). robosuite Panda action = [dx, dy, dz, droll, dpitch, dyaw, gripper], gripper=-1 open.
        if self._num_steps_wait > 0:
            zero_action = np.zeros(7, dtype=np.float32)
            zero_action[-1] = -1.0
            for _ in range(self._num_steps_wait):
                observation, _, _, _ = self._env.step(zero_action)
        return self._process_observation(observation), {"success": self._env.check_success()}

    def step(self, action):
        action_vector = np.concatenate(
            [
                np.asarray(action["action.eef_pos_delta"], dtype=np.float32).reshape(-1),
                np.asarray(action["action.eef_rot_delta"], dtype=np.float32).reshape(-1),
                np.asarray(action["action.gripper_close"], dtype=np.float32).reshape(-1),
            ],
            axis=0,
        )
        observation, reward, done, info = self._env.step(action_vector)
        info["success"] = self._env.check_success()
        return self._process_observation(observation), reward, done, False, info


def register_libero_envs():
    benchmark_dict = benchmark.get_benchmark_dict()
    for task_suite_name in ["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90"]:
        task_suite = benchmark_dict[task_suite_name]()
        for task_id in range(task_suite.get_num_tasks()):
            task = task_suite.get_task(task_id)
            register(
                id=f"libero_sim/{task.name}",
                entry_point="fred.eval.sim.libero.libero_env:LiberoEnv",
                kwargs={
                    "task_bddl_file": os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                    "task_description": task.language,
                },
            )
