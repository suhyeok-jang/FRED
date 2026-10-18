"""Observation-history stacking + action-chunk execution around a flat-key sim env."""

from collections import defaultdict, deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np


def stack_repeated(x, n, loc):
    return np.repeat(np.expand_dims(x, axis=loc), n, axis=loc)


def repeated_box(box_space, n, loc):
    return spaces.Box(
        low=stack_repeated(box_space.low, n, loc),
        high=stack_repeated(box_space.high, n, loc),
        shape=box_space.shape[:loc] + (n,) + box_space.shape[loc:],
        dtype=box_space.dtype,
    )


def repeated_space(space, n, loc=0):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n, loc)
    if isinstance(space, spaces.Dict):
        return spaces.Dict({key: repeated_space(value, n, loc) for key, value in space.items()})
    if isinstance(space, spaces.Discrete):  # robot0_base_mode of the PandaOmron kitchen envs
        return spaces.MultiDiscrete([[space.n] for _ in range(n)])
    if isinstance(space, spaces.Text):  # language is not stacked; only the latest is kept
        return space
    raise RuntimeError(f"Unsupported space type {type(space)}")


def assert_delta_indices(delta_indices: np.ndarray):
    # non-positive: future observations do not exist
    assert np.all(delta_indices <= 0), f"{delta_indices=}"
    # the latest observation must be included
    assert delta_indices[-1] == 0, f"{delta_indices=}"
    if len(delta_indices) > 1:
        # uniform positive stride (real-robot observation sampling relies on a fixed dt)
        stride = delta_indices[1] - delta_indices[0]
        assert stride > 0 and np.all(np.diff(delta_indices) == stride), f"{delta_indices=}"


class MultiStepWrapper(gym.Wrapper):
    """Observations: every ``video.*`` / ``state.*`` key stacked over its delta indices (T, ...), ``annotation.*``
    keys as the latest value. Actions: dict of (n_action_steps, ...) chunks executed step by step. The episode
    ends on the env's done, on ``max_episode_steps`` raw steps, or (``terminate_on_success``) on success."""

    def __init__(
        self,
        env,
        video_delta_indices: np.ndarray,
        state_delta_indices: np.ndarray,
        n_action_steps: int,
        max_episode_steps: int,
        terminate_on_success: bool = False,
    ):
        super().__init__(env)
        assert_delta_indices(video_delta_indices)
        assert_delta_indices(state_delta_indices)
        self.video_delta_indices = video_delta_indices
        self.state_delta_indices = state_delta_indices
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = self.convert_observation_space(self.observation_space)

        self.max_episode_steps = max_episode_steps
        self.n_action_steps = n_action_steps
        self.terminate_on_success = terminate_on_success
        # oldest observation any delta index reaches back to, plus the latest
        self.max_steps_needed = int(max(np.ptp(video_delta_indices), np.ptp(state_delta_indices)) + 1)

        self.obs = deque(maxlen=self.max_steps_needed + 1)
        self.reward = []
        self.done = []
        self.info = defaultdict(lambda: deque(maxlen=self.n_action_steps))

    def convert_observation_space(self, observation_space):
        new_observation_space = {}
        for k, space in observation_space.items():
            if k.startswith("video"):
                new_observation_space[k] = repeated_space(space, len(self.video_delta_indices))
            elif k.startswith("state"):
                new_observation_space[k] = repeated_space(space, len(self.state_delta_indices))
            else:
                assert k.startswith("annotation"), f"observation key without a video/state/annotation prefix: {k}"
                new_observation_space[k] = space
        return spaces.Dict(new_observation_space)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.obs = deque([obs] * (self.max_steps_needed + 1), maxlen=self.max_steps_needed + 1)
        self.reward = []
        self.done = []
        self.info = defaultdict(lambda: deque(maxlen=self.n_action_steps))
        return self._get_obs(), {k: [v] for k, v in info.items()}

    def step(self, action):
        """action: dict whose values are (n_action_steps,) + action_shape."""
        truncated = False
        for step in range(self.n_action_steps):
            if self.done and self.done[-1]:
                break
            observation, reward, done, truncated, info = super().step({k: v[step] for k, v in action.items()})
            self.obs.append(observation)
            self.reward.append(reward)
            if len(self.reward) >= self.max_episode_steps:
                done = True
            self.done.append(done)
            for key, value in info.items():
                self.info[key].append(value)

        reward = max(self.reward)
        done = any(self.done)
        # the last n_action_steps values of every info key
        info = {k: np.array(v) for k, v in self.info.items()}
        if self.terminate_on_success and any(info["success"]):
            done = True
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        """video/state keys stacked over their delta indices; annotation keys from the latest observation.
        Delta index 0 is the latest observation (deque position -1): [-4, ..., 0] -> [obs[-5], ..., obs[-1]]."""
        result = {}
        for key in self.observation_space.keys():
            if key.startswith("annotation"):
                result[key] = self.obs[-1][key]
                continue
            delta_indices = (self.video_delta_indices if key.startswith("video") else self.state_delta_indices) - 1
            result[key] = np.stack([self.obs[i][key] for i in delta_indices], axis=0)
        return result
