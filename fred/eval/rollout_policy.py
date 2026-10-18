"""Sim rollout client: one process evaluates one task against a running policy server (serve_policy.py).

Episodes form an (n_envs x n_episodes // n_envs) grid; env column ``i`` is seeded ``seed + i`` and episode
``k`` of it is reset with a seed derived from both (VideoRecordingWrapper). The run resumes from the
``<prefix>-episode_<k>-{success,failure}.mp4`` files already in ``--output-dir``: every (env, episode) id that
has no such file is owed, so a run killed mid-way (including one that left an interior hole) is re-launched
as is. Results go to ``simulation_results.csv`` and ``summary.txt`` in ``--output-dir``.
"""

import argparse
import csv
from dataclasses import dataclass
from functools import partial
import os
from pathlib import Path
import random
import re
import time

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from fred.eval.sim.env_wrappers.multistep_wrapper import MultiStepWrapper
from fred.eval.sim.env_wrappers.video_recording_wrapper import VideoRecordingWrapper
from fred.policy.policy import BasePolicy
from fred.policy.server_client import PolicyClient

_CSV_FIELDS = ["env_idx", "episode_idx", "success", "reward", "steps", "video_path"]
# as written by VideoRecordingWrapper: <env_name>_env00-episode_3-failure.mp4
_RECORDED_MP4_RE = re.compile(r".*_env(?P<env>\d+)-episode_(?P<episode>\d+)-(?P<status>success|failure)\.mp4$")


@dataclass
class VideoConfig:
    """VideoRecordingWrapper settings. save_frames: also dump each recorded frame as PNG plus a per-episode
    instruction.txt next to the mp4; video_modality_keys: with save_frames, only the ``video.<name>`` keys whose
    suffix is listed (None = every ``video.*`` key, the set that goes into the mp4)."""

    output_dir: Path
    max_episode_steps: int
    save_frames: bool = False
    video_modality_keys: list[str] | None = None


def video_name_prefix(env_name: str, env_idx: int) -> str:
    return f"{env_name.replace('/', '_')}_env{env_idx:02d}"


def get_gym_env(env_name: str, seed: int) -> gym.Env:
    """The bare (unwrapped) environment."""
    if env_name.startswith(("robocasa", "gr1_unified")):
        os.environ.setdefault("MUJOCO_GL", "egl")
        import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401  registers robocasa_*/gr1_unified envs

        return gym.make(env_name, enable_render=True, seed=seed)
    if env_name.startswith("libero_sim/"):
        from fred.eval.sim.libero.libero_env import register_libero_envs

        register_libero_envs()
        return gym.make(env_name)
    raise ValueError(f"Invalid environment name: {env_name}")


def create_eval_env(
    env_name: str,
    env_idx: int,
    seed: int,
    pending_episode_ids: list[int],
    video: VideoConfig,
    video_delta_indices: np.ndarray,
    state_delta_indices: np.ndarray,
    n_action_steps: int,
) -> gym.Env:
    """Grid column ``env_idx``: the bare env (seed + env_idx) in the VideoRecordingWrapper in the MultiStepWrapper."""
    env_seed = seed + int(env_idx)
    # Under AsyncVectorEnv(spawn) each env is its own process, so also pin the process-global RNGs.
    random.seed(env_seed)
    np.random.seed(env_seed)
    print(f"[i] Creating environment {env_name} (env index: {env_idx}) with seed={env_seed}, "
          f"episode ids {pending_episode_ids}")

    env = VideoRecordingWrapper(
        get_gym_env(env_name, seed=env_seed),
        output_dir=video.output_dir,
        name_prefix=video_name_prefix(env_name, env_idx),
        pending_episode_ids=pending_episode_ids,
        base_seed=env_seed,
        max_episode_steps=video.max_episode_steps,
        save_frames=video.save_frames,
        video_modality_keys=video.video_modality_keys,
    )
    return MultiStepWrapper(
        env,
        video_delta_indices=video_delta_indices,
        state_delta_indices=state_delta_indices,
        n_action_steps=n_action_steps,
        max_episode_steps=video.max_episode_steps,
        terminate_on_success=True,
    )


def _any_success(success) -> bool:
    """An env's success flag: a bool, or the per-step list/array the MultiStepWrapper reports."""
    return bool(np.any(success))


def load_recorded_episodes(n_envs: int, output_dir: Path) -> dict[int, dict[int, bool]]:
    """{env_idx: {episode_idx: success}} from the finished mp4s in ``output_dir``. A mid-episode file without a
    status suffix (run interrupted before the end-of-episode rename) is not counted; its id is re-run and the
    file overwritten."""
    recorded: dict[int, dict[int, bool]] = {i: {} for i in range(n_envs)}
    for file in output_dir.glob("*.mp4"):
        match = _RECORDED_MP4_RE.match(file.name)
        if not match:
            continue
        env_idx = int(match.group("env"))
        if env_idx >= n_envs:
            print(f"[warn] Found video for out-of-range env {env_idx}: {file.name}")
            continue
        recorded[env_idx][int(match.group("episode"))] = match.group("status") == "success"
    return recorded


def compute_pending_episode_ids(
    recorded: dict[int, dict[int, bool]], n_envs: int, target_per_env: int
) -> dict[int, list[int]]:
    """The exact episode ids each env still owes: range(target_per_env) minus the recorded ids. An env holding
    {0..9} minus {3} (a run killed mid-episode) owes [3]; one with no record owes the whole range."""
    return {i: [e for e in range(target_per_env) if e not in recorded[i]] for i in range(n_envs)}


def _update_prediction_csv(csv_path: Path, row: dict) -> None:
    """Upsert ``row`` into simulation_results.csv, keyed by (env_idx, episode_idx)."""
    rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = [
                r for r in csv.DictReader(f)
                if (int(r["env_idx"]), int(r["episode_idx"])) != (row["env_idx"], row["episode_idx"])
            ]
    rows.append(row)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_rollout(
    env_name: str,
    policy: BasePolicy,
    video: VideoConfig,
    video_delta_indices: np.ndarray,
    state_delta_indices: np.ndarray,
    n_action_steps: int,
    n_episodes: int,
    n_envs: int,
    seed: int,
    env_indices: list[int] | None = None,
) -> tuple[str, list[bool]]:
    """Roll the policy out over the episode grid, resuming from the mp4s already in video.output_dir.

    n_envs is the grid width (it fixes env_seed = seed + env_idx, the per-env quota n_episodes // n_envs and
    the mp4 names); env_indices selects which columns this process builds (None = all). Running one column
    per process keeps a single MuJoCo/EGL renderer resident yet produces the identical grid, whereas lowering
    n_envs would be a different experiment. Returns (env_name, episode successes incl. the resumed ones).
    """
    start_time = time.time()
    # Clamp before deriving the per-env quota: a 1-2 episode smoke test on an 8-env grid runs one per env.
    n_episodes = max(n_episodes, n_envs)
    assert n_episodes % n_envs == 0, f"n_episodes={n_episodes} must be a multiple of n_envs={n_envs}"
    target_per_env = n_episodes // n_envs

    all_indices = list(range(n_envs))
    if env_indices is None:
        env_indices = all_indices
    else:
        env_indices = [int(i) for i in env_indices]
        assert env_indices, "env_indices must not be empty"
        assert len(set(env_indices)) == len(env_indices), f"env_indices has duplicates: {env_indices}"
        assert all(0 <= i < n_envs for i in env_indices), (
            f"env_indices={env_indices} is out of range for the n_envs={n_envs} grid (n_envs is the grid "
            "width and keeps its full value in sequential mode)"
        )
    env_subset = env_indices != all_indices
    n_slots = len(env_indices)
    if env_subset:
        print(f"[i] SEQUENTIAL-ENV: this process runs grid column(s) {env_indices} of {n_envs} "
              f"(env_seed = {[seed + i for i in env_indices]}); quota {target_per_env} episode(s) per env.")

    # resume: owed work = the set of missing ids per env, so an interior hole is refilled too
    video.output_dir.mkdir(parents=True, exist_ok=True)
    recorded = load_recorded_episodes(n_envs, video.output_dir)
    pending = compute_pending_episode_ids(recorded, n_envs, target_per_env)
    existing_successes = [
        recorded[env_idx][ep]
        for ep, env_idx in sorted((ep, i) for i in range(n_envs) for ep in recorded[i] if ep < target_per_env)
    ]
    for i in range(n_envs):
        if pending[i]:
            print(f"[i] Env {i}: owes {len(pending[i])} episode id(s) {pending[i]}")
    print(f"[i] Resume: {len(existing_successes)}/{n_episodes} recorded, "
          f"{sum(len(v) for v in pending.values())} owed")
    owed = sum(len(pending[i]) for i in env_indices)
    if owed == 0:
        print("[i] Every selected column already holds its episodes. Nothing to run.")
        return env_name, existing_successes

    env_fns = [
        partial(
            create_eval_env,
            env_name=env_name,
            env_idx=idx,  # the absolute grid column (env_seed, mp4 prefix, quota)
            seed=seed,
            pending_episode_ids=pending[idx],
            video=video,
            video_delta_indices=video_delta_indices,
            state_delta_indices=state_delta_indices,
            n_action_steps=n_action_steps,
        )
        for idx in env_indices
    ]
    # a one-column process should not spawn an AsyncVectorEnv worker for one renderer
    if n_slots == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = gym.vector.AsyncVectorEnv(env_fns, shared_memory=False, context="spawn")

    csv_path = video.output_dir / "simulation_results.csv"
    # per slot (the vector env's axis); env_indices[slot] is the grid column it plays
    run_counts = [0] * n_slots
    current_rewards = [0.0] * n_slots
    current_lengths = [0] * n_slots
    current_successes = [False] * n_slots
    episode_successes = list(existing_successes)

    observations, _ = env.reset()
    policy.reset()  # the policy is stateless, so it is not reset per episode
    pbar = tqdm(total=owed, desc="Episodes")
    # lockstep: an env that has played every id it owed keeps stepping (recording nothing) until the others finish
    while any(run_counts[slot] < len(pending[env_indices[slot]]) for slot in range(n_slots)):
        actions, _ = policy.get_action(observations)
        observations, rewards, terminations, truncations, infos = env.step(actions)
        for slot, env_idx in enumerate(env_indices):
            current_successes[slot] |= _any_success(infos["success"][slot])
            # gymnasium autoreset: the terminal step's info moves to final_info, infos holds the reset's
            if "final_info" in infos and infos["final_info"][slot] is not None:
                current_successes[slot] |= _any_success(infos["final_info"][slot]["success"])
            current_rewards[slot] += float(rewards[slot])
            current_lengths[slot] += 1
            if not (terminations[slot] or truncations[slot]):
                continue
            if run_counts[slot] < len(pending[env_idx]):
                episode_idx = pending[env_idx][run_counts[slot]]  # the id the wrapper named the mp4 with
                run_counts[slot] += 1
                episode_successes.append(current_successes[slot])
                status = "success" if current_successes[slot] else "failure"
                _update_prediction_csv(csv_path, {
                    "env_idx": env_idx,
                    "episode_idx": episode_idx,
                    "success": int(current_successes[slot]),
                    "reward": current_rewards[slot],
                    "steps": current_lengths[slot],
                    "video_path": f"{video_name_prefix(env_name, env_idx)}-episode_{episode_idx}-{status}.mp4",
                })
                pbar.update(1)
            current_successes[slot] = False
            current_rewards[slot] = 0.0
            current_lengths[slot] = 0
    pbar.close()

    env.reset()  # finalizes (renames) the last mp4 of every env
    env.close()
    elapsed = time.time() - start_time
    print(f"Collecting {owed} episodes took {elapsed:.2f} seconds")
    assert len(episode_successes) == len(existing_successes) + owed, (
        f"Expected {len(existing_successes) + owed} episodes, got {len(episode_successes)}"
    )
    # a one-column process must not overwrite the whole task's summary.txt with its slice
    summary_name = "summary.txt" if not env_subset else "summary_env" + "-".join(f"{i:02d}" for i in env_indices) + ".txt"
    with open(video.output_dir / summary_name, "w") as f:
        f.write(f"Collecting {owed} episodes took {elapsed:.2f} seconds\n")
        f.write(f"Success rate: {np.mean(episode_successes):.2f}\n")
    return env_name, episode_successes


def run_sim_policy(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    policy_client_host: str,
    policy_client_port: int,
    n_envs: int,
    n_action_steps: int,
    output_dir: str,
    seed: int,
    save_frames: bool = False,
    save_frames_modality_keys: str = "",
    env_indices: list[int] | None = None,
) -> tuple[str, list[bool]]:
    policy = PolicyClient(host=policy_client_host, port=policy_client_port)
    modality_configs = policy.get_modality_config()
    video_delta_indices = np.array(modality_configs["video"].delta_indices)
    state_delta_indices = np.array(modality_configs["state"].delta_indices)
    print(f"Using video_delta_indices: {video_delta_indices}, state_delta_indices: {state_delta_indices}")

    # The VideoRecordingWrapper sees the raw env, whose keys need not be the policy's modality keys, so
    # --save-frames-modality-keys gives the PNG filter explicitly; --save-frames alone = every video.* key.
    video_modality_keys = None
    if save_frames_modality_keys:
        video_modality_keys = [k.strip() for k in save_frames_modality_keys.split(",")]
    video = VideoConfig(
        output_dir=Path(output_dir),
        max_episode_steps=max_episode_steps,
        save_frames=save_frames,
        video_modality_keys=video_modality_keys,
    )
    results = run_rollout(
        env_name=env_name,
        policy=policy,
        video=video,
        video_delta_indices=video_delta_indices,
        state_delta_indices=state_delta_indices,
        n_action_steps=n_action_steps,
        n_episodes=n_episodes,
        n_envs=n_envs,
        seed=seed,
        env_indices=env_indices,
    )
    print("Results in:", output_dir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-name", type=str, required=True,
                        help="robocasa_panda_omron/<Task>_PandaOmron_Env, gr1_unified/<Task>_GR1ArmsAndWaistFourierHands_Env "
                             "or libero_sim/<task>")
    parser.add_argument("--policy-client-host", type=str, default="localhost")
    parser.add_argument("--policy-client-port", type=int, required=True)
    parser.add_argument("--output-dir", dest="output_dir", type=str, required=True,
                        help="Output dir: mp4s, simulation_results.csv, summary.txt; the run resumes from the mp4s in it.")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=504)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-frames", action="store_true", default=False,
                        help="Save per-step PNG frames plus one instruction.txt per episode, alongside the mp4.")
    parser.add_argument("--save-frames-modality-keys", type=str, default="",
                        help="Comma-separated raw video obs keys to dump with --save-frames (default: every video.* key).")
    parser.add_argument(
        "--env-indices", type=str, default="",
        help="SEQUENTIAL-ENV mode: comma-separated grid columns this process runs, e.g. '3'. --n-envs keeps its "
             "full value (the grid width: env_seed = seed + column, quota n_episodes // n_envs per column), so "
             "running the columns one process at a time keeps one renderer resident and produces the identical "
             "grid. Empty (default) = every column in one AsyncVectorEnv.",
    )
    args = parser.parse_args()
    env_indices_arg = [int(t) for t in args.env_indices.replace(",", " ").split()] if args.env_indices.strip() else None

    results = run_sim_policy(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        save_frames=args.save_frames,
        save_frames_modality_keys=args.save_frames_modality_keys,
        env_indices=env_indices_arg,
    )
    print("results: ", results)
    print("success rate: ", np.mean(results[1]))
