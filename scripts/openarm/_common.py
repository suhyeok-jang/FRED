"""Constants and helpers shared by the OpenArm data-tooling scripts in this directory."""
from __future__ import annotations

import json
import os
import resource
from pathlib import Path

import numpy as np

DIM = 28
OPENARM28_GROUPS = {   # the 28-dim state/action layout (openarm_config.py)
    "neck_joints": (0, 2), "left_arm_joints": (2, 9), "right_arm_joints": (9, 16),
    "left_hand_joints": (16, 22), "right_hand_joints": (22, 28),
}
ANNOTATION_KEYS = ["human.action.task_description", "human.task_description"]


def joint_names(robot_subset: str | Path) -> list[str]:
    """The 28 joint names of a robot subset (its meta/info.json `action.names`)."""
    names = json.load(open(Path(robot_subset) / "meta" / "info.json"))["features"]["action"]["names"]
    assert len(names) == DIM, f"{robot_subset}: expected {DIM} action names, got {len(names)}"
    return names


def stats_1d(x: np.ndarray) -> dict[str, list[float]]:
    """stats.json entry {mean,std,min,max,q01,q99} of a 1-d float feature (timestamp)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1, 1)
    return dict(mean=np.mean(x, 0).tolist(), std=np.std(x, 0).tolist(), min=np.min(x, 0).tolist(),
                max=np.max(x, 0).tolist(), q01=np.quantile(x, 0.01, 0).tolist(), q99=np.quantile(x, 0.99, 0).tolist())


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def write_json(path: Path, payload, dry_run: bool = False) -> None:
    """Atomic json dump (tmp + os.replace)."""
    if dry_run:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=4)
    os.replace(tmp, path)
