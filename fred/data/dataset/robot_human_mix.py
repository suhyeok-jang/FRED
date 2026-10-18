"""Torch-free helpers for the OpenArm robot+human mixture (pure logic; importable without torch).

data.human_datasets
    Datasets carrying meta/wam_human.json {"human": true} are video-only members of the mixture: their rows get
    is_human=1, state := 0, action_mask := 0 (excluded from the action FM loss, whose reduction divides by
    action_mask.sum()), and a configured view the dataset lacks is rendered all-black. Consumers:
    lerobot_episode_loader.py (marker + missing-view skip), sharded_single_step_dataset.py (tagging / zeroing),
    fred/model/wam/processor.py (black missing view).

data.batch_group_fractions = "robot:0.5,human:0.5"
    Every per-GPU batch of PER_DEV clips holds exactly round(PER_DEV * f_human) human clips at fixed positions
    (`interleave_pattern`). Consumers: factory.py -> ExactRobotHumanMixDataset; the collator asserts the split.

data.dataset_action_groups
    A dataset carrying meta/wam_action_groups.json {"valid_groups": [...]} supplies only some action groups; the
    others are zero-filled in its parquet, so per sample their columns are zeroed in both action_mask (dropped
    from the loss) and action (the noisy trajectory is exactly 0 there, like the
    padding columns). Without the flag such a dataset is refused rather than trained on zeros. Consumers:
    lerobot_episode_loader.py (marker + refusal), sharded_single_step_dataset.py (per-sample masking),
    action_head.py (_record_valid_dim tolerates a non-leading-block mask).

data.view_mask_probs = "0.5,0.25,0.25"
    Per-sample draw {both views, mask view[0], mask view[1]}; the masked view is all-black (uint8 0 -> -1 after
    [-1, 1] scaling) for cond and future frames, with the same draw for both SelfDistill views. Missing views are
    always black and a sample is never fully black (single-view samples keep their only view). Train only and
    not persisted: an eval harness that has one view must black the other itself (processor.py
    _process_video_frames missing-view path).
"""
from __future__ import annotations

import json
from pathlib import Path

HUMAN_MARKER_FILENAME = "wam_human.json"
ACTION_GROUPS_MARKER_FILENAME = "wam_action_groups.json"


def read_human_marker(dataset_path: str | Path) -> bool:
    """True iff <dataset>/meta/wam_human.json exists and says {"human": true}."""
    p = Path(dataset_path) / "meta" / HUMAN_MARKER_FILENAME
    if not p.exists():
        return False
    with open(p) as f:
        d = json.load(f)
    return bool(d.get("human", False))


# ──────────────────────────── action groups (data.dataset_action_groups) ────────────────────────────
def read_action_groups_marker(dataset_path: str | Path) -> list[str] | None:
    """<dataset>/meta/wam_action_groups.json -> ["right_arm_joints", ...], or None when absent.

    Shape: {"valid_groups": [<action modality key>, ...]} (a non-empty list of strings). Anything else
    raises — a malformed marker must never degrade silently into "all groups valid".
    """
    p = Path(dataset_path) / "meta" / ACTION_GROUPS_MARKER_FILENAME
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    g = d.get("valid_groups")
    assert not (not isinstance(g, list) or not g or not all(isinstance(x, str) and x for x in g)), f"{p}: 'valid_groups' must be a non-empty list of action modality keys, got {g!r}"
    assert len(set(g)) == len(g), f"{p}: duplicate entries in 'valid_groups': {g!r}"
    return list(g)


def action_group_column_mask(
    action_keys: list[str],
    group_spans: dict[str, tuple[int, int]],
    valid_groups: list[str],
    total_dim: int,
) -> list[int]:
    """0/1 per column of the PADDED action vector (length `total_dim` = max_action_dim).

    The processed action vector is `cat([action[k] for k in action_keys])` (processing.py __call__),
    each key contributing `group_spans[k][1] - group_spans[k][0]` columns, then zero-padded to
    `total_dim`. Columns of a key not in `valid_groups` — and every padding column — are 0.

    >>> action_group_column_mask(["a", "b"], {"a": (0, 2), "b": (2, 5)}, ["b"], 6)
    [0, 0, 1, 1, 1, 0]
    """
    unknown = [g for g in valid_groups if g not in action_keys]
    assert not (unknown), f"valid_groups {unknown} are not action modality keys {action_keys}"
    mask = [0] * int(total_dim)
    off = 0
    for k in action_keys:
        assert k in group_spans, f"action key {k!r} has no span in the dataset's modality.json ({sorted(group_spans)})"
        s, e = int(group_spans[k][0]), int(group_spans[k][1])
        d = e - s
        assert d > 0, f"action key {k!r} has a non-positive span {(s, e)}"
        assert off + d <= total_dim, f"action keys {action_keys} need {off + d} columns > max_action_dim {total_dim}"
        if k in valid_groups:
            for i in range(off, off + d):
                mask[i] = 1
        off += d
    assert any(mask), f"valid_groups {valid_groups} select no column of {action_keys}"
    return mask


# ──────────────────────────────── exact mix (data.batch_group_fractions) ────────────────────────────────
def parse_batch_group_fractions(s: str | None) -> dict[str, float] | None:
    """'robot:0.5,human:0.5' → {'robot':0.5,'human':0.5}; None when unset (exact mix off).

    Only the two groups robot/human are supported; fractions must sum to 1.
    """
    if s is None or not s.strip():
        return None
    out: dict[str, float] = {}
    for tok in s.strip().split(","):
        k, v = tok.split(":")
        out[k.strip()] = float(v)
    assert set(out) == {"robot", "human"}, f"data.batch_group_fractions groups must be exactly robot,human — got {sorted(out)} from {s!r}"
    assert not (any(v <= 0 for v in out.values()) or abs(sum(out.values()) - 1.0) > 1e-6), f"data.batch_group_fractions fractions must be >0 and sum to 1, got {out}"
    return out


def robot_human_split_counts(per_dev: int, human_frac: float) -> tuple[int, int]:
    """(n_robot, n_human) clips per batch of `per_dev`. Raises if the split is not exact."""
    n_h = per_dev * human_frac
    assert abs(n_h - round(n_h)) <= 1e-9, (
        f"data.batch_group_fractions: per-device batch {per_dev} × human_frac={human_frac} = {n_h} is not an integer — "
        f"choose PER_DEV so the split is exact (e.g. even PER_DEV for 0.5)."
    )
    n_h = int(round(n_h))
    assert not (n_h <= 0 or n_h >= per_dev), f"data.batch_group_fractions: degenerate split n_human={n_h} of per-device batch {per_dev}"
    return per_dev - n_h, n_h


def interleave_pattern(per_dev: int, n_human: int) -> list[str]:
    """Fixed per-batch group order of length per_dev with exactly n_human 'human' entries,
    spread as evenly as possible (Bresenham), robot first. 8/4 → r,h,r,h,r,h,r,h."""
    assert 0 < n_human < per_dev
    pat: list[str] = []
    acc = 0
    for _ in range(per_dev):
        acc += n_human
        if acc >= per_dev:
            acc -= per_dev
            pat.append("human")
        else:
            pat.append("robot")
    assert pat.count("human") == n_human, (pat, n_human)
    return pat
