#!/usr/bin/env python
"""Make OpenArm robot subsets loadable by ONE modality config (openarm_config.py).

Idempotent, CPU-only, edits ONLY meta/ of the given subsets:

  1. video keys: every subset must expose `camera_ego_left/right` in meta/modality.json (the loader
     resolves `modality.json[video][key][original_key]`, lerobot_episode_loader.py::_load_video_data, and
     asserts that key is in info.json features). A subset naming them ego_left/ego_right gets aliases.
  2. annotation aliases — the config uses `annotation.human.action.task_description`, some subsets natively
     have `human.task_description`; both are added everywhere (both → original_key task_index).
  3. every configured view of EVERY episode must resolve to an existing video file (asserted, using the
     loader's own path formula: info["video_path"].format(episode_chunk=ep//chunks_size, video_key=<original
     key>, episode_index=ep)); no substitutes.
  4. per-dataset action groups (--action-groups): a subset that supplies only SOME of the 28-dim layout's
     action groups (the other dims are zero-filled) gets `meta/wam_action_groups.json` {"valid_groups": [...]}.
     The loader refuses such a dataset unless ``--dataset-action-groups`` (launcher DATASET_ACTION_GROUPS=1) is
     set; with it the missing groups' columns are zeroed in action_mask + action per sample, so they are
     EXCLUDED from the action FM loss instead of being taught as zeros.

Usage: python scripts/openarm/prep_openarm_meta.py <robot subset dir> [...]
           [--action-groups <subset dir>=right_arm_joints,right_hand_joints ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ANNOTATION_KEYS, OPENARM28_GROUPS, write_json

VIDEO_ALIASES = {"camera_ego_left": "ego_left", "camera_ego_right": "ego_right"}   # unified → short name
ACTION_GROUPS_FILENAME = "wam_action_groups.json"


def parse_action_groups(specs: list[str]) -> dict[Path, list[str]]:
    """['<subset dir>=<group>,<group>', ...] -> {resolved subset dir: [group, ...]}."""
    out = {}
    for spec in specs:
        d, sep, groups = spec.rpartition("=")
        names = [g for g in groups.split(",") if g]
        assert sep and d and names, f"--action-groups expects <subset dir>=<group>[,<group>...], got {spec!r}"
        unknown = [g for g in names if g not in OPENARM28_GROUPS]
        assert not unknown, f"--action-groups {spec!r}: unknown groups {unknown} (have {list(OPENARM28_GROUPS)})"
        out[Path(d).resolve()] = names
    return out


def patch_modality(path: Path, info_features: dict, dry_run: bool) -> list[str]:
    m = json.load(open(path))
    changes: list[str] = []
    video = m.setdefault("video", {})
    for uni, old in VIDEO_ALIASES.items():
        if uni not in video and old in video:
            video[uni] = dict(video[old])
            changes.append(f"video[{uni}] = alias of {old} ({video[old]['original_key']})")
    for uni in VIDEO_ALIASES:
        ok = video.get(uni, {}).get("original_key")
        assert ok in info_features, f"{path}: {uni} → {ok} not in info.json features"
    ann = m.setdefault("annotation", {})
    src = next((ann[k] for k in ANNOTATION_KEYS if k in ann), None)
    assert src is not None, f"{path}: no known annotation key in {list(ann)}"
    for k in ANNOTATION_KEYS:
        if k not in ann:
            ann[k] = dict(src)
            changes.append(f"annotation[{k}] = alias ({src.get('original_key')})")
    if changes:
        write_json(path, m, dry_run)
    return changes


def write_action_groups(d: Path, groups: list[str] | None, dry_run: bool) -> list[str]:
    """Write meta/wam_action_groups.json when the subset supplies only ``groups`` (idempotent)."""
    if groups is None:
        return []
    m = json.load(open(d / "meta" / "modality.json"))
    missing = [g for g in groups if g not in m.get("action", {})]
    assert not missing, f"{d}: action groups {missing} are not in meta/modality.json action layout"
    path = d / "meta" / ACTION_GROUPS_FILENAME
    payload = {
        "valid_groups": list(groups),
        "note": ("only these action groups are real in this dataset; the other groups are zero-filled and "
                 "are masked out of the action loss (--dataset-action-groups)"),
    }
    if path.exists() and json.load(open(path)).get("valid_groups") == list(groups):
        return []
    write_json(path, payload, dry_run)
    return [f"meta/{ACTION_GROUPS_FILENAME} = {groups}"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare the meta/ of OpenArm robot subsets for openarm_config.py.")
    ap.add_argument("subsets", nargs="+", help="robot subset directories (LeRobot v2)")
    ap.add_argument("--action-groups", action="append", default=[], metavar="DIR=GROUP[,GROUP...]",
                    help="the subset at DIR supplies only these action groups (repeatable)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    subsets = [Path(s) for s in args.subsets]
    action_groups = parse_action_groups(args.action_groups)
    stray = sorted(str(p) for p in action_groups if p not in {d.resolve() for d in subsets})
    assert not stray, f"--action-groups names directories that are not among the subsets: {stray}"

    for d in subsets:
        info = json.load(open(d / "meta" / "info.json"))
        changes = patch_modality(d / "meta" / "modality.json", info["features"], args.dry_run)
        changes += write_action_groups(d, action_groups.get(d.resolve()), args.dry_run)
        assert (d / "videos").is_dir(), f"{d}: videos/ missing — a subset without its own videos is not usable"
        # every configured view of every episode must resolve to an existing video file
        # (loader formula: lerobot_episode_loader.py::_load_video_data)
        m = json.load(open(d / "meta" / "modality.json"))
        chunks = info["chunks_size"]
        n_ep = info["total_episodes"]
        n_checked = 0
        for uni in VIDEO_ALIASES:
            ok = m["video"][uni]["original_key"]
            for ep in range(n_ep):
                vp = d / info["video_path"].format(episode_chunk=ep // chunks, video_key=ok, episode_index=ep)
                assert vp.exists() or args.dry_run, f"{d}: {vp} missing"
                n_checked += 1
        print(f"[prep] {d}: " + ("; ".join(changes) if changes else "no change (already prepared)")
              + f"  [{n_checked} video files resolved over {n_ep} episodes x {len(VIDEO_ALIASES)} views]")


if __name__ == "__main__":
    main()
