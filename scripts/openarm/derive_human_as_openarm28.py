#!/usr/bin/env python
"""Derive "video-only" OpenArm-28-shaped copies of human-video subsets. CPU-only, torch-free.

Input  : human subsets in LeRobot v2.1 (any `observation.state`, NO action, a single camera view `--view`,
         default camera_ego_right, in meta/modality.json)
Output : <out-root>/<subset directory name>/
  videos/            symlink → the source videos (nothing copied)
  data/…parquet      source rows with `observation.state` := zeros(28, float32) and a new
                     `action` := zeros(28, float32) (robot layout); all other columns kept
  meta/info.json     features observation.state/action → 28-dim with the robot joint names (--names-from),
                     robot_type suffixed
  meta/modality.json robot state/action groups (neck|left_arm|right_arm|left_hand|right_hand), video
                     ONLY the --view (→ source original_key), annotation both language keys
  meta/tasks.jsonl, meta/episodes.jsonl  copied (prompts used as-is)
  meta/stats.json    POOLED robot stats for observation.state/action (+ timestamp computed here) so
                     normalisation is identical to the robot subsets (state is zeroed AFTER normalisation
                     by the loader anyway; fred/data/stats.py::generate_stats needs every float feature)
  meta/relative_stats.json  POOLED robot relative stats (processor gate for RELATIVE keys)
  meta/wam_human.json {"human": true, …}  → ``--human-datasets`` (launcher HUMAN_DATASETS=1) tags the rows
                     (state 0, action_mask 0, left view black); without the flag the loader REFUSES the dataset.

Run AFTER pool_openarm_stats.py (--pooled-stats / --pooled-rel-stats are its pooled payloads).
Idempotent: re-running rewrites everything. RSS: one parquet at a time.

Usage: python scripts/openarm/derive_human_as_openarm28.py <human subset dir> [...] --out-root D
           --names-from <robot subset dir> --pooled-stats F --pooled-rel-stats F [--view camera_ego_right] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ANNOTATION_KEYS, DIM, OPENARM28_GROUPS, joint_names, rss_mb, stats_1d

VIEW_KEY = "camera_ego_right"


def build_modality(src_modality: dict, view: str) -> dict:
    groups = {k: {"start": s, "end": e} for k, (s, e) in OPENARM28_GROUPS.items()}
    src_video = src_modality["video"]
    assert view in src_video and len(src_video) == 1, f"expected only {view}, got {list(src_video)}"
    ann_src = next((src_modality["annotation"][k] for k in ANNOTATION_KEYS if k in src_modality.get("annotation", {})), None)
    assert ann_src is not None, f"no known annotation key in {src_modality.get('annotation')}"
    return {
        "state": json.loads(json.dumps(groups)),
        "action": json.loads(json.dumps(groups)),
        "video": {view: dict(src_video[view])},
        "annotation": {k: dict(ann_src) for k in ANNOTATION_KEYS},
    }


def build_info(src_info: dict, names: list[str], note: str) -> dict:
    info = json.loads(json.dumps(src_info))
    feats = {}
    for k, v in src_info["features"].items():
        if k == "observation.state":
            feats["observation.state"] = {"dtype": "float32", "shape": [DIM], "names": list(names),
                                          "description": "zero-filled placeholder (human data has no robot state)"}
            feats["action"] = {"dtype": "float32", "shape": [DIM], "names": list(names),
                               "description": "zero-filled placeholder (human data has no action; masked out)"}
        elif k == "action":
            continue
        else:
            feats[k] = v
    assert "observation.state" in feats and "action" in feats
    info["features"] = feats
    info["robot_type"] = f"{src_info.get('robot_type', 'human')}_as_openarm28_human"
    info["_note"] = note
    return info


def rewrite_parquet(src: Path, dst: Path, dry_run: bool) -> tuple[int, np.ndarray]:
    df = pd.read_parquet(src)
    n = len(df)
    zeros = [np.zeros(DIM, dtype=np.float32) for _ in range(n)]
    cols = {"observation.state": zeros, "action": [z.copy() for z in zeros]}
    for c in df.columns:
        if c in ("observation.state", "action"):
            continue
        cols[c] = df[c]
    out = pd.DataFrame(cols)
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(dst, index=False)
    ts = np.asarray(df["timestamp"], dtype=np.float32) if "timestamp" in df.columns else np.zeros(0, np.float32)
    return n, ts


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive video-only OpenArm-28 copies of human-video subsets.")
    ap.add_argument("subsets", nargs="+", help="human subset directories (LeRobot v2.1)")
    ap.add_argument("--out-root", required=True, help="each copy is written to <out-root>/<subset directory name>")
    ap.add_argument("--names-from", required=True,
                    help="robot subset whose meta/info.json action names label the 28 dims")
    ap.add_argument("--pooled-stats", required=True, help="pooled_openarm28_stats.json of pool_openarm_stats.py")
    ap.add_argument("--pooled-rel-stats", required=True,
                    help="pooled_openarm28_relative_stats.json of pool_openarm_stats.py")
    ap.add_argument("--view", default=VIEW_KEY, help="the single camera view of the human subsets")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    sources = [Path(s) for s in args.subsets]
    out_names = [s.resolve().name for s in sources]
    assert len(set(out_names)) == len(out_names), f"subset directory names must be unique: {out_names}"
    pooled = json.load(open(args.pooled_stats))
    pooled_rel = json.load(open(args.pooled_rel_stats))
    assert all(len(pooled[c]["q01"]) == DIM for c in ("observation.state", "action"))
    names = joint_names(args.names_from)
    note = ("human video-only copy: observation.state/action zero-filled 28-dim (OpenArm layout), videos "
            "symlinked, stats = pooled robot stats; loader marks rows is_human (--human-datasets)")

    for src, name in zip(sources, out_names):
        dst = out_root / name
        assert (src / "meta" / "info.json").exists(), f"missing source {src}"
        src_info = json.load(open(src / "meta" / "info.json"))
        src_mod = json.load(open(src / "meta" / "modality.json"))
        if not args.dry_run:
            (dst / "meta").mkdir(parents=True, exist_ok=True)
            # videos: relative symlink (replace whatever is there)
            vlink = dst / "videos"
            if vlink.is_symlink() or vlink.exists():
                if vlink.is_dir() and not vlink.is_symlink():
                    shutil.rmtree(vlink)
                else:
                    vlink.unlink()
            os.symlink(os.path.relpath(src.resolve() / "videos", dst.resolve()), vlink)
        # corrupt/truncated source parquets are DROPPED: the episode is removed from episodes.jsonl (the loader
        # enumerates episodes from it and builds paths from episode_index, so no renumbering) and listed in
        # meta/wam_human.json["dropped_episodes"].
        n_frames, n_eps, ts_all, dropped = 0, 0, [], []
        for pq in sorted((src / "data").glob("*/*.parquet")):
            try:
                n, ts = rewrite_parquet(pq, dst / "data" / pq.parent.name / pq.name, args.dry_run)
            except Exception as e:  # pyarrow.lib.ArrowInvalid / OSError on a truncated file
                ep = int(pq.stem.split("_")[-1])
                dropped.append(ep)
                print(f"[derive]   DROP episode {ep}: unreadable parquet {pq.name} ({pq.stat().st_size} B): {str(e)[:70]}", flush=True)
                continue
            n_frames += n
            n_eps += 1
            ts_all.append(ts)
        ts_all = np.concatenate(ts_all) if ts_all else np.zeros(0, np.float32)
        # meta
        info = build_info(src_info, names, note)
        episodes = [json.loads(l) for l in open(src / "meta" / "episodes.jsonl") if l.strip()]
        kept = [e for e in episodes if int(e["episode_index"]) not in set(dropped)]
        assert len(kept) == n_eps, (len(kept), n_eps, dropped)
        info["total_episodes"] = len(kept)
        info["total_frames"] = n_frames
        info["splits"] = {"train": f"0:{len(kept)}"}
        float_feats = [k for k, v in info["features"].items() if "float" in v["dtype"]]
        stats = {"observation.state": pooled["observation.state"], "action": pooled["action"]}
        if "timestamp" in float_feats:
            stats["timestamp"] = stats_1d(ts_all)
        missing = [k for k in float_feats if k not in stats]
        assert not missing, f"float features without stats (generate_stats would rewrite stats.json): {missing}"
        modality = build_modality(src_mod, args.view)
        assert modality["video"][args.view]["original_key"] in info["features"]
        if not args.dry_run:
            marker = {"human": True, "source": str(src), "note": note,
                      "dropped_episodes": sorted(dropped), "n_episodes": len(kept), "n_frames": n_frames}
            for name, payload in (("info.json", info), ("modality.json", modality), ("stats.json", stats),
                                  ("relative_stats.json", pooled_rel), ("wam_human.json", marker)):
                with open(dst / "meta" / name, "w") as f:
                    json.dump(payload, f, indent=4)
            shutil.copyfile(src / "meta" / "tasks.jsonl", dst / "meta" / "tasks.jsonl")
            with open(dst / "meta" / "episodes.jsonl", "w") as f:
                for e in kept:
                    f.write(json.dumps(e) + "\n")
            # sanity: every kept episode's parquet AND video exist (video through the symlink)
            for e in kept:
                ep = int(e["episode_index"])
                pp = dst / info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
                vp = dst / info["video_path"].format(episode_chunk=ep // info["chunks_size"],
                                                     video_key=modality["video"][args.view]["original_key"], episode_index=ep)
                assert pp.exists(), f"derived parquet missing: {pp}"
                assert vp.exists(), f"video missing through symlink: {vp}"
        print(f"[derive] {src}: {n_eps} episodes / {n_frames} frames (dropped {len(dropped)}: {sorted(dropped)}) → {dst}  "
              f"(maxrss {rss_mb():.0f} MiB)", flush=True)
    print(f"[derive] done{' (dry-run)' if args.dry_run else ''}  maxrss={rss_mb():.0f} MiB")


if __name__ == "__main__":
    main()
