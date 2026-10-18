#!/usr/bin/env python
"""Convert stereo human clips into ONE video-only LeRobot v2.1 dataset shaped like the human subsets of
derive_human_as_openarm28.py. CPU-only, torch-free (numpy/pandas/PyAV).

Input  : --src with episode_manifest.json {"ep<N>_<category>": {"include": bool, "category": str, "task": str},
         ...} (only `include: true` entries are converted and must exist as files) +
         video/<category>/ep<N>_<category>_{left,right}.mp4 (256x192, 30 fps, h264, trimmed to the motion,
         frame index starts at 0 per clip). No robot data / state / action.
Output : --dst
  data/chunk-000/episode_XXXXXX.parquet  observation.state / action = float32 zeros(28), language_instruction,
                                         next.done, timestamp, frame_index, episode_index, index, task_index;
                                         frame count = decoded frame count of the clip (PyAV). If left/right
                                         differ the MIN is used and the episode is listed in
                                         meta/wam_human.json["lr_mismatch"].
  videos/chunk-000/observation.images.camera_ego_left/episode_XXXXXX.mp4   ← COPY of the *_left.mp4
  videos/chunk-000/observation.images.camera_ego_right/episode_XXXXXX.mp4  ← COPY of the *_right.mp4
         (left = camera_ego_left = TOP of the robot vconcat canvas, right = bottom — same as the robot subsets)
  meta/info.json       feature schema of --ref-subset (a derive_human_as_openarm28.py output) + BOTH video
                       keys, fps 30
  meta/modality.json   robot 28-dim state/action groups, video {camera_ego_left, camera_ego_right}, both
                       annotation keys → task_index
  meta/tasks.jsonl     task strings VERBATIM from the manifest's per-episode `task`
  meta/episodes.jsonl  episode_index / tasks / length / source id / category
  meta/stats.json      = --pooled-stats (the pooling every other subset carries) + timestamp stats;
                       meta/relative_stats.json = --pooled-rel-stats
  meta/wam_human.json  {"human": true, ...} → rows are video-only under ``--human-datasets`` (launcher
                       HUMAN_DATASETS=1): state 0, action 0, action_mask 0. BOTH views are present so
                       nothing is rendered black by the missing-view path and ``--view-mask-probs`` applies
                       exactly as for a robot sample.

Episode order: manifest entries with include=true sorted by (category, ep number) → episode_index 0..N-1.
`--exclude-category C` (repeatable) drops whole categories BEFORE indexing, so episode_index / index / task_index
stay contiguous; the excluded list is recorded in meta/wam_human.json["excluded_categories"] and
info.json["_note"]. total_videos = episodes x 2 views.
Idempotent: parquet/meta are rewritten every run; a video is copied only if missing or size-different — when
changing the episode set, delete the old --dst first so stale higher-index files do not survive.

Usage: .venv/bin/python scripts/openarm/convert_human_clips_to_lerobot.py --src S --dst D --ref-subset R
           --names-from <robot subset dir> --pooled-stats F --pooled-rel-stats F [--exclude-category C ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ANNOTATION_KEYS, DIM, OPENARM28_GROUPS, joint_names, rss_mb, stats_1d, write_json

FPS = 30
VIEWS = {"left": "camera_ego_left", "right": "camera_ego_right"}   # source suffix → unified key
_EP_RE = re.compile(r"^ep(\d+)_(.+)$")


def count_frames(path: Path) -> tuple[int, int, int, float]:
    """(decoded frames, width, height, fps) via PyAV — decode everything, do not trust stream.frames."""
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        n = sum(1 for _ in c.decode(s))
        return n, int(s.codec_context.width), int(s.codec_context.height), float(s.average_rate or 0)


def _copy_if_needed(src: Path, dst: Path, dry_run: bool) -> bool:
    if dst.exists() and not dst.is_symlink() and dst.stat().st_size == src.stat().st_size:
        return False
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".mp4.tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert stereo human clips into a video-only OpenArm-28 dataset.")
    ap.add_argument("--src", required=True, help="directory with episode_manifest.json and video/")
    ap.add_argument("--dst", required=True, help="output dataset directory")
    ap.add_argument("--ref-subset", required=True,
                    help="a human subset written by derive_human_as_openarm28.py (feature schema / modality template)")
    ap.add_argument("--names-from", required=True,
                    help="robot subset whose meta/info.json action names label the 28 dims")
    ap.add_argument("--pooled-stats", required=True, help="pooled_openarm28_stats.json of pool_openarm_stats.py")
    ap.add_argument("--pooled-rel-stats", required=True,
                    help="pooled_openarm28_relative_stats.json of pool_openarm_stats.py")
    ap.add_argument("--exclude-category", action="append", default=[], metavar="C",
                    help="drop every episode of this manifest category (repeatable); indices stay contiguous")
    ap.add_argument("--dry-run", action="store_true", help="count frames + report; write nothing")
    args = ap.parse_args()
    src, dst, ref = Path(args.src), Path(args.dst), Path(args.ref_subset)

    manifest = json.load(open(src / "episode_manifest.json"))
    ref_info = json.load(open(ref / "meta" / "info.json"))
    ref_mod = json.load(open(ref / "meta" / "modality.json"))
    pooled = json.load(open(args.pooled_stats))
    pooled_rel = json.load(open(args.pooled_rel_stats))
    names = joint_names(args.names_from)
    assert all(len(pooled[c]["q01"]) == DIM for c in ("observation.state", "action"))

    # ── episode list: include=true, minus --exclude-category, sorted by (category, ep number) ──
    excluded = sorted(set(args.exclude_category))
    all_cats = {v["category"] for v in manifest.values() if v.get("include")}
    unknown = [c for c in excluded if c not in all_cats]
    assert not unknown, f"--exclude-category not in manifest: {unknown} (have {sorted(all_cats)})"
    eps, n_excluded = [], 0
    for ep_id, info in manifest.items():
        if not info.get("include"):
            continue
        m = _EP_RE.match(ep_id)
        assert m and m.group(2) == info["category"], (ep_id, info)
        if info["category"] in excluded:
            n_excluded += 1
            continue
        eps.append((info["category"], int(m.group(1)), ep_id, info["task"]))
    eps.sort()
    n_total = sum(1 for v in manifest.values() if v.get("include"))
    assert len(eps) + n_excluded == n_total
    print(f"[convert] manifest: {len(manifest)} entries, include=true {n_total}, excluded categories {excluded} "
          f"({n_excluded} episodes) -> {len(eps)} episodes", flush=True)

    # tasks: verbatim strings, index by sorted order (deterministic)
    task_strings = sorted({t for _, _, _, t in eps})
    task_index = {t: i for i, t in enumerate(task_strings)}

    # ── info.json from the reference schema: swap the video feature(s) for BOTH views ──
    info = json.loads(json.dumps(ref_info))
    feats = {}
    ref_video_feat = None
    for k, v in ref_info["features"].items():
        if v.get("dtype") == "video":
            ref_video_feat = ref_video_feat or v
            continue
        feats[k] = v
    assert ref_video_feat is not None
    for suffix, key in VIEWS.items():
        vf = json.loads(json.dumps(ref_video_feat))
        vf["description"] = f"Camera view: {key} ({suffix} clip of the stereo pair)"
        vf["video_info"]["fps"] = FPS
        vf["info"]["video.fps"] = FPS
        vf["info"].pop("video.preprocess", None)   # the reference subset's note does not describe these clips
        feats[f"observation.images.{key}"] = vf
    for c in ("observation.state", "action"):
        feats[c]["names"] = list(names)
    info["features"] = feats
    info["robot_type"] = "human_clips_as_openarm28_human"
    info["fps"] = FPS
    info["total_videos"] = len(eps) * len(VIEWS)
    info["total_tasks"] = len(task_strings)
    info["total_chunks"] = 1
    for k in ("action_format", "active_hands"):
        info.pop(k, None)
    note = (f"human video-only LeRobot dataset built from stereo human clips: observation.state/action zero-filled "
            f"28-dim (OpenArm layout), BOTH stereo views copied (left=camera_ego_left top / right=camera_ego_right bottom), stats = pooled "
            f"robot stats; loader marks rows is_human (--human-datasets)"
            + (f"; excluded categories: {excluded} ({n_excluded} episodes)" if excluded else ""))
    info["_note"] = note

    modality = {
        "state": {k: {"start": s, "end": e} for k, (s, e) in OPENARM28_GROUPS.items()},
        "action": {k: {"start": s, "end": e} for k, (s, e) in OPENARM28_GROUPS.items()},
        "video": {key: {"original_key": f"observation.images.{key}"} for key in VIEWS.values()},
        "annotation": {k: {"original_key": "task_index"} for k in ANNOTATION_KEYS},
    }
    assert set(modality["state"]) == set(ref_mod["state"]) and modality["annotation"].keys() == ref_mod["annotation"].keys()

    # ── per-episode: frame counts, video copies, parquet ──
    chunks = int(info["chunks_size"])
    episodes_meta, lr_mismatch, per_cat = [], [], {}
    ts_all, n_frames_total, n_copied, global_index = [], 0, 0, 0
    t0 = time.time()
    for ep_idx, (cat, _n, ep_id, task) in enumerate(eps):
        counts = {}
        for suffix, key in VIEWS.items():
            vsrc = src / "video" / cat / f"{ep_id}_{suffix}.mp4"
            assert vsrc.exists(), f"missing clip {vsrc}"
            n, w, h, fps = count_frames(vsrc)
            assert (w, h) == (256, 192), (vsrc, w, h)
            assert abs(fps - FPS) < 0.5, (vsrc, fps)
            counts[suffix] = n
            vdst = dst / info["video_path"].format(episode_chunk=ep_idx // chunks,
                                                  video_key=f"observation.images.{key}", episode_index=ep_idx)
            n_copied += int(_copy_if_needed(vsrc, vdst, args.dry_run))
        n = min(counts.values())
        if counts["left"] != counts["right"]:
            lr_mismatch.append({"episode_index": ep_idx, "id": ep_id, "left": counts["left"], "right": counts["right"], "used": n})
        assert n > 0, (ep_id, counts)
        ts = (np.arange(n, dtype=np.float32) / FPS).astype(np.float32)
        zeros = [np.zeros(DIM, dtype=np.float32) for _ in range(n)]
        df = pd.DataFrame({
            "observation.state": zeros,
            "action": [z.copy() for z in zeros],
            "language_instruction": [task] * n,
            "next.done": np.array([False] * (n - 1) + [True]),
            "timestamp": ts,
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ep_idx, dtype=np.int64),
            "index": np.arange(global_index, global_index + n, dtype=np.int64),
            "task_index": np.full(n, task_index[task], dtype=np.int64),
        })
        pq = dst / info["data_path"].format(episode_chunk=ep_idx // chunks, episode_index=ep_idx)
        if not args.dry_run:
            pq.parent.mkdir(parents=True, exist_ok=True)
            tmp = pq.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False)
            os.replace(tmp, pq)
        episodes_meta.append({"episode_index": ep_idx, "tasks": [task], "length": n,
                              "source_episode_id": ep_id, "category": cat})
        c = per_cat.setdefault(cat, {"episodes": 0, "frames": 0})
        c["episodes"] += 1
        c["frames"] += n
        ts_all.append(ts)
        n_frames_total += n
        global_index += n
        if (ep_idx + 1) % 50 == 0 or ep_idx + 1 == len(eps):
            print(f"[convert] {ep_idx + 1}/{len(eps)} episodes, {n_frames_total} frames, {n_copied} videos copied "
                  f"({time.time() - t0:.0f}s, maxrss {rss_mb():.0f} MiB)", flush=True)

    info["total_episodes"] = len(eps)
    info["total_frames"] = n_frames_total
    info["splits"] = {"train": f"0:{len(eps)}"}
    float_feats = [k for k, v in info["features"].items() if "float" in v["dtype"]]
    stats = {"observation.state": pooled["observation.state"], "action": pooled["action"],
             "timestamp": stats_1d(np.concatenate(ts_all))}
    missing = [k for k in float_feats if k not in stats]
    assert not missing, f"float features without stats (generate_stats would rewrite stats.json): {missing}"
    marker = {"human": True, "source": str(src), "note": note, "dropped_episodes": [],
              "excluded_categories": excluded, "n_excluded_episodes": n_excluded,
              "lr_mismatch": lr_mismatch, "n_episodes": len(eps), "n_frames": n_frames_total,
              "views": list(VIEWS.values()), "per_category": per_cat}

    if not args.dry_run:
        (dst / "meta").mkdir(parents=True, exist_ok=True)
        for name, payload in (("info.json", info), ("modality.json", modality), ("stats.json", stats),
                              ("relative_stats.json", pooled_rel), ("wam_human.json", marker)):
            write_json(dst / "meta" / name, payload)
        with open(dst / "meta" / "tasks.jsonl.tmp", "w") as f:
            for t in task_strings:
                f.write(json.dumps({"task_index": task_index[t], "task": t}) + "\n")
        os.replace(dst / "meta" / "tasks.jsonl.tmp", dst / "meta" / "tasks.jsonl")
        with open(dst / "meta" / "episodes.jsonl.tmp", "w") as f:
            for e in episodes_meta:
                f.write(json.dumps(e) + "\n")
        os.replace(dst / "meta" / "episodes.jsonl.tmp", dst / "meta" / "episodes.jsonl")
        # sanity: every episode's parquet + both videos resolve through the loader's own formula
        for e in episodes_meta:
            ep = e["episode_index"]
            assert (dst / info["data_path"].format(episode_chunk=ep // chunks, episode_index=ep)).exists()
            for key in VIEWS.values():
                vp = dst / info["video_path"].format(episode_chunk=ep // chunks, video_key=f"observation.images.{key}", episode_index=ep)
                assert vp.exists() and not vp.is_symlink(), vp

    print("[convert] per category: " + ", ".join(f"{k}={v['episodes']}ep/{v['frames']}f" for k, v in sorted(per_cat.items())))
    print(f"[convert] {len(eps)} episodes / {n_frames_total} frames / {len(task_strings)} tasks; "
          f"left/right frame-count mismatches: {len(lr_mismatch)} {lr_mismatch[:5]}")
    print(f"[convert] {'(dry-run) ' if args.dry_run else ''}done → {dst}  maxrss={rss_mb():.0f} MiB")


if __name__ == "__main__":
    main()
