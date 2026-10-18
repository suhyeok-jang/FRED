#!/usr/bin/env python
"""Pooled OpenArm-28 statistics over the robot subsets. CPU-only, torch-free.

Why: the ShardedMixtureDataset merges per-dataset stats with a weighted mean/std and min/max of the
per-subset q01/q99 (sharded_mixture_dataset.py::merge_statistics) — not the true pooled quantiles.
Writing the SAME pooled stats into every subset makes that merge a no-op, so every subset is
normalised with identical, truly-pooled q01/q99 (``--use-percentiles``).

What it writes (idempotent; originals backed up ONCE as meta/<name>.subset_orig.json and always
re-read from the backup, so re-running never pools already-pooled numbers):
  <subset>/meta/stats.json           — original file with `observation.state` / `action` replaced by the
                                        pooled {mean,std,min,max,q01,q99} (28-dim lists, the exact schema
                                        of fred/data/stats.py::calculate_dataset_statistics). All other
                                        entries (torque, timestamp, …) are kept: fred/data/stats.py::
                                        generate_stats() REGENERATES stats.json at launch if any float
                                        feature of info.json is missing from it.
  <subset>/meta/relative_stats.json  — {left_arm_joints, right_arm_joints}: H×7 per-timestep
                                        {max,min,q01,q99,mean,std} in the exact format of
                                        fred/data/stats.py::calculate_stats_for_key (relative sample =
                                        action[t+k] − state[t] over k∈[0,H), float64 → float32, pooled
                                        over the RAW samples of all subsets, not over per-subset percentiles).
  <pooled-dir>/pooled_openarm28_stats.json, <pooled-dir>/pooled_openarm28_relative_stats.json — the pooled
                                        payloads (consumed by derive_human_as_openarm28.py and
                                        convert_human_clips_to_lerobot.py).

RSS: one parquet at a time; arrays N×28 float32; relative samples N×H×7 float32 per key (keys processed
sequentially, quantiles per timestep slice).

--out-suffix S: write the pooled payloads to meta/stats.<S>.json / meta/relative_stats.<S>.json (and
  <pooled-dir>/pooled_openarm28_{stats,relative_stats}.<S>.json) instead of overwriting meta/stats.json /
  meta/relative_stats.json, and skip the *.subset_orig.json backups (nothing is overwritten). This produces an
  alternative pooling (e.g. over another set of subsets) without changing the normalisation of runs already
  training on the in-place files; point the loader at it with
  EXTRA_ARGS="--data.stats-file stats.<S>.json --data.relative-stats-file relative_stats.<S>.json".

--write-also D... : extra dataset dirs that RECEIVE the pooled payloads but do NOT contribute to the pooling
  (derived human subsets, whose stats are by construction the pooled robot stats).

--action-horizon H (default 24, the horizon of openarm_config.py): the relative tables are PER-TIMESTEP,
  shape (H, D) per arm key, and fred/data/utils.py::normalize_values_minmax broadcasts them against the (H, D)
  action chunk — so H MUST equal the modality config's len(action.delta_indices). A mismatched table does not mis-slice, it raises
  (numpy: shapes (24,7) and (48,7) are not broadcastable). The ABSOLUTE stats.json is per-DIM and is
  horizon-independent (identical bytes for any H).

Usage: python scripts/openarm/pool_openarm_stats.py <robot subset dir> [...] --pooled-dir D
                [--write-also D...] [--action-horizon 24] [--out-suffix S] [--dry-run] [--skip-relative]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import DIM, rss_mb, write_json

ABS_COLUMNS = ["observation.state", "action"]
RELATIVE_KEYS = ["left_arm_joints", "right_arm_joints"]   # openarm_config.py RELATIVE groups
ACTION_HORIZON = 24                                        # openarm_config.py action horizon; --action-horizon overrides
STAT_KEYS = ["mean", "std", "min", "max", "q01", "q99"]    # stats.json schema (fred/data/stats.py)
REL_STAT_KEYS = ["max", "min", "q01", "q99", "mean", "std"]  # relative_stats.json order (calculate_stats_for_key)


def abs_stats(x: np.ndarray) -> dict[str, list[float]]:
    """EXACT formula of fred/data/stats.py::calculate_dataset_statistics on a float32 (N, D) array."""
    x = np.asarray(x, dtype=np.float32)
    return dict(
        mean=np.mean(x, axis=0).tolist(),
        std=np.std(x, axis=0).tolist(),
        min=np.min(x, axis=0).tolist(),
        max=np.max(x, axis=0).tolist(),
        q01=np.quantile(x, 0.01, axis=0).tolist(),
        q99=np.quantile(x, 0.99, axis=0).tolist(),
    )


def relative_samples(state: np.ndarray, action: np.ndarray, horizon: int = ACTION_HORIZON) -> np.ndarray:
    """(usable, horizon, D) float32 relative windows: action[t+k] − state[t], k∈[0,horizon).

    Mirrors fred/data/stats.py::RelativeActionLoader.load_relative_actions for NON_EEF keys
    (JointPose.__sub__ = float64 subtraction, then np.stack(dtype=float32)); state delta [0],
    action deltas range(horizon) → usable = len − (horizon−1)."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    usable = len(action) - (horizon - 1)
    if usable <= 0:
        return np.zeros((0, horizon, action.shape[1]), dtype=np.float32)
    idx = np.arange(usable)[:, None] + np.arange(horizon)[None, :]        # (usable, horizon)
    rel = action[idx] - state[:usable, None, :]                             # (usable, horizon, D)
    return rel.astype(np.float32)


def rel_stats(samples: np.ndarray) -> dict[str, np.ndarray]:
    """calculate_stats_for_key's 6 stats over axis 0 of (N, horizon, D) — quantiles per timestep slice
    (memory) which equals np.quantile(samples, q, axis=0)."""
    n, h, d = samples.shape
    out = {k: np.zeros((h, d), dtype=np.float64) for k in REL_STAT_KEYS}
    for t in range(h):
        s = samples[:, t, :]
        out["max"][t] = np.max(s, axis=0)
        out["min"][t] = np.min(s, axis=0)
        out["q01"][t] = np.quantile(s, 0.01, axis=0)
        out["q99"][t] = np.quantile(s, 0.99, axis=0)
        out["mean"][t] = np.mean(s, axis=0)
        out["std"][t] = np.std(s, axis=0)
    return out


def merge_into_stats_json(orig: dict, pooled: dict[str, dict]) -> dict:
    """Return orig with the pooled columns replaced (other entries untouched, key order kept)."""
    out = {}
    for k, v in orig.items():
        out[k] = pooled[k] if k in pooled else v
    for k, v in pooled.items():
        if k not in out:
            out[k] = v
    return out


def _load_orig(meta: Path, name: str) -> dict | None:
    """Original (pre-pool) payload: the backup if it exists, else the current file (or None)."""
    bak = meta / f"{name}.subset_orig.json"
    cur = meta / f"{name}.json"
    if bak.exists():
        return json.load(open(bak))
    if cur.exists():
        return json.load(open(cur))
    return None


def _backup_once(meta: Path, name: str, dry_run: bool) -> None:
    bak = meta / f"{name}.subset_orig.json"
    cur = meta / f"{name}.json"
    if cur.exists() and not bak.exists() and not dry_run:
        bak.write_bytes(cur.read_bytes())


def _suffixed(stem: str, out_suffix: str | None) -> str:
    """'stats' -> 'stats.json' (no suffix) / 'stats.pool7.json' (--out-suffix pool7)."""
    return f"{stem}.json" if not out_suffix else f"{stem}.{out_suffix}.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Pool the OpenArm-28 statistics over robot subsets.")
    ap.add_argument("subsets", nargs="+", help="robot subset directories to pool over")
    ap.add_argument("--pooled-dir", required=True,
                    help="directory that receives pooled_openarm28_stats.json / pooled_openarm28_relative_stats.json")
    ap.add_argument("--write-also", nargs="*", default=[],
                    help="extra dataset dirs that receive the pooled payloads but are not pooled over "
                         "(derived human subsets)")
    ap.add_argument("--action-horizon", type=int, default=ACTION_HORIZON,
                    help="H of the per-timestep relative tables; MUST equal len(action.delta_indices) of the "
                         "modality config the run uses (openarm_config.py: 24)")
    ap.add_argument("--out-suffix", default=None,
                    help="write meta/stats.<S>.json / meta/relative_stats.<S>.json instead of overwriting "
                         "meta/stats.json / meta/relative_stats.json (no *.subset_orig.json backup)")
    ap.add_argument("--dry-run", action="store_true", help="compute + report, write nothing")
    ap.add_argument("--skip-relative", action="store_true")
    args = ap.parse_args()
    assert args.action_horizon >= 1, f"--action-horizon must be >= 1, got {args.action_horizon}"
    assert args.out_suffix is None or args.out_suffix.replace("_", "").isalnum(), \
        f"--out-suffix must be alphanumeric/underscore, got {args.out_suffix!r}"
    subsets = [Path(s) for s in args.subsets]
    write_also = [Path(s) for s in args.write_also]
    pooled_dir = Path(args.pooled_dir)
    if not args.dry_run:
        pooled_dir.mkdir(parents=True, exist_ok=True)
    for d in write_also:
        assert (d / "meta").is_dir(), f"missing --write-also subset {d}"
    for d in subsets:
        assert (d / "meta" / "modality.json").exists(), f"missing subset {d}"

    # ── modality slices (identical across subsets; asserted) ──
    mod0 = json.load(open(subsets[0] / "meta" / "modality.json"))
    slices = {k: (v["start"], v["end"]) for k, v in mod0["state"].items()}
    for d in subsets[1:]:
        m = json.load(open(d / "meta" / "modality.json"))
        assert {k: (v["start"], v["end"]) for k, v in m["state"].items()} == slices, f"{d}: state layout differs"
        assert {k: (v["start"], v["end"]) for k, v in m["action"].items()} == slices, f"{d}: action layout differs"

    # ── pass 1: absolute stats over observation.state / action (stream parquet by parquet) ──
    cols: dict[str, list[np.ndarray]] = {c: [] for c in ABS_COLUMNS}
    rel_parts: dict[str, list[np.ndarray]] = {k: [] for k in RELATIVE_KEYS}
    per_subset_frames: dict[str, int] = {}
    for d in subsets:
        n = 0
        for pq in sorted((d / "data").glob("*/*.parquet")):
            df = pd.read_parquet(pq, columns=ABS_COLUMNS)
            arrs = {c: np.vstack([np.asarray(x, dtype=np.float32) for x in df[c]]) for c in ABS_COLUMNS}
            for c in ABS_COLUMNS:
                assert arrs[c].shape[1] == DIM, (pq, c, arrs[c].shape)
                cols[c].append(arrs[c])
            n += len(df)
            if not args.skip_relative:
                for k in RELATIVE_KEYS:
                    s0, s1 = slices[k]
                    rel_parts[k].append(relative_samples(arrs["observation.state"][:, s0:s1], arrs["action"][:, s0:s1],
                                                         horizon=args.action_horizon))
        per_subset_frames[str(d)] = n
        print(f"[pool] {d}: {n} frames  (maxrss {rss_mb():.0f} MiB)", flush=True)
    pooled_abs = {c: abs_stats(np.concatenate(cols[c], axis=0)) for c in ABS_COLUMNS}
    total = sum(per_subset_frames.values())
    print(f"[pool] action horizon (relative tables) H={args.action_horizon}")
    print(f"[pool] total frames {total}: " + ", ".join(f"{k}={v}" for k, v in per_subset_frames.items()))
    for c in ABS_COLUMNS:
        print(f"[pool] {c} pooled ranges per group:")
        for g, (s0, s1) in slices.items():
            q01 = np.round(pooled_abs[c]["q01"][s0:s1], 3).tolist()
            q99 = np.round(pooled_abs[c]["q99"][s0:s1], 3).tolist()
            mn = np.round(pooled_abs[c]["min"][s0:s1], 3).tolist()
            mx = np.round(pooled_abs[c]["max"][s0:s1], 3).tolist()
            print(f"    {g:18s} q01={q01}\n    {'':18s} q99={q99}\n    {'':18s} min={mn}\n    {'':18s} max={mx}")
    del cols

    # ── pass 2: pooled relative stats (raw samples concatenated, one key at a time) ──
    pooled_rel: dict[str, dict[str, list]] = {}
    if not args.skip_relative:
        for k in RELATIVE_KEYS:
            samples = np.concatenate(rel_parts[k], axis=0)
            rel_parts[k] = []
            st = rel_stats(samples)
            pooled_rel[k] = {n: st[n].tolist() for n in REL_STAT_KEYS}
            _tl = samples.shape[1] - 1   # last timestep of the table (H-1)
            print(f"[pool] relative {k}: {samples.shape[0]} windows × {samples.shape[1]}×{samples.shape[2]}; "
                  f"q01[t={_tl}]={np.round(st['q01'][-1], 3).tolist()} q99[t={_tl}]={np.round(st['q99'][-1], 3).tolist()} "
                  f"min={np.round(st['min'].min(0), 3).tolist()} max={np.round(st['max'].max(0), 3).tolist()} "
                  f"(maxrss {rss_mb():.0f} MiB)", flush=True)
            del samples, st

    # ── write ──
    sfx = args.out_suffix
    write_json(pooled_dir / _suffixed("pooled_openarm28_stats", sfx), pooled_abs, args.dry_run)
    if pooled_rel:
        write_json(pooled_dir / _suffixed("pooled_openarm28_relative_stats", sfx), pooled_rel, args.dry_run)
    stats_name = _suffixed("stats", sfx)
    rel_name = _suffixed("relative_stats", sfx)
    for d in list(subsets) + list(write_also):
        meta = d / "meta"
        # --write-also dirs are derived (their stats ARE the pooled robot stats), so a *.subset_orig.json there
        # would be the previous pooling, not an original; never back them up (the previous pooling is kept at
        # <pooled-dir>/pooled_openarm28_{stats,relative_stats}.json).
        derived = d in write_also
        orig = _load_orig(meta, "stats")
        assert orig is not None, f"{meta}/stats.json missing"
        if not sfx and not derived:   # only the in-place mode overwrites, so only it needs a backup
            _backup_once(meta, "stats", args.dry_run)
        write_json(meta / stats_name, merge_into_stats_json(orig, pooled_abs), args.dry_run)
        if pooled_rel:
            orig_rel = _load_orig(meta, "relative_stats") or {}
            if not sfx and not derived:
                _backup_once(meta, "relative_stats", args.dry_run)
            write_json(meta / rel_name, merge_into_stats_json(orig_rel, pooled_rel), args.dry_run)
        print(f"[pool] {'(dry-run) ' if args.dry_run else ''}wrote {meta}/{stats_name}"
              + (f" + {rel_name}" if pooled_rel else ""))
    print(f"[pool] done  maxrss={rss_mb():.0f} MiB")


if __name__ == "__main__":
    main()
