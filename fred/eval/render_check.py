"""Find episodes whose camera images were corrupted by the renderer, and move them aside so the rollout re-runs them.

On the MuJoCo/EGL renderer a small share of kitchen episodes record broken views: the three panels show one camera,
or two images alternate frame to frame. The policy sees the same broken images, so those episodes almost always
fail (b256 80k, seed 42, 1 env: 46 of 1200 flagged, 45 of them failures). They measure the renderer, not the policy.

eval_fred.sh / eval_fred_task.sh call this after every rollout:
    render_check.py <task dir> --quarantine <round>   # flag, move the flagged mp4s to <task dir>/render_quarantine/round<k>/
    (rollout_policy.py then refills exactly those episode ids with their own seeds, and the CSV row is upserted)
    render_check.py <task dir>                        # report only: what is still flagged after the last round
An episode still flagged after the last round keeps its result, and the report names it.

Detector (unchanged from the pre-fred eval's swap scan): frames decoded as 192x64 grey;
  gap  = median corr(frame t, t+2) - median corr(frame t, t+1) over a window from n/3   (period-2 alternation)
  med  = median mean |frame t+1 - frame t|,  lag1m = median corr of consecutive 16x16 thumbnails
  corrupted  <=>  gap > 0.12  or  (med > 20 and lag1m < 0.7)   (fast real motion keeps lag1m >= 0.7)
Clips under 40 frames (an early success) are too short to judge and are kept; an undecodable file is flagged.
Results are cached in <task dir>/render_check.jsonl by (name, size, mtime).
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import av
import numpy as np

MP4 = re.compile(r".*-episode_\d+-(success|failure)\.mp4$")
GAP_TH, MED_TH, LAG1M_TH, MIN_FRAMES = 0.12, 20.0, 0.7, 40


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def episode_signature(path: Path) -> dict | None:
    """None = the file does not decode; {'short': n} = too few frames to judge; else med / gap / lag1m."""
    try:
        with av.open(str(path)) as container:
            frames = [f.reformat(width=192, height=64, format="gray").to_ndarray() for f in container.decode(video=0)]
    except (av.error.FFmpegError, ValueError):
        return None
    n = len(frames)
    if n < MIN_FRAMES:
        return {"short": n}
    g = np.stack(frames).astype(np.float32)                    # (n, 64, 192)
    med = float(np.median(np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))))
    a = n // 3
    seg = g[a:min(n, a + 80)]
    with np.errstate(invalid="ignore", divide="ignore"):
        lag1 = np.median([_corr(seg[i], seg[i + 1]) for i in range(len(seg) - 1)])
        lag2 = np.median([_corr(seg[i], seg[i + 2]) for i in range(len(seg) - 2)])
        thumbs = seg[:60].reshape(-1, 16, 4, 16, 12).mean(axis=(2, 4))   # 16x16 block means
        lag1m = np.median([_corr(thumbs[i], thumbs[i + 1]) for i in range(len(thumbs) - 1)])
    return {"med": round(med, 2), "gap": round(float(lag2 - lag1), 3), "lag1m": round(float(lag1m), 3)}


def is_corrupted(sig: dict | None) -> bool:
    if sig is None:
        return True
    if "short" in sig:
        return False
    return sig["gap"] > GAP_TH or (sig["med"] > MED_TH and sig["lag1m"] < LAG1M_TH)


def scan(task_dir: Path) -> list[dict]:
    cache_path = task_dir / "render_check.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            cache[(r["file"], r["size"], r["mtime_ns"])] = r
    results = []
    for p in sorted(task_dir.glob("*.mp4")):
        if not MP4.match(p.name):
            continue                                            # an unfinished episode (no status suffix yet)
        st = p.stat()
        key = (p.name, st.st_size, st.st_mtime_ns)
        if key not in cache:
            sig = episode_signature(p)
            cache[key] = {"file": p.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sig": sig,
                          "corrupted": is_corrupted(sig)}
        results.append(cache[key])
    with open(cache_path, "w") as f:
        for r in cache.values():
            if (task_dir / r["file"]).exists():
                f.write(json.dumps(r) + "\n")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--quarantine", type=int, metavar="ROUND",
                    help="move the flagged mp4s to <task dir>/render_quarantine/round<ROUND>/ so the rollout re-runs them")
    args = ap.parse_args()
    results = scan(args.task_dir)
    bad = [r for r in results if r["corrupted"]]
    if args.quarantine is not None and bad:
        dest = args.task_dir / "render_quarantine" / f"round{args.quarantine}"
        dest.mkdir(parents=True, exist_ok=True)
        for r in bad:
            shutil.move(str(args.task_dir / r["file"]), str(dest / r["file"]))
        with open(dest / "manifest.json", "w") as f:
            json.dump(bad, f, indent=1)
        print(f"[render-check] {args.task_dir.name}: round {args.quarantine}: {len(bad)}/{len(results)} episodes "
              f"render-corrupted, moved to {dest} for a re-run", file=sys.stderr)
    elif bad:
        print(f"[render-check] {args.task_dir.name}: {len(bad)}/{len(results)} episodes still render-corrupted after "
              f"the re-runs, kept as recorded: {[r['file'] for r in bad]}", file=sys.stderr)
    else:
        print(f"[render-check] {args.task_dir.name}: {len(results)} episodes, none render-corrupted", file=sys.stderr)
    print(len(bad))                                             # stdout: the count, for the shell loop
    return 0


if __name__ == "__main__":
    sys.exit(main())
