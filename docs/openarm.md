# OpenArm humanoid (Wan2.2-5B)

This page describes the data layout the two OpenArm recipes expect.

## 1. Data

Robot subsets are LeRobot v2 directories, one per collection, listed in `DATASETS` with one weight each in
`MIX_RATIOS`. Every subset exposes the 28-dim OpenArm state/action layout of `fred/configs/data/openarm_config.py`:

| key | dims | action representation |
|---|---|---|
| `neck_joints` | 2 | absolute |
| `left_arm_joints` | 7 | relative (action − current state of the same key) |
| `right_arm_joints` | 7 | relative |
| `left_hand_joints` | 6 | absolute |
| `right_hand_joints` | 6 | absolute |

with two egocentric cameras `camera_ego_left`, `camera_ego_right` (192×256, stacked vertically into one canvas), the
instruction under `annotation.human.action.task_description`, 24-step action chunks at frame stride 3, and q01/q99
normalisation pooled over the subsets (`scripts/openarm/pool_openarm_stats.py` writes the same pooled
`meta/stats.json` / `meta/relative_stats.json` into every subset). `scripts/openarm/prep_openarm_meta.py` adds
the camera / annotation aliases a subset may be missing.

Two optional markers in `meta/` change how a subset is used:

- `meta/wam_human.json` (`{"human": true}`): a human-video subset. It trains the video model only; its state and
  action columns are zeros of the same 28-dim shape (`scripts/openarm/derive_human_as_openarm28.py` builds such
  copies of LeRobot human-video subsets, `scripts/openarm/convert_human_clips_to_lerobot.py` of raw stereo clips).
  Requires `HUMAN_DATASETS=1`.
- `meta/wam_action_groups.json`: the subset carries only some of the five action groups; the others are zero-filled
  and excluded from the loss (`prep_openarm_meta.py --action-groups`). Requires `DATASET_ACTION_GROUPS=1`.

A minimal subset:

```
<subset>/
  meta/info.json            LeRobot v2 (fps, features: observation.state [28], action [28], two video keys)
  meta/modality.json        state/action groups in the order above; video camera_ego_left/right
  meta/tasks.jsonl          instruction strings
  meta/episodes.jsonl
  meta/stats.json           pooled q01/q99 (pool_openarm_stats.py)
  meta/relative_stats.json  pooled bounds of the arm deltas, (24, 7) per arm key
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.camera_ego_left/episode_000000.mp4
  videos/chunk-000/observation.images.camera_ego_right/episode_000000.mp4
```

Preparing the subsets (each script's `--help` lists its inputs and outputs):

```bash
# camera / annotation aliases; --action-groups marks a subset that carries only some action groups
.venv/bin/python scripts/openarm/prep_openarm_meta.py <robot subset> [...] \
    [--action-groups <robot subset>=right_arm_joints,right_hand_joints]
# pooled q01/q99 and per-timestep arm-delta bounds (H=24), written into every subset and into <stats dir>
.venv/bin/python scripts/openarm/pool_openarm_stats.py <robot subset> [...] --pooled-dir <stats dir>
# human subsets (video only): zero-filled 28-dim copies of LeRobot human-video subsets ...
.venv/bin/python scripts/openarm/derive_human_as_openarm28.py <human subset> [...] --out-root <dir> \
    --names-from <robot subset> --pooled-stats <stats dir>/pooled_openarm28_stats.json \
    --pooled-rel-stats <stats dir>/pooled_openarm28_relative_stats.json
# ... or one dataset built from stereo human clips (episode_manifest.json + video/<category>/*_{left,right}.mp4)
.venv/bin/python scripts/openarm/convert_human_clips_to_lerobot.py --src <clips dir> --dst <dir> \
    --ref-subset <derived human subset> --names-from <robot subset> \
    --pooled-stats <stats dir>/pooled_openarm28_stats.json \
    --pooled-rel-stats <stats dir>/pooled_openarm28_relative_stats.json
```

## 2. Text cache

For efficiency the umT5 encoder is not loaded during training: the text embeddings of every task string are
precomputed once into a cache.

```bash
.venv/bin/python scripts/precompute_text_embeds.py --data-root <subset> [--data-root ...] \
    --output-path ~/.cache/wam/openarm_wan22_umt5.pt --include-empty-prompt
```

- `--data-root`: dataset directory, repeatable; the cache holds every task string of all of them
- `--include-empty-prompt`: also embed the empty string, which caption dropout (`TEXT_DROPOUT`) samples
- `--append --output-path <cache.pt> --prompts '...'`: add prompts to an existing cache

## 3. Train

Robot data only (32 clips per step, 50K iterations in the paper):

```bash
NUM_GPUS=4 MAX_STEPS=50000 WARMUP_STEPS=2500 TEXT_EMBED_CACHE=~/.cache/wam/openarm_wan22_umt5.pt \
DATASETS="<robot subset> ..." MIX_RATIOS="<episode counts or weights>" MODEL_OUTPUT_DIR=<ckpt root> \
bash run_scripts/train/train_fred_wan22_5b_openarm_robot.sh
```

Robot + human co-training (every batch 32 robot + 32 human clips):

```bash
NUM_GPUS=4 MAX_STEPS=50000 WARMUP_STEPS=2500 TEXT_EMBED_CACHE=~/.cache/wam/openarm_wan22_umt5.pt \
DATASETS="<robot subsets> <human subsets>" MIX_RATIOS="<weights, same order>" MODEL_OUTPUT_DIR=<ckpt root> \
bash run_scripts/train/train_fred_wan22_5b_openarm_robot_human.sh
```

## 4. Evaluate

OpenArm has no simulator; serve the checkpoint to the robot as in `docs/droid.md`.
