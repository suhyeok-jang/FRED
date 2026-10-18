# DROID (Wan2.2-5B and Wan2.1-14B)

## 1. Data

- Dataset: [GEAR-Dreams/DreamZero-DROID-Data](https://huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data)
  - The DreamZero LeRobot export of DROID; `meta/relative_stats_dreamzero.json` ships with it

```bash
huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir <dataset dir>   # DATASETS
```

- Modality config: `fred/configs/data/droid_config.py` (arm action relative to the current joint state, gripper absolute, q01/q99 bounds)

## 2. Text cache

For efficiency the umT5 encoder is not loaded during training: the text embeddings of every task string are
precomputed once into a cache.
Both backbones read the same cache (the 14B uses the same umT5-XXL tokenizer).

```bash
.venv/bin/python scripts/precompute_text_embeds.py --data-root <dataset dir> \
    --output-path ~/.cache/wam/droid_wan22_umt5.pt --include-empty-prompt --trim-to-mask --batch-size 32
```

- `--data-root`: dataset directory, repeatable; the cache holds every task string of all of them
- `--include-empty-prompt`: also embed the empty string, which caption dropout (`TEXT_DROPOUT`) samples
- `--trim-to-mask`: store each embedding cut to its token length (DROID has ~123k task strings; untrimmed the cache
  is ~517 GB); the loader zero-pads it back
- `--batch-size`: umT5 encode batch
- `--append --output-path <cache.pt> --prompts '...'`: add prompts to an existing cache

## 3. Train

Wan2.2-5B:

```bash
NUM_GPUS=4 GLOBAL_BATCH_SIZE=64 MAX_STEPS=200000 \
TEXT_EMBED_CACHE=~/.cache/wam/droid_wan22_umt5.pt \
DATASETS=<dataset dir> MODEL_OUTPUT_DIR=<ckpt root> \
bash run_scripts/train/train_fred_wan22_5b_droid.sh
```

Wan2.1-I2V-14B (`train_fred_wan21_14b_droid.sh`), reference plate 16 GPUs × 4 clips (128 clips per step):

```bash
NUM_GPUS=16 GLOBAL_BATCH_SIZE=128 \
TEXT_EMBED_CACHE=~/.cache/wam/droid_wan22_umt5.pt \
DATASETS=<dataset dir> MODEL_OUTPUT_DIR=<ckpt root> \
bash run_scripts/train/train_fred_wan21_14b_droid.sh
```

## 4. Serving the policy

```bash
.venv/bin/python fred/eval/serve_policy.py --model-path <checkpoint dir> --host 0.0.0.0 --port 5555 \
    --text-embed-cache ~/.cache/wam/droid_wan22_umt5.pt
```

The server loads the EMA weights, drops the training-only modules and answers over ZMQ (`fred/policy/server_client.py`);
`--host 0.0.0.0` accepts connections from the robot computer (the default `127.0.0.1` only from the same machine).
Your robot loop talks to it with `PolicyClient`:

```python
from fred.policy.server_client import PolicyClient
client = PolicyClient(host="<server>", port=5555)
cfg = client.get_modality_config()           # the keys below
action, _ = client.get_action({
    "video.exterior_image_1_left": frames_left,      # (1, T, H, W, 3) uint8, T = len(cfg["video"].delta_indices)
    "video.exterior_image_2_left": frames_right,
    "video.wrist_image_left": frames_wrist,
    "state.joint_position": joints,                  # (1, 1, 7) float32
    "state.gripper_position": gripper,               # (1, 1, 1)
    "annotation.language.language_instruction": ["put the cup in the sink"],
})
# action["action.joint_position"]: (1, 24, 7) absolute joint targets (the relative prediction is already added to
# the current state); action["action.gripper_position"]: (1, 24, 1). Execute the first N steps, then query again.
```

- History frames: `cfg["video"].delta_indices` are the frame offsets relative to the current step (dataset steps at
  15 fps). Keep a ring buffer of camera frames and pass the frames at those offsets.
- `--action-seed` / `--video-seed` on the server make the sampled noise repeatable across runs.
