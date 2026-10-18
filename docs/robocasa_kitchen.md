# RoboCasa kitchen (Wan2.2-5B)

## 1. Data

- Dataset: [kimtaey/robocasa_mg_gr00t_300](https://huggingface.co/datasets/kimtaey/robocasa_mg_gr00t_300)
  - The 24 `single_panda_gripper.*` kitchen tasks of [nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) with the first 300 of the 3,000 demonstrations per task

```bash
huggingface-cli download kimtaey/robocasa_mg_gr00t_300 --repo-type dataset --local-dir <dataset dir>
```

- Modality config: `fred/configs/data/robocasa_config.py`

## 2. Text cache

For efficiency the umT5 encoder is not loaded during training: the text embeddings of every task string are
precomputed once into a cache.

```bash
.venv/bin/python scripts/precompute_text_embeds.py --data-root <dataset dir> \
    --output-path ~/.cache/wam/robocasa300_wan22_umt5.pt --include-empty-prompt
```

- `--data-root`: dataset directory, repeatable; the cache holds every task string of all of them
- `--include-empty-prompt`: also embed the empty string, which caption dropout (`TEXT_DROPOUT`) samples
- `--append --output-path <cache.pt> --prompts '...'`: add prompts to an existing cache

## 3. Train

```bash
NUM_GPUS=4 GLOBAL_BATCH_SIZE=64 MAX_STEPS=100000 \
TEXT_EMBED_CACHE=~/.cache/wam/robocasa300_wan22_umt5.pt \
DATASETS=<dataset dir> MODEL_OUTPUT_DIR=<ckpt root> \
bash run_scripts/train/train_fred_wan22_5b_robocasa_kitchen.sh
```

## 4. Evaluation setup

```bash
bash fred/eval/sim/robocasa/setup_robocasa.sh kitchen      # venv: fred/eval/sim/robocasa/robocasa_kitchen_uv/.venv
```

## 5. Evaluate

```bash
TEXT_EMBED_CACHE=~/.cache/wam/robocasa300_wan22_umt5.pt \
bash run_scripts/eval/eval_fred.sh robocasa_kitchen <checkpoint dir> <output dir>
```

One task:

```bash
TEXT_EMBED_CACHE=~/.cache/wam/robocasa300_wan22_umt5.pt \
bash run_scripts/eval/eval_fred_task.sh robocasa_kitchen <checkpoint dir> <output dir> robocasa_panda_omron/TurnOnStove_PandaOmron_Env
```

**Note.** The kitchen eval scripts run `fred/eval/render_check.py` after each task. It moves episodes whose camera
images the MuJoCo/EGL renderer corrupted (one camera shown in all three panels, or two images alternating frame to
frame) to `<task dir>/render_quarantine/round<k>/` and re-runs them with the same seeds, up to 3 rounds. The policy
receives the broken images in these episodes, so they almost always fail. An episode still corrupted after the last
round keeps its result and is listed in `rollout.log`.
