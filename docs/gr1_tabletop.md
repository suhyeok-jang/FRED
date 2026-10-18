# GR-1 tabletop (Wan2.2-5B)

## 1. Data

- Dataset: the 24 `gr1_unified.*` directories of [nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim)
  - "Humanoid robot tabletop manipulation - downsampled", 1,000 trajectories per task; one directory per task, mixed with equal weight

```bash
huggingface-cli download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim --repo-type dataset --include "gr1_unified.*/*" --local-dir <root>   # GR1_ROOT
```

- Modality config: `fred/configs/data/gr1_config.py`

## 2. Text cache

For efficiency the umT5 encoder is not loaded during training: the text embeddings of every task string are
precomputed once into a cache.

```bash
.venv/bin/python scripts/precompute_text_embeds.py --data-root <root>/gr1_unified.* \
    --output-path ~/.cache/wam/gr1_wan22_umt5.pt --include-empty-prompt
```

- `--data-root`: dataset directory, repeatable; the cache holds every task string of all of them
- `--include-empty-prompt`: also embed the empty string, which caption dropout (`TEXT_DROPOUT`) samples
- `--append --output-path <cache.pt> --prompts '...'`: add prompts to an existing cache

## 3. Train

```bash
NUM_GPUS=4 GLOBAL_BATCH_SIZE=64 MAX_STEPS=100000 \
TEXT_EMBED_CACHE=~/.cache/wam/gr1_wan22_umt5.pt \
GR1_ROOT=<root containing gr1_unified.*> MODEL_OUTPUT_DIR=<ckpt root> \
bash run_scripts/train/train_fred_wan22_5b_gr1.sh
```

The shim discovers the 24 `gr1_unified.*` directories under `GR1_ROOT` (`DATASETS="..."` overrides the list).

## 4. Evaluation setup

```bash
bash fred/eval/sim/gr1_tabletop/setup_gr1_tabletop.sh     # venv: fred/eval/sim/gr1_tabletop/gr1_tabletop_uv/.venv
```

## 5. Evaluate

```bash
TEXT_EMBED_CACHE=~/.cache/wam/gr1_wan22_umt5.pt \
bash run_scripts/eval/eval_fred.sh gr1_tabletop <checkpoint dir> <output dir>
```
