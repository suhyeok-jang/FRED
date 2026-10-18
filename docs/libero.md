# LIBERO (Wan2.2-5B)

## 1. Data

- Dataset: [kimtaey/libero_gr00t_delta](https://huggingface.co/datasets/kimtaey/libero_gr00t_delta)
  - A LeRobot export of [openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds)

```bash
huggingface-cli download kimtaey/libero_gr00t_delta --repo-type dataset --local-dir <dataset dir>   # DATASETS
```

- Modality config: `fred/configs/data/libero_config.py`

## 2. Text cache

For efficiency the umT5 encoder is not loaded during training: the text embeddings of every task string are
precomputed once into a cache.

```bash
.venv/bin/python scripts/precompute_text_embeds.py --data-root <libero_delta dir> \
    --output-path ~/.cache/wam/libero_wan22_umt5.pt --include-empty-prompt
```

- `--data-root`: dataset directory, repeatable; the cache holds every task string of all of them
- `--include-empty-prompt`: also embed the empty string, which caption dropout (`TEXT_DROPOUT`) samples
- `--append --output-path <cache.pt> --prompts '...'`: add prompts to an existing cache

## 3. Train

```bash
NUM_GPUS=4 GLOBAL_BATCH_SIZE=64 MAX_STEPS=100000 \
TEXT_EMBED_CACHE=~/.cache/wam/libero_wan22_umt5.pt \
DATASETS=<libero_delta dir> MODEL_OUTPUT_DIR=<ckpt root> \
bash run_scripts/train/train_fred_wan22_5b_libero.sh
```

## 4. Evaluation setup

```bash
bash fred/eval/sim/libero/setup_libero.sh                   # venv: fred/eval/sim/libero/libero_uv/.venv
```

## 5. Evaluate

```bash
TEXT_EMBED_CACHE=~/.cache/wam/libero_wan22_umt5.pt \
bash run_scripts/eval/eval_fred.sh libero <checkpoint dir> <output dir>
```

