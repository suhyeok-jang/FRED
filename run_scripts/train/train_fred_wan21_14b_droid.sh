#!/usr/bin/env bash
# DROID on Wan2.1-I2V-14B (tap layer 24, DiT-L, sharded EMA follow the backbone): per-module learning rates, AdamW
# (0.95, 0.999) with weight decay 1e-5, gradient checkpointing. Reference plate 16 GPUs x 8 clips. Dataset: droid_common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/shim.sh"

default RUN_GROUP               droid_14b
source "$BASE_DIR/run_scripts/train/droid_common.sh"

default BACKBONE_TYPE           wan21_14b
default GC                      1             # gradient checkpointing

default LR_PER_MODULE           1
default LR_BACKBONE             1e-5          # video DiT
default LR_BASE                 1e-4
default LR          1e-4
export EXTRA_ARGS="${EXTRA_ARGS:-}
  --training.weight-decay 1e-5
  --training.adam-beta1   0.95
  --training.adam-beta2   0.999"

default MAX_STEPS               200000
default SAVE_STEPS              5000

launch
