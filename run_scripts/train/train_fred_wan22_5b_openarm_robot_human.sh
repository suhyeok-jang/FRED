#!/usr/bin/env bash
# OpenArm, robot + human co-training: robot datasets train video + action, human datasets (meta/wam_human.json)
# train video only, every batch split 1:1. DATASETS="<robot dirs> <human dirs>" MIX_RATIOS="<weights, same order>".
source "$(dirname "${BASH_SOURCE[0]}")/shim.sh"

default RUN_GROUP                     openarm_robot_human
: "${DATASETS:?DATASETS (robot dirs then human dirs) required}"
source "$BASE_DIR/run_scripts/train/openarm_common.sh"

default HUMAN_DATASETS      1
default BATCH_GROUP_FRACTIONS           robot:0.5,human:0.5
default GLOBAL_BATCH_SIZE       64

launch
