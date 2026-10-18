#!/usr/bin/env bash
# OpenArm, robot datasets only (video + action). DATASETS="<robot dirs>" MIX_RATIOS="<weights, same order>".
source "$(dirname "${BASH_SOURCE[0]}")/shim.sh"

default RUN_GROUP                     openarm_robot
: "${DATASETS:?DATASETS (robot LeRobot dirs) required}"
source "$BASE_DIR/run_scripts/train/openarm_common.sh"

default HUMAN_DATASETS      0
default GLOBAL_BATCH_SIZE       32

launch
