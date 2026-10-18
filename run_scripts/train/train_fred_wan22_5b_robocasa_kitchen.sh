#!/usr/bin/env bash
# RoboCasa kitchen (robocasa_mg_gr00t_300) on Wan2.2-5B. Override any line with an environment variable.
source "$(dirname "${BASH_SOURCE[0]}")/shim.sh"

default RUN_GROUP                     robocasa_kitchen
: "${DATASETS:?DATASETS (robocasa_mg_gr00t_300 LeRobot dir) required}"
default MODALITY_CONFIG_PATH    "$BASE_DIR/fred/configs/data/robocasa_config.py"

default VIDEO_RESOLUTION        256
default MULTIVIEW_LAYOUT        tmosaic       # wrist view on top, left | right half-size below
default NUM_FRAMES_IN           9
default NUM_FRAMES_OUT          25
default FRAME_DOWNSAMPLE_FACTOR 2

default MIN_LR_RATIO        0.01

launch
