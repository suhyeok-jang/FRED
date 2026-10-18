#!/usr/bin/env bash
# LIBERO (libero_delta) on Wan2.2-5B: the DiT4DiT LIBERO recipe (224 hconcat, torch-bilinear resize, no rotation,
# sin/cos state, horizon 16). DATASETS must point at the libero_delta LeRobot dir.
source "$(dirname "${BASH_SOURCE[0]}")/shim.sh"

default RUN_GROUP                     libero
: "${DATASETS:?DATASETS (libero_delta LeRobot dir) required}"
default MODALITY_CONFIG_PATH    "$BASE_DIR/fred/configs/data/libero_config.py"

default VIDEO_RESOLUTION        224
default MULTIVIEW_LAYOUT        hconcat
default IMAGE_RESIZE_BACKEND    torch_bilinear
default NUM_FRAMES_IN           9
default NUM_FRAMES_OUT          25
default FRAME_DOWNSAMPLE_FACTOR 2
default IMAGE_AUG_ROTATE_DEG 0

default ACTION_HORIZON          16
default MAX_ACTION_DIM          8
default MAX_STATE_DIM           16
default APPLY_SINCOS            1             # state 8 -> 16

default MIN_LR_RATIO        0.01

launch
