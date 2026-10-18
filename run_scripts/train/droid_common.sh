# DROID (DreamZero LeRobot export) dataset settings, shared by the 5B and 14B DROID shims.
: "${DATASETS:?DATASETS (DreamZero-DROID-Data LeRobot dir) required}"
default MODALITY_CONFIG_PATH    "$BASE_DIR/fred/configs/data/droid_config.py"

default VIDEO_RESOLUTION        320
default VIDEO_RESOLUTION_HW 192x320       # one 192x320 canvas (aspect-preserving crop)
default MULTIVIEW_LAYOUT        tmosaic
default NUM_FRAMES_IN           13
default NUM_FRAMES_OUT          37
default FRAME_DOWNSAMPLE_FACTOR 3
default ACTION_HORIZON          24

default USE_PERCENTILES     1
default USE_RELATIVE_ACTION 1             # arm action = action - current joint state
export EXTRA_ARGS="${EXTRA_ARGS:-}
  --data.relative-stats-file relative_stats_dreamzero.json"
default MIN_LR_RATIO        0.1           # real robot
