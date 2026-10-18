# OpenArm settings shared by the robot-only and robot+human shims: two cameras stacked vertically at 192x256,
# relative arm action (hand/neck absolute), per-dataset action groups, view masking.
default MODALITY_CONFIG_PATH    "$BASE_DIR/fred/configs/data/openarm_config.py"

default VIDEO_RESOLUTION        256
default VIDEO_RESOLUTION_HW 192x256
default VIDEO_CROP_ASPECT   1
default MULTIVIEW_LAYOUT        vconcat
default NUM_FRAMES_IN           13
default NUM_FRAMES_OUT          37
default FRAME_DOWNSAMPLE_FACTOR 3
default ACTION_HORIZON          24

default USE_PERCENTILES     1
default USE_RELATIVE_ACTION 1
default DATASET_ACTION_GROUPS 1           # honour <dataset>/meta/wam_action_groups.json
default VIEW_MASK_PROBS     0.5,0.25,0.25 # p(no mask), p(mask first view), p(mask second view)
default NUM_WORKERS             2             # more workers ran out of host memory while caching shards
default MIN_LR_RATIO        0.1
