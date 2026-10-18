#!/usr/bin/env bash
# GR-1 tabletop (the 24 gr1_unified.* LeRobot dirs under GR1_ROOT, equal mix) on Wan2.2-5B.
source "$(dirname "${BASH_SOURCE[0]}")/shim.sh"

default RUN_GROUP                     gr1_tabletop
[ -n "${DATASETS:-}" ] || : "${GR1_ROOT:?GR1_ROOT (root of the 24 gr1_unified.* LeRobot dirs) required unless DATASETS is set}"
default MODALITY_CONFIG_PATH    "$BASE_DIR/fred/configs/data/gr1_config.py"

if [ -z "${DATASETS:-}" ]; then   # one dataset dir per task
  dirs=()
  for d in "$GR1_ROOT"/gr1_unified.*; do [ -s "$d/meta/tasks.jsonl" ] && dirs+=("$d"); done
  [ "${#dirs[@]}" -eq 24 ] || { echo "found ${#dirs[@]} gr1_unified.* dirs under $GR1_ROOT, expected 24" >&2; exit 1; }
  export DATASETS="${dirs[*]}"
fi

default VIDEO_RESOLUTION        256
default MULTIVIEW_LAYOUT        hconcat       # one camera
default NUM_FRAMES_IN           9
default NUM_FRAMES_OUT          25
default FRAME_DOWNSAMPLE_FACTOR 2

default USE_PERCENTILES     1             # q01/q99 normalisation
default MIN_LR_RATIO        0.01

launch
