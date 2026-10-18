#!/usr/bin/env bash
# DROID on Wan2.2-5B. Dataset settings: droid_common.sh.
source "$(dirname "${BASH_SOURCE[0]}")/shim.sh"

default RUN_GROUP                     droid
source "$BASE_DIR/run_scripts/train/droid_common.sh"

launch
