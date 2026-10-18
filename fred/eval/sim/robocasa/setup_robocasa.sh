#!/usr/bin/env bash
# Build the RoboCasa kitchen sim venv (fred/eval/sim/robocasa/robocasa_kitchen_uv/.venv) for robocasa_panda_omron/* envs.
set -euxo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/robocasa_venv.sh"
ROBOCASA_REPO="$SCRIPT_DIR/../../../../external_dependencies/robocasa"
build_robocasa_venv "$SCRIPT_DIR/robocasa_kitchen_uv" "$ROBOCASA_REPO"
# the upstream script prompts on stdin; it does not consume a "-y" flag in this fork revision
printf 'y\n' | python "$ROBOCASA_REPO/robocasa/scripts/download_kitchen_assets.py"
smoke_env "robocasa_panda_omron/OpenSingleDoor_PandaOmron_Env"
