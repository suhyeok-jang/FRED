#!/usr/bin/env bash
# Build the GR-1 tabletop sim venv (fred/eval/sim/gr1_tabletop/gr1_tabletop_uv/.venv) for gr1_unified/* envs, with the
# determinism patches of ./patches applied to the vendored robocasa-gr1-tabletop-tasks fork.
set -euxo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../robocasa/robocasa_venv.sh"
ROBOCASA_REPO="$SCRIPT_DIR/../../../../external_dependencies/robocasa-gr1-tabletop-tasks"
build_robocasa_venv "$SCRIPT_DIR/gr1_tabletop_uv" "$ROBOCASA_REPO"
python "$ROBOCASA_REPO/robocasa/scripts/download_tabletop_assets.py" -y
for patch in "$SCRIPT_DIR"/patches/*.patch; do   # reset settle + 24-DC seeding; check_gr1_patches.sh verifies the result
  if git -C "$ROBOCASA_REPO" apply --check "$patch" 2>/dev/null; then
    git -C "$ROBOCASA_REPO" apply "$patch"; echo "  applied:  $(basename "$patch")"
  elif git -C "$ROBOCASA_REPO" apply --reverse --check "$patch" 2>/dev/null; then
    echo "  present:  $(basename "$patch")"
  else
    echo "  FAILED:   $(basename "$patch") does not apply to $ROBOCASA_REPO (see check_gr1_patches.sh --help)" >&2; exit 1
  fi
done
bash "$SCRIPT_DIR/check_gr1_patches.sh"
smoke_env "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
