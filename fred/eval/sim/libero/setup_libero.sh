#!/usr/bin/env bash
# Build the LIBERO sim venv (fred/eval/sim/libero/libero_uv/.venv) for libero_sim/* envs.
set -euxo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$SCRIPT_DIR/../../../.."
LIBERO_REPO="$PROJECT_REPO/external_dependencies/LIBERO"
LIBERO_UV_ENV="$SCRIPT_DIR/libero_uv"

rm -rf "$LIBERO_UV_ENV"
mkdir -p "$LIBERO_UV_ENV"
uv venv "$LIBERO_UV_ENV/.venv" --python 3.10
source "$LIBERO_UV_ENV/.venv/bin/activate"
uv pip install --requirements "$LIBERO_REPO/requirements.txt"
uv pip install -e "$LIBERO_REPO" --config-settings editable_mode=compat
uv pip install torch==2.5.1 torchvision==0.20.1 pydantic av tyro dm_tree einops==0.8.1 albumentations==1.4.18 zmq
uv pip install transformers==4.51.3 msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1
uv pip install numpy==1.26.4

# this repo, editable, without re-resolving deps
uv pip install --editable "$PROJECT_REPO" --no-deps

# LIBERO writes its config (asset paths) to ~/.libero on first import; start from a clean one
rm -rf "$HOME/.libero"
python - <<'PY'
from fred.eval.sim.libero.libero_env import register_libero_envs
register_libero_envs()
import gymnasium as gym
env = gym.make("libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate")
env.reset()
env.close()
print("Env OK:", type(env))
PY
