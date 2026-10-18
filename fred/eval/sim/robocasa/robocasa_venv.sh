#!/usr/bin/env bash
# Shared by setup_robocasa.sh (kitchen) and ../gr1_tabletop/setup_gr1_tabletop.sh: build one RoboCasa-fork venv.
#   build_robocasa_venv <venv dir> <fork repo dir>
build_robocasa_venv() {
  local UV_ENV="$1" ROBOCASA_REPO="$2" PROJECT_REPO
  PROJECT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
  rm -rf "$UV_ENV"; mkdir -p "$UV_ENV"
  uv venv "$UV_ENV/.venv" --python 3.10
  source "$UV_ENV/.venv/bin/activate"
  uv pip install setuptools wheel
  uv pip install torch==2.5.1 torchvision==0.20.1
  if [[ "$(uname -s)" == "Linux" && "${INSTALL_FLASH_ATTN:-1}" == "1" ]]; then
    uv pip install --no-build-isolation flash-attn==2.7.4.post1 || echo "flash-attn install skipped/failed; continuing"
  fi
  uv pip install robosuite==1.5.1
  uv pip install -e "$ROBOCASA_REPO" --config-settings editable_mode=compat
  uv pip install gymnasium==0.29.1 pydantic av==15.0.0 zmq transformers==4.51.3 msgpack==1.1.0 msgpack-numpy==0.4.8 idna
  uv pip install --editable "$PROJECT_REPO" --no-deps
}
smoke_env() {   # sanity import + env construction
  SMOKE_ENV="$1" python - <<'EOF'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import gymnasium as gym, robosuite
import robocasa.utils.gym_utils.gymnasium_groot
print("Imports OK:", robosuite.__version__)
env = gym.make(os.environ["SMOKE_ENV"], enable_render=True)
print("Env OK:", type(env))
EOF
}
