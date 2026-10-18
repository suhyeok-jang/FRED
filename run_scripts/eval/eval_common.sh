# Sourced by eval_fred.sh; eval_fred_task.sh spells the same steps out for one task.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${PORT:-5555}"
N_ENVS="${N_ENVS:-1}"

benchmark_settings() {   # sets SIM (simulator venv python) and N_EPISODES for benchmark $1
  case "$1" in
    robocasa_kitchen) SIM_VENV=robocasa/robocasa_kitchen_uv; N_EPISODES="${N_EPISODES:-50}" ;;
    gr1_tabletop)     SIM_VENV=gr1_tabletop/gr1_tabletop_uv; N_EPISODES="${N_EPISODES:-50}" ;;
    libero)           SIM_VENV=libero/libero_uv;             N_EPISODES="${N_EPISODES:-10}" ;;
    *) echo "unknown benchmark: $1 (robocasa_kitchen | gr1_tabletop | libero)" >&2; exit 1 ;;
  esac
  SIM="$ROOT/fred/eval/sim/$SIM_VENV/.venv/bin/python"
  [ -x "$SIM" ] || { echo "simulator venv missing: $SIM (run fred/eval/sim/*/setup_*.sh)" >&2; exit 1; }
}

max_episode_steps() {    # LIBERO caps per suite; 720 for the kitchen and GR-1
  case "$1" in
    libero_spatial/*) echo 220 ;;  libero_object/*) echo 280 ;;  libero_goal/*) echo 300 ;;  libero_10/*) echo 520 ;;
    libero_*/*)       echo 400 ;;  *)               echo 720 ;;
  esac
}

env_name() {             # LIBERO tasks are listed as <suite>/<task> and run in the one libero_sim env
  case "$1" in libero_*/*) echo "libero_sim/${1#*/}" ;; *) echo "$1" ;; esac
}

start_policy_server() {  # $1 = checkpoint dir, $2 = output dir (server.log); killed when the caller exits
  "$ROOT/.venv/bin/python" "$ROOT/fred/eval/serve_policy.py" --model-path "$1" --port "$PORT" \
    ${TEXT_EMBED_CACHE:+--text-embed-cache "$TEXT_EMBED_CACHE"} > "$2/server.log" 2>&1 &
  SERVER_PID=$!
  trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
  until "$ROOT/.venv/bin/python" -c "import socket, sys; sys.exit(socket.socket().connect_ex(('127.0.0.1', $PORT)))" 2>/dev/null; do
    kill -0 $SERVER_PID 2>/dev/null || { echo "server died, see $2/server.log" >&2; exit 1; }
    sleep 2
  done
}

rollout_once() {         # $1 = task, $2 = task dir; one rollout_policy.py pass (resumes from the mp4s in $2)
  "$SIM" "$ROOT/fred/eval/rollout_policy.py" --env-name "$(env_name "$1")" --policy-client-port "$PORT" --output-dir "$2" \
    --n-episodes "$N_EPISODES" --n-envs "$N_ENVS" --n-action-steps 16 --max-episode-steps "$(max_episode_steps "$1")"
}

RENDER_REPAIR_ROUNDS=3   # re-runs of render-corrupted episodes (fred/eval/render_check.py); always on for the kitchen

rollout_task() {         # $1 = task, $2 = output dir; writes $2/<task>/ (mp4s, simulation_results.csv, rollout.log)
  local task="$1" dir="$2/${1#*/}" round
  mkdir -p "$dir"
  echo ">>> $task"
  rollout_once "$task" "$dir" > "$dir/rollout.log" 2>&1 || { echo "!!! $task failed, see $dir/rollout.log"; return 0; }
  # Renderer-corrupted episodes are moved aside and their ids re-run with the same seeds; whatever is still
  # flagged after the last round keeps its result and is named in the report. Kitchen only: the detector is
  # calibrated on the kitchen's three-view recording.
  [ "${BENCHMARK:-}" = robocasa_kitchen ] || return 0
  for round in $(seq 1 $RENDER_REPAIR_ROUNDS); do
    [ "$("$SIM" "$ROOT/fred/eval/render_check.py" "$dir" --quarantine "$round" 2>>"$dir/rollout.log")" -eq 0 ] && break
    rollout_once "$task" "$dir" >> "$dir/rollout.log" 2>&1 || { echo "!!! $task re-run failed, see $dir/rollout.log"; return 0; }
  done
  "$SIM" "$ROOT/fred/eval/render_check.py" "$dir" > /dev/null
}
