#!/usr/bin/env bash
# bash run_scripts/eval/eval_fred_task.sh <robocasa_kitchen|gr1_tabletop|libero> <checkpoint dir> <output dir> <task>
#
# One task, with the two processes of an evaluation spelled out:
#   1. the policy server (training venv) loads the checkpoint on the GPU and answers action requests on tcp://*:PORT
#   2. the simulator client (simulator venv) builds the env and asks the server for an action chunk every 16 steps
# eval_fred.sh runs the same two processes for every task of a benchmark. Task names: run_scripts/eval/tasks/<benchmark>.txt.
# Optional env: TEXT_EMBED_CACHE (umT5 cache, wan22_5b checkpoints), N_EPISODES, N_ENVS, PORT.
set -euo pipefail
BENCHMARK="${1:?robocasa_kitchen | gr1_tabletop | libero}"; CKPT="${2:?checkpoint dir}"; OUT="${3:?output dir}"; TASK="${4:?task}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${PORT:-5555}"
N_ENVS="${N_ENVS:-1}"

case "$BENCHMARK" in
  robocasa_kitchen) SIM="$ROOT/fred/eval/sim/robocasa/robocasa_kitchen_uv/.venv/bin/python"; N_EPISODES="${N_EPISODES:-50}"; MAX_EPISODE_STEPS=720; ENV="$TASK" ;;
  gr1_tabletop)     SIM="$ROOT/fred/eval/sim/gr1_tabletop/gr1_tabletop_uv/.venv/bin/python"; N_EPISODES="${N_EPISODES:-50}"; MAX_EPISODE_STEPS=720; ENV="$TASK" ;;
  libero)           SIM="$ROOT/fred/eval/sim/libero/libero_uv/.venv/bin/python";             N_EPISODES="${N_EPISODES:-10}"; ENV="libero_sim/${TASK#*/}"
                    case "$TASK" in libero_spatial/*) MAX_EPISODE_STEPS=220 ;; libero_object/*) MAX_EPISODE_STEPS=280 ;; libero_goal/*) MAX_EPISODE_STEPS=300 ;; libero_10/*) MAX_EPISODE_STEPS=520 ;; *) MAX_EPISODE_STEPS=400 ;; esac ;;
  *) echo "unknown benchmark: $BENCHMARK" >&2; exit 1 ;;
esac
TASK_DIR="$OUT/${TASK#*/}"
mkdir -p "$TASK_DIR"

# 1. policy server
"$ROOT/.venv/bin/python" "$ROOT/fred/eval/serve_policy.py" --model-path "$CKPT" --port "$PORT" \
  ${TEXT_EMBED_CACHE:+--text-embed-cache "$TEXT_EMBED_CACHE"} > "$TASK_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
until "$ROOT/.venv/bin/python" -c "import socket, sys; sys.exit(socket.socket().connect_ex(('127.0.0.1', $PORT)))" 2>/dev/null; do
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died, see $TASK_DIR/server.log" >&2; exit 1; }
  sleep 2
done

# 2. simulator client: <task dir>/ gets one mp4 per episode, simulation_results.csv, rollout.log (and server.log above)
rollout() {
  "$SIM" "$ROOT/fred/eval/rollout_policy.py" --env-name "$ENV" --policy-client-port "$PORT" --output-dir "$TASK_DIR" \
    --n-episodes "$N_EPISODES" --n-envs "$N_ENVS" --n-action-steps 16 --max-episode-steps "$MAX_EPISODE_STEPS" \
    2>&1 | tee -a "$TASK_DIR/rollout.log"
}
: > "$TASK_DIR/rollout.log"
rollout

# 2b. render check (always on for the kitchen, whose three-view recording the detector is calibrated on): episodes
#     whose camera images the renderer corrupted are moved to <task dir>/render_quarantine/round<k>/ and re-run with
#     the same seeds, up to 3 rounds (fred/eval/render_check.py)
if [ "$BENCHMARK" = robocasa_kitchen ]; then
  for round in 1 2 3; do
    [ "$("$SIM" "$ROOT/fred/eval/render_check.py" "$TASK_DIR" --quarantine "$round")" -eq 0 ] && break
    rollout
  done
  "$SIM" "$ROOT/fred/eval/render_check.py" "$TASK_DIR" > /dev/null
fi

# 3. success rate
"$ROOT/.venv/bin/python" "$ROOT/run_scripts/eval/success_rates.py" "$OUT"
