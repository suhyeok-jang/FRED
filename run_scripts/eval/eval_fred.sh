#!/usr/bin/env bash
# bash run_scripts/eval/eval_fred.sh <robocasa_kitchen|gr1_tabletop|libero> <checkpoint dir> <output dir>
#
# Starts the policy server on the checkpoint, rolls out every task of run_scripts/eval/tasks/<benchmark>.txt and
# prints the success rate per task and the mean (also <output dir>/results.txt). Re-running on the same output dir
# resumes. Optional env: TEXT_EMBED_CACHE (umT5 cache, wan22_5b checkpoints), TASKS="<task> ..." (subset),
# N_EPISODES, N_ENVS, PORT.
source "$(dirname "${BASH_SOURCE[0]}")/eval_common.sh"
BENCHMARK="${1:?robocasa_kitchen | gr1_tabletop | libero}"; CKPT="${2:?checkpoint dir}"; OUT="${3:?output dir}"
benchmark_settings "$BENCHMARK"

if [ -n "${TASKS:-}" ]; then read -ra TASK_LIST <<< "$TASKS"
else mapfile -t TASK_LIST < <(grep -v '^#' "$ROOT/run_scripts/eval/tasks/$BENCHMARK.txt" | grep .); fi

mkdir -p "$OUT"
start_policy_server "$CKPT" "$OUT"
for task in "${TASK_LIST[@]}"; do rollout_task "$task" "$OUT"; done
"$ROOT/.venv/bin/python" "$ROOT/run_scripts/eval/success_rates.py" "$OUT" | tee "$OUT/results.txt"
