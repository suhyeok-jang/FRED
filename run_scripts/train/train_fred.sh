#!/usr/bin/env bash
# fred training launcher: Wan video DiT (2.2-5B / 2.1-14B) + action DiT, flow matching + SelfDistill.
#
# Every setting is an environment variable, and the table below is the complete list. The dataset shims
# (train_fred_<backbone>_<dataset>.sh) export their values and exec this script. Each table line reads
#   value NAME  --flag  DEFAULT     passes "--flag $NAME"; DEFAULT "-" = passed only when NAME is set (fred/configs default otherwise)
#   bool  NAME  --flag  DEFAULT     NAME=1 passes --flag, NAME=0 passes --no-flag
# The table lists the settings experiments vary; everything else keeps its fred/configs default and can be passed through
# EXTRA_ARGS="--model.compile-mode None ..." (launch_train.py --help lists every flag). DRY_RUN=1 prints the command only.
set -euo pipefail
cd "${BASE_DIR:-$(pwd)}"
die()   { echo "[train_fred] $*" >&2; exit 1; }
ARGS=()
value() { local v="${!1:-$3}"; [ "$v" = "-" ] && return 0; export "$1=$v"; ARGS+=("$2" "$v"); }
bool()  { local v="${!1:-$3}" flag="$2"; export "$1=$v"
          [[ $flag == *.* ]] && local no="${flag%.*}.no-${flag##*.}" || local no="--no-${flag#--}"
          case "$v" in 1) ARGS+=("$flag") ;; 0) ARGS+=("$no") ;; *) die "$1 must be 0 or 1, got $v" ;; esac; }

# ── data ──
: "${DATASETS:?DATASETS required: LeRobot dataset dir(s), space-separated}"
: "${MODALITY_CONFIG_PATH:?MODALITY_CONFIG_PATH required (fred/configs/data/*_config.py)}"
ARGS+=(--dataset-paths $DATASETS --modality-config-path "$MODALITY_CONFIG_PATH")
[ -z "${MIX_RATIOS:-}" ] || ARGS+=(--dataset-mix-ratios $MIX_RATIOS)              # one weight per dataset (default: equal)
bool  USE_PERCENTILES          --data.use-percentiles                  0          # q01/q99 normalisation instead of min/max
bool  USE_RELATIVE_ACTION      --data.use-relative-action              0          # arm action = action - current joint state
bool  APPLY_SINCOS             --data.apply-sincos-state-encoding      0          # LIBERO: state 8 -> 16
bool  HUMAN_DATASETS           --data.human-datasets                   0          # accept meta/wam_human.json (video-only) datasets
bool  DATASET_ACTION_GROUPS    --data.dataset-action-groups            0          # accept meta/wam_action_groups.json
value BATCH_GROUP_FRACTIONS    --data.batch-group-fractions            -          # "robot:0.5,human:0.5": exact per-batch split

# ── video ──
value VIDEO_RESOLUTION         --data.video-resolution                 256
value VIDEO_RESOLUTION_HW      --data.video-resolution-hw              -          # "192x320": non-square canvas
bool  VIDEO_CROP_ASPECT        --data.video-crop-aspect                0          # aspect-preserving crop before the resize
value MULTIVIEW_LAYOUT         --data.multiview-layout                 tmosaic    # tmosaic | hconcat | vconcat
value IMAGE_RESIZE_BACKEND     --data.image-resize-backend             cv2_linear
value VIEW_MASK_PROBS          --data.view-mask-probs                  -          # "0.5,0.25,0.25": p(no mask), p(mask view 1), p(mask view 2)
value NUM_FRAMES_IN            --model.num-frames-in                   9          # history + current frames (after FRAME_DOWNSAMPLE_FACTOR)
value NUM_FRAMES_OUT           --model.num-frames-out                  25         # history + current + future frames
value FRAME_DOWNSAMPLE_FACTOR  --model.frame-downsample-factor         2

# ── augmentation ──
bool  IMAGE_AUG                --data.image-aug                        1          # train-time crop + colour jitter (eval: center crop)
value IMAGE_AUG_ROTATE_DEG     --data.image-aug-rotate-deg             5          # 0 = no rotation
value IMAGE_AUG_BRIGHTNESS     --data.image-aug-brightness             0.2        # colour jitter half-ranges (DreamZero: 0.3 / 0.4 / 0.5 / 0.08)
value IMAGE_AUG_CONTRAST       --data.image-aug-contrast               0.2
value IMAGE_AUG_SATURATION     --data.image-aug-saturation             0.2
value IMAGE_AUG_HUE            --data.image-aug-hue                    0.05
value TEXT_DROPOUT             --data.text-dropout                     0.1

# ── model ──
value BACKBONE_TYPE            --model.backbone-type                   wan22_5b   # wan22_5b (Wan2.2-TI2V-5B: tap layer 18, DiT-B) | wan21_14b (Wan2.1-I2V-14B: layer 24, DiT-L, sharded EMA)
value ACTION_HORIZON           --model.action-horizon                  -
value MAX_ACTION_DIM           --model.max-action-dim                  -
value MAX_STATE_DIM            --model.max-state-dim                   -
value TEXT_EMBED_CACHE         --model.text-embed-cache                -          # required: scripts/precompute_text_embeds.py

# ── optimisation ──
value MAX_STEPS                --training.max-steps                    100000
value WARMUP_STEPS             --training.warmup-steps                 5000
value GLOBAL_BATCH_SIZE        --training.global-batch-size            64         # clips per micro-batch over all GPUs
value GRAD_ACCUM               --training.gradient-accumulation-steps  1          # effective batch = GLOBAL_BATCH_SIZE × GRAD_ACCUM
value LR                       --training.learning-rate                1e-4       # the base learning rate
bool  LR_PER_MODULE            --training.lr-per-module-enabled        0          # 1: video DiT at LR_BACKBONE, action head at LR, rest at LR_BASE
if [ "$LR_PER_MODULE" = 1 ]; then
  value LR_BASE                  --training.lr-base                      "$LR"
  value LR_BACKBONE              --training.lr-backbone                  "$LR"
  ARGS+=(--training.lr-action-head "$LR")
fi
value MIN_LR_RATIO             --training.min-lr-ratio                 0.01       # cosine floor = ratio × lr (sim 0.01, real robot 0.1)
bool  GC                       --training.gradient-checkpointing       0          # on the video DiT

# ── compute ──
value NUM_GPUS                 --training.num-gpus                     4
value NUM_WORKERS              --training.dataloader-num-workers       4          # per rank

# ── checkpoints / logging ──
value SAVE_STEPS               --training.save-steps                   10000
value BASE_MODEL_PATH          --base-model-path                       -          # warm start: checkpoint dir or HF repo (weights + its config.yaml)
bool  ALLOW_CONFIG_CHANGES     --allow-config-changes                  0          # permit model-field differences from that config

[ -f "${TEXT_EMBED_CACHE:-}" ] || die "TEXT_EMBED_CACHE missing: ${TEXT_EMBED_CACHE:-unset}"

[ -z "${EXTRA_ARGS:-}" ] || ARGS+=($EXTRA_ARGS)

# ── run name, checkpoint dir, resume ──
RUN_GROUP="${RUN_GROUP:-wam_fred}"          # checkpoints go to MODEL_OUTPUT_DIR/RUN_GROUP/RUN_NAME
RUN_NAME="${RUN_NAME:-${BACKBONE_TYPE}_gb$((GLOBAL_BATCH_SIZE * GRAD_ACCUM))_${MAX_STEPS}}"
CKPT_DIR="${CKPT_DIR_OVERRIDE:-${MODEL_OUTPUT_DIR:-$PWD/ckpt}/$RUN_GROUP}"
EXP_DIR="$CKPT_DIR/$RUN_NAME"
ARGS+=(--training.output-dir "$CKPT_DIR" --training.experiment-name "$RUN_NAME")

RESUME="${RESUME:-auto}"      # auto: continue from the latest checkpoint if there is one | 1: one must exist | 0: start fresh
latest_ckpt=""
for c in $(ls -d "$EXP_DIR"/checkpoint-* 2>/dev/null | sort -V -r); do
  if [ -f "$c/latest" ] && [ -d "$c/$(cat "$c/latest")" ]; then latest_ckpt="$c"; break; fi
done
case "$RESUME" in
  auto) ;;
  1)    [ -n "$latest_ckpt" ] || die "RESUME=1 but no usable checkpoint-*/latest under $EXP_DIR" ;;
  0)    latest_ckpt="" ;;
  *)    die "RESUME must be 0|1|auto, got $RESUME" ;;
esac
if [ -n "$latest_ckpt" ]; then ARGS+=(--training.resume-from-checkpoint); else ARGS+=(--training.no-resume-from-checkpoint); fi

echo "[train_fred] run=$RUN_NAME  dir=$EXP_DIR  resume=${latest_ckpt:-no}${BASE_MODEL_PATH:+  warm_start=$BASE_MODEL_PATH}"
echo "[train_fred] $(printf '%q ' "${ARGS[@]}")"
[ "${DRY_RUN:-0}" = 1 ] && exit 0
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
exec uv run --no-sync torchrun --nproc_per_node="$NUM_GPUS" --master_port="${MASTER_PORT:-$(shuf -i 29510-39990 -n 1)}" \
  fred/experiment/launch_train.py "${ARGS[@]}"
