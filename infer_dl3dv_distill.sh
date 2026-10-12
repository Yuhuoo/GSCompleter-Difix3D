#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

to_abs() {
  if [[ "$1" == /* ]]; then
    realpath -m "$1"
  else
    realpath -m "$ROOT_DIR/$1"
  fi
}

DATA_ROOT="$(to_abs "${DATA_ROOT:-datasets/dl3dv}")"
INDEX_PATH="$(to_abs "${INDEX_PATH:-assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n10.json}")"
CHECKPOINT="$(to_abs "${CHECKPOINT:-pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth}")"
RUN_ROOT="$(to_abs "${RUN_ROOT:-outputs/dl3dv_difix_distill}")"
DIFIX_REPO="$(to_abs "${DIFIX_REPO:-../Difix3D}")"
DIFIX_PYTHON="${DIFIX_PYTHON:-python}"
DIFIX_MODEL="${DIFIX_MODEL:-$DIFIX_REPO/checkpoints/nvidia-difix}"
if [[ "$DIFIX_MODEL" == ./* || "$DIFIX_MODEL" == ../* ]]; then
  DIFIX_MODEL="$(to_abs "$DIFIX_MODEL")"
fi
GPU="${GPU:-0}"
STEPS="${STEPS:-3000}"
CONTEXT_LOSS_WEIGHT="${CONTEXT_LOSS_WEIGHT:-0.1}"
DENSIFY_INTERVAL="${DENSIFY_INTERVAL:-250}"
DENSIFY_GRAD_THRESHOLD="${DENSIFY_GRAD_THRESHOLD:-2e-4}"
MAX_NEW_GAUSSIANS="${MAX_NEW_GAUSSIANS:-4096}"
MAX_GAUSSIANS="${MAX_GAUSSIANS:-300000}"
PRUNE_OPACITY="${PRUNE_OPACITY:-0.005}"
MIN_GAUSSIANS="${MIN_GAUSSIANS:-1024}"
SCENE="${SCENE:-}"
OVERWRITE="${OVERWRITE:-0}"
EXPECTED_SCENES="${EXPECTED_SCENES:-140}"

[[ -d "$DATA_ROOT" ]] || { echo "Missing DATA_ROOT: $DATA_ROOT" >&2; exit 1; }
[[ -f "$INDEX_PATH" ]] || { echo "Missing INDEX_PATH: $INDEX_PATH" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing CHECKPOINT: $CHECKPOINT" >&2; exit 1; }
[[ -f "$ROOT_DIR/src/run_difix.py" ]] || { echo "Missing local DiFix adapter: $ROOT_DIR/src/run_difix.py" >&2; exit 1; }
[[ -d "$DIFIX_MODEL" ]] || { echo "Missing DIFIX_MODEL directory: $DIFIX_MODEL" >&2; exit 1; }
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "STEPS must be a positive integer" >&2; exit 1; }
[[ "$CONTEXT_LOSS_WEIGHT" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "CONTEXT_LOSS_WEIGHT must be non-negative" >&2; exit 1; }
[[ "$DENSIFY_INTERVAL" =~ ^[1-9][0-9]*$ ]] || { echo "DENSIFY_INTERVAL must be positive" >&2; exit 1; }

common_args=(
  +experiment=dl3dv
  mode=test
  "dataset.roots=[$DATA_ROOT]"
  dataset/view_sampler=evaluation
  dataset.view_sampler.num_context_views=2
  "dataset.view_sampler.index_path=$INDEX_PATH"
  dataset.image_shape=[256,448]
  model.encoder.num_scales=2
  model.encoder.upsample_factor=4
  model.encoder.lowest_feature_resolution=8
  model.encoder.monodepth_vit_type=vitb
  "checkpointing.pretrained_model=$CHECKPOINT"
  hydra/job_logging=stdout
  hydra.output_subdir=null
)

echo "[1/4] DepthSplat initialization and target-view render"
CUDA_VISIBLE_DEVICES="$GPU" DIFIX_PHASE=prepare DIFIX_ROOT="$RUN_ROOT" \
  DIFIX_SCENE="$SCENE" DIFIX_OVERWRITE="$OVERWRITE" \
  python -m src.difix_distill "${common_args[@]}"

echo "[2/4] DiFix3D single-view target-view repair"
difix_args=(
  --input-root "$RUN_ROOT/target_input"
  --output-root "$RUN_ROOT/target_fixed"
  --model-path "$DIFIX_MODEL"
  --prompt "remove degradation"
  --timestep 199
  --difix-repo "$DIFIX_REPO"
)
[[ -z "$SCENE" ]] || difix_args+=(--scene "$SCENE")
[[ "$OVERWRITE" != "1" ]] || difix_args+=(--overwrite)
CUDA_VISIBLE_DEVICES="$GPU" "$DIFIX_PYTHON" "$ROOT_DIR/src/run_difix.py" "${difix_args[@]}"

echo "[3/4] One-shot pseudo-GT distillation with context-Gaussian optimization ($STEPS steps/scene)"
CUDA_VISIBLE_DEVICES="$GPU" DIFIX_PHASE=distill DIFIX_ROOT="$RUN_ROOT" \
  DIFIX_SCENE="$SCENE" DIFIX_OVERWRITE="$OVERWRITE" DIFIX_STEPS="$STEPS" \
  DIFIX_CONTEXT_LOSS_WEIGHT="$CONTEXT_LOSS_WEIGHT" \
  DIFIX_DENSIFY_INTERVAL="$DENSIFY_INTERVAL" \
  DIFIX_DENSIFY_GRAD_THRESHOLD="$DENSIFY_GRAD_THRESHOLD" \
  DIFIX_MAX_NEW_GAUSSIANS="$MAX_NEW_GAUSSIANS" \
  DIFIX_MAX_GAUSSIANS="$MAX_GAUSSIANS" \
  DIFIX_PRUNE_OPACITY="$PRUNE_OPACITY" \
  DIFIX_MIN_GAUSSIANS="$MIN_GAUSSIANS" \
  python -m src.difix_distill "${common_args[@]}"

echo "[4/4] Evaluating held-out target renders"
metric_args=(
  --render-root "$RUN_ROOT/depthsplat"
  --refined-root "$RUN_ROOT/distilled"
  --output "$RUN_ROOT/metrics.json"
)
if [[ -n "$SCENE" ]]; then
  metric_args+=(--scene "$SCENE")
else
  metric_args+=(--expected-scenes "$EXPECTED_SCENES")
fi
CUDA_VISIBLE_DEVICES="$GPU" python -m src.eval_difix3d "${metric_args[@]}"
echo "Done: $RUN_ROOT/metrics.json"
