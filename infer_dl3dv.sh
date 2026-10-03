#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# The exact 2-context/1-target split used by the revision experiments.
DATA_ROOT="${DATA_ROOT:-datasets/dl3dv}"
INDEX_PATH="${INDEX_PATH:-assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n10.json}"
CHECKPOINT="${CHECKPOINT:-pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth}"
GPU="${GPU:-0}"

to_abs() {
  if [[ "$1" = /* ]]; then
    realpath -m "$1"
  else
    realpath -m "$ROOT_DIR/$1"
  fi
}

DATA_ROOT="$(to_abs "$DATA_ROOT")"
INDEX_PATH="$(to_abs "$INDEX_PATH")"
CHECKPOINT="$(to_abs "$CHECKPOINT")"

RUN_ROOT="${RUN_ROOT:-outputs/dl3dv_difix}"
DEPTHSPLAT_OUTPUT="${DEPTHSPLAT_OUTPUT:-$RUN_ROOT/depthsplat}"
DIFIX_INPUT_ROOT="${DIFIX_INPUT_ROOT:-$RUN_ROOT/difix_input}"
DIFIX_OUTPUT_ROOT="${DIFIX_OUTPUT_ROOT:-$RUN_ROOT/difix}"
METRICS_PATH="${METRICS_PATH:-$RUN_ROOT/metrics.json}"

DIFIX_REPO="${DIFIX_REPO:-../Difix3D}"
DIFIX_REPO="$(cd "$DIFIX_REPO" && pwd)"
DIFIX_PYTHON="${DIFIX_PYTHON:-python}"
DIFIX_MODEL="${DIFIX_MODEL:-$DIFIX_REPO/checkpoints/model.pkl}"
RUN_ROOT="$(to_abs "$RUN_ROOT")"
DEPTHSPLAT_OUTPUT="$(to_abs "$DEPTHSPLAT_OUTPUT")"
DIFIX_INPUT_ROOT="$(to_abs "$DIFIX_INPUT_ROOT")"
DIFIX_OUTPUT_ROOT="$(to_abs "$DIFIX_OUTPUT_ROOT")"
METRICS_PATH="$(to_abs "$METRICS_PATH")"
if [[ "$DIFIX_MODEL" != /* ]]; then
  DIFIX_MODEL="$(to_abs "$DIFIX_MODEL")"
fi
SCENE="${SCENE:-}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_DEPTHSPLAT="${SKIP_DEPTHSPLAT:-0}"

[[ -d "$DATA_ROOT" ]] || { echo "Missing DATA_ROOT: $DATA_ROOT" >&2; exit 1; }
[[ -f "$INDEX_PATH" ]] || { echo "Missing INDEX_PATH: $INDEX_PATH" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing CHECKPOINT: $CHECKPOINT" >&2; exit 1; }
[[ -f "$DIFIX_MODEL" ]] || { echo "Missing DIFIX_MODEL: $DIFIX_MODEL" >&2; exit 1; }

if [[ "$OVERWRITE" == "1" ]]; then
  if [[ "$SKIP_DEPTHSPLAT" != "1" ]]; then
    rm -rf "$DEPTHSPLAT_OUTPUT"
  fi
  rm -rf "$DIFIX_INPUT_ROOT" "$DIFIX_OUTPUT_ROOT"
fi

if [[ "$SKIP_DEPTHSPLAT" != "1" ]]; then
  echo "[1/4] Running DepthSplat: 2 context views -> 1 target view"
  CUDA_VISIBLE_DEVICES="$GPU" python -m src.main \
    +experiment=dl3dv \
    mode=test \
    dataset.roots="[$DATA_ROOT]" \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=2 \
    dataset.view_sampler.index_path="$INDEX_PATH" \
    model.encoder.num_scales=2 \
    model.encoder.upsample_factor=4 \
    model.encoder.lowest_feature_resolution=8 \
    model.encoder.monodepth_vit_type=vitb \
    checkpointing.pretrained_model="$CHECKPOINT" \
    output_dir="$DEPTHSPLAT_OUTPUT" \
    test.compute_scores=true \
    test.save_image=true \
    test.save_gt_image=true \
    hydra/job_logging=stdout \
    hydra.output_subdir=null
else
  echo "[1/4] Skipping DepthSplat; using existing output: $DEPTHSPLAT_OUTPUT"
fi

echo "[2/4] Preparing target renders for DiFix3D+"
prepare_args=(
  --render-root "$DEPTHSPLAT_OUTPUT"
  --input-root "$DIFIX_INPUT_ROOT"
)
if [[ -n "$SCENE" ]]; then
  prepare_args+=(--scene "$SCENE")
fi
python -m src.prepare_difix_inputs "${prepare_args[@]}"

echo "[3/4] Applying DiFix3D+ to DepthSplat target renders"
difix_args=(
  --input-root "$DIFIX_INPUT_ROOT"
  --output-root "$DIFIX_OUTPUT_ROOT"
  --model-path "$DIFIX_MODEL"
  --prompt "remove degradation"
  --timestep 199
)
if [[ "$OVERWRITE" == "1" ]]; then
  difix_args+=(--overwrite)
fi
if [[ -n "$SCENE" ]]; then
  difix_args+=(--scene "$SCENE")
fi
(
  cd "$DIFIX_REPO"
  PYTHONPATH="$DIFIX_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
    CUDA_VISIBLE_DEVICES="$GPU" "$DIFIX_PYTHON" infer_dl3dv.py "${difix_args[@]}"
)

echo "[4/4] Computing target-view metrics"
metric_args=(
  --render-root "$DEPTHSPLAT_OUTPUT"
  --difix-root "$DIFIX_OUTPUT_ROOT"
  --output "$METRICS_PATH"
)
if [[ -n "$SCENE" ]]; then
  metric_args+=(--scene "$SCENE")
fi
python -m src.eval_difix3d "${metric_args[@]}"

echo "Done. Metrics: $METRICS_PATH"
