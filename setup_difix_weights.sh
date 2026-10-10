#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${DIFIX_ENV_PREFIX:-$ROOT_DIR/.conda/envs/gscompleter-difix3d}"
WEIGHTS_DIR="${DIFIX_WEIGHTS_DIR:-$ROOT_DIR/../Difix3D/checkpoints}"
MODEL_DIR="$WEIGHTS_DIR/nvidia-difix"

[[ -x "$ENV_PREFIX/bin/python" ]] || { echo "Missing environment: $ENV_PREFIX (run setup_difix_env.sh first)" >&2; exit 1; }
mkdir -p "$WEIGHTS_DIR"

export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

"$ENV_PREFIX/bin/python" - "$MODEL_DIR" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
target = sys.argv[1]
required = [
    "model_index.json",
    "text_encoder/model.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]
if all(os.path.isfile(os.path.join(target, item)) for item in required):
    print(f"Already present: {target}")
else:
    snapshot_download("nvidia/difix", local_dir=target)
    print(f"Downloaded: {target}")
PY

echo "Difix weights ready: $MODEL_DIR"
