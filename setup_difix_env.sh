#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${DIFIX_ENV_PREFIX:-$ROOT_DIR/.conda/envs/gscompleter-difix3d}"
CONDA_EXE="${CONDA_EXE:-conda}"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_EXE" create -y -p "$ENV_PREFIX" python=3.10 pip
fi

"$ENV_PREFIX/bin/python" -m pip install --upgrade pip
"$ENV_PREFIX/bin/python" -m pip install \
  torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
"$ENV_PREFIX/bin/python" -m pip install ninja
"$ENV_PREFIX/bin/python" -m pip install -r <(grep -v '^git+' "$ROOT_DIR/requirements.txt")
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" \
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}" \
"$ENV_PREFIX/bin/python" -m pip install --no-build-isolation \
  git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
"$ENV_PREFIX/bin/python" -m pip install -r "$ROOT_DIR/../Difix3D/requirements.txt"
"$ENV_PREFIX/bin/python" -m pip install \
  'numpy==1.24.4' 'diffusers==0.25.1' 'transformers==4.38.0' \
  'huggingface-hub==0.25.1' 'accelerate==0.28.0' 'xformers==0.0.27.post2' \
  'peft==0.9.0'

echo "Environment ready: $ENV_PREFIX"
echo "Activate with: conda activate $ENV_PREFIX"
