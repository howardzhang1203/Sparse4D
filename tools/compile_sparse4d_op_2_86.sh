#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-/share/work_hdd/howardzhang/anaconda/envs/sparse4d}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

export PATH="${CONDA_ENV}/bin:${CUDA_HOME}/bin:/usr/bin:/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-4}"

cd "${REPO_ROOT}/projects/mmdet3d_plugin/ops"
"${CONDA_ENV}/bin/python" setup.py build_ext --inplace
