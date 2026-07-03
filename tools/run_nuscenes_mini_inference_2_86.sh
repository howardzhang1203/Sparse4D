#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-/share/work_hdd/howardzhang/anaconda/envs/sparse4d}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

export PATH="${CONDA_ENV}/bin:${CUDA_HOME}/bin:/usr/bin:/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd "${REPO_ROOT}"
"${CONDA_ENV}/bin/python" tools/test.py \
  projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py \
  ckpt/sparse4dv3_r50.pth \
  --out outputs/sparse4d_nuscenes_mini/results.pkl \
  --eval bbox \
  --cfg-options \
    data.test.version=v1.0-mini \
    data.test.ann_file=data/nuscenes_anno_pkls/nuscenes-mini_infos_val.pkl \
    data.val.version=v1.0-mini \
    data.val.ann_file=data/nuscenes_anno_pkls/nuscenes-mini_infos_val.pkl \
    data.workers_per_gpu=0 \
  --eval-options out_dir=outputs/sparse4d_nuscenes_mini
