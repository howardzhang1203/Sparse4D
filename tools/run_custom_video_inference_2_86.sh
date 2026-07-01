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

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <VIDEO_PATH> <CHECKPOINT_PATH> [extra args]" >&2
  echo "Example: $0 /path/to/video.tlv ckpt/sparse4dv3_r50.pth --plugin /path/to/plugin.dll --param /path/to/param.yml" >&2
  exit 1
fi

VIDEO_PATH="$1"
CHECKPOINT="$2"
shift 2

"${CONDA_ENV}/bin/python" tools/run_custom_video_inference.py \
  --video "${VIDEO_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --config projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py \
  --disable-eval \
  "$@"
