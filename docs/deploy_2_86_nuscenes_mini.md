# Sparse4D Deployment on 192.168.2.86

This note records the working Sparse4D deployment used on `MLServer7`
(`192.168.2.86`) for nuScenes mini inference.

## Environment

- Host: `MLServer7`
- GPUs: 2x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- Driver: `580.95.05`
- CUDA toolkit: `/usr/local/cuda-13.0`
- Conda env: `/share/work_hdd/howardzhang/anaconda/envs/sparse4d`
- Repository: `/share/work_hdd/howardzhang/sparse4d_deploy/Sparse4D`

The upstream quick start pins `torch==1.13.0+cu116`, but that stack is not a
good fit for the Blackwell GPUs on this host. This deployment uses the existing
CUDA 13/PyTorch 2.9-compatible runtime instead.

Important package versions:

- `torch==2.9.0+cu130`
- `mmcv==1.7.1`
- `mmdet==2.28.2`
- `numpy==1.23.5`
- `nuscenes-devkit==1.1.10`
- `motmetrics==1.1.3`

## Compatibility Fixes

- Built the Sparse4D custom CUDA op with `TORCH_CUDA_ARCH_LIST=12.0` for
  Blackwell (`sm_120`).
- Used lightweight `mmcv==1.7.1` instead of `mmcv-full==1.7.1`, because the
  old `mmcv-full` extension source is not compatible with PyTorch 2.9/CUDA 13.
- Patched `mmcv.utils.ext_loader` in the conda environment so imports do not
  fail when `mmcv._ext` is unavailable. Sparse4D inference uses its own custom
  CUDA op, not the missing mmcv CUDA ops.
- Patched `mmcv.parallel._functions` in the conda environment so PyTorch 2.9
  receives a `torch.device` when calling `_get_stream`.
- Patched `motmetrics.metrics` in the conda environment to import `Iterable`
  from `collections.abc`, which is required on Python 3.10.
- Set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for `tools/test.py`, because newer
  PyTorch defaults can otherwise reject older checkpoint pickle payloads.

## Downloaded Assets

- nuScenes mini: `data/v1.0-mini.tgz`
- Extracted dataset root: `data/nuscenes`
- Converted annotations:
  - `data/nuscenes_anno_pkls/nuscenes-mini_infos_train.pkl`
  - `data/nuscenes_anno_pkls/nuscenes-mini_infos_val.pkl`
- Sparse4Dv3 checkpoint: `ckpt/sparse4dv3_r50.pth`
- ResNet-50 backbone: `ckpt/resnet50-19c8e357.pth`
- Anchor file: `nuscenes_kmeans900.npy`

The full nuScenes dataset was not downloaded for this pass because the
available disk budget on the host was insufficient. This deployment validates
the pipeline with `v1.0-mini`.

## Rebuild the Custom Op

```bash
bash tools/compile_sparse4d_op_2_86.sh
```

## Run nuScenes Mini Inference

```bash
bash tools/run_nuscenes_mini_inference_2_86.sh
```

## Verified Output

- Result pickle:
  `outputs/sparse4d_nuscenes_mini/results.pkl`
- Visualization video:
  `outputs/sparse4d_nuscenes_mini/visual/video.avi`
- Video verification:
  - OpenCV opened the video successfully.
  - Frame count: `81`
  - FPS: `7.0`
  - Resolution: `6600x1800`
  - First frame read: `True`

Detection evaluation on nuScenes mini:

- `NDS: 0.5070`
- `mAP: 0.4645`
