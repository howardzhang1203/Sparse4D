#!/usr/bin/env python3

"""Convert custom-format videos and run Sparse4D inference.

This utility creates a nuScenes-like info file from a custom video source and
reuses the existing ``tools/test.py`` inference pipeline.

Supported sources:
* OTO tester formats (TLV/YUV) via ``ototester`` / ``tlv_tools`` interfaces.
* Generic ffmpeg-decodable files as fallback (OpenCV + ffmpeg).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mmcv
import numpy as np


DEFAULT_CAMERA_MAP = [
    ("CAM_FRONT", 0),
    ("CAM_FRONT_RIGHT", 1),
    ("CAM_FRONT_LEFT", 2),
    ("CAM_BACK", 3),
    ("CAM_BACK_LEFT", 4),
    ("CAM_BACK_RIGHT", 5),
]


def parse_camera_map(raw_map: str, num_cams: int) -> List[Tuple[str, int]]:
    entries = [x.strip() for x in raw_map.split(",") if x.strip()]
    if len(entries) == 0:
        return DEFAULT_CAMERA_MAP[:num_cams]
    parsed: List[Tuple[str, int]] = []
    for entry in entries:
        if ":" in entry:
            name, idx = entry.split(":", 1)
        elif "=" in entry:
            name, idx = entry.split("=", 1)
        else:
            raise ValueError(
                f"camera map token must be name:index format, got {entry!r}"
            )
        parsed.append((name.strip(), int(idx.strip())))

    if len(parsed) > num_cams:
        parsed = parsed[:num_cams]
    if len(parsed) < num_cams:
        for name, _ in DEFAULT_CAMERA_MAP[len(parsed):num_cams]:
            parsed.append((name, len(parsed)))
    return parsed


def parse_intrinsics(json_or_none: Optional[str], num_cams: int) -> List[List[List[float]]]:
    if not json_or_none:
        return [np.eye(3).tolist() for _ in range(num_cams)]

    path = Path(json_or_none)
    if not path.exists():
        raise FileNotFoundError(f"Intrinsic file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    mats = []
    if isinstance(raw, dict):
        if "intrinsics" in raw and isinstance(raw["intrinsics"], list):
            entries = raw["intrinsics"]
        else:
            # allow {"CAM_FRONT": [[...]], ...}
            # fill from keys listed in camera map if present
            entries = []
            for name in [
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_FRONT_LEFT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]:
                if name in raw:
                    entries.append(raw[name])
            if not entries:
                raise ValueError("intrinsic json dict must contain an intrinsics list")
    else:
        entries = raw

    if not isinstance(entries, list):
        raise ValueError("intrinsic file must be json list or dict")
    if len(entries) == 0:
        return [np.eye(3).tolist() for _ in range(num_cams)]

    for i in range(num_cams):
        if i < len(entries):
            mat = np.array(entries[i], dtype=np.float32)
            if mat.shape != (3, 3):
                raise ValueError("each intrinsic matrix must be 3x3")
            mats.append(mat.tolist())
        else:
            mats.append(np.eye(3).tolist())
    return mats


def _quaternion_from_yaw(yaw: float) -> List[float]:
    half = float(yaw) * 0.5
    return [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]


def _first_available(header: Dict[str, Any], names: Sequence[str]) -> Optional[float]:
    for key in names:
        if key in header and header[key] is not None:
            return float(header[key])
    return None


def _header_pose_from_odom(header: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    x = _first_available(
        header,
        [
            "odom_rtk_x",
            "odom_x",
            "odom_dr_x",
            "odom_dr_rtk_x",
            "odometry_x",
            "center_x",
        ],
    )
    y = _first_available(
        header,
        [
            "odom_rtk_y",
            "odom_y",
            "odom_dr_y",
            "odom_dr_rtk_y",
            "odometry_y",
            "center_y",
        ],
    )
    yaw = _first_available(
        header,
        [
            "odom_rtk_theta",
            "odom_theta",
            "odom_dr_theta",
            "odom_dr_rtk_theta",
            "odometry_theta",
            "center_theta",
        ],
    )

    translation = [0.0 if v is None else float(v) for v in (x, y, 0.0)]
    rotation = _quaternion_from_yaw(0.0 if yaw is None else yaw)
    return translation, rotation


def _timestamp_us(
    header: Dict[str, Any], channel_idx: int, frame_idx: int, fallback_fps: int
) -> int:
    candidates = [
        f"frame_timestamp_ch{channel_idx}",
        f"frame_timestamp_ch{channel_idx}_us",
        "timestamp",
        "frame_timestamp",
    ]
    for key in candidates:
        val = header.get(key) if isinstance(header, dict) else None
        if isinstance(val, (int, float)):
            return int(val)
    return int(frame_idx * (1_000_000.0 / float(fallback_fps)))


class OtoTesterFrameReader:
    """Read frames via OTO plugin/param path and convert to frame/header pairs."""

    def __init__(
        self,
        video_path: str,
        plugin_path: str = "",
        param_path: str = "",
        enable_ffmpeg: bool = False,
        formula_type: int = 1,
        disable_reset: bool = False,
        channel_aligned: bool = False,
        frame_start: int = 0,
        frame_end: int = -1,
    ) -> None:
        self.video_path = video_path
        self.plugin_path = plugin_path
        self.param_path = param_path
        self.enable_ffmpeg = enable_ffmpeg
        self.formula_type = formula_type
        self.disable_reset = disable_reset
        self.channel_aligned = channel_aligned
        self.frame_start = max(int(frame_start), 0)
        self.frame_end = frame_end
        self.state: Dict[str, Any] = {"header": None, "images": None}
        self._init_runner()

    def _init_runner(self) -> None:
        from ototester import DLLRunner, Plugin

        self.DLLRunner = DLLRunner
        self.Plugin = Plugin
        self.runner = DLLRunner(
            self.video_path,
            plugin_path=self.plugin_path,
            param_path=self.param_path,
            cache_path="",
            cache_mode=0,
            dst_fmt="BGR3",
            debug=False,
            enable_ffmpeg=self.enable_ffmpeg,
        )

        def on_header(_: Any, hdr: Dict[str, Any]) -> None:
            self.state["header"] = hdr

        def on_image(_: Any, imgs: Sequence[np.ndarray]) -> None:
            self.state["images"] = list(imgs) if imgs is not None else []

        if self.plugin_path:
            plugin = self.Plugin(
                on_header,
                on_image,
                img_fmt="BGR3",
                formula_type=self.formula_type,
                disable_reset=self.disable_reset,
                channel_aligned=self.channel_aligned,
            )
            self.runner.add_plugin(plugin)
        else:
            # no plugin for decoding: still need callbacks to capture frames
            plugin = self.Plugin(
                on_header,
                on_image,
                img_fmt="BGR3",
                formula_type=self.formula_type,
                disable_reset=self.disable_reset,
                channel_aligned=self.channel_aligned,
            )
            self.runner.add_plugin(plugin)

    def __iter__(self):
        runner = self.runner
        total = runner.get_frame_count()
        start = min(self.frame_start, total if total > 0 else self.frame_start)
        end = self.frame_end if self.frame_end >= 0 else total

        if start > 0:
            runner.set_frame_id(start)
        for frame_id in range(start, end):
            status = runner.run_frame()
            code = status.code if hasattr(status, "code") else int(status)
            if code != 0:
                if code > 0:
                    break
                raise RuntimeError(f"Frame decode error at {frame_id}: {status}")

            header = self.state.get("header")
            images = self.state.get("images")
            if header is None:
                continue
            yield frame_id, copy.deepcopy(header), copy.deepcopy(images or [])


class FFmpegFallbackReader:
    """Read generic videos with OpenCV + ffmpeg."""

    def __init__(self, video_path: str, frame_start: int = 0, frame_end: int = -1):
        self.video_path = video_path
        self.frame_start = frame_start
        self.frame_end = frame_end

    def __iter__(self):
        import cv2

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {self.video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = max(cap.get(cv2.CAP_PROP_FPS) or 25.0, 1.0)
        start = max(self.frame_start, 0)
        end = self.frame_end if self.frame_end >= 0 else total
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for frame_id in range(start, end):
            ok, frame = cap.read()
            if not ok:
                break
            ts = int(frame_id * (1_000_000.0 / fps))
            header = {
                "frame_timestamp_ch0": ts,
                "frame_width_ch0": int(frame.shape[1]),
                "frame_height_ch0": int(frame.shape[0]),
            }
            yield frame_id, header, [frame]
        cap.release()


def _build_cam_info(
    sample_token: str,
    cam_name: str,
    img_path: str,
    timestamp: int,
    intrinsics: Sequence[Sequence[float]],
    pose_translation: Sequence[float],
    pose_rotation: Sequence[float],
) -> Dict[str, Any]:
    return {
        "type": "camera",
        "data_path": img_path,
        "sample_data_token": sample_token,
        "sensor2ego_translation": [0.0, 0.0, 0.0],
        "sensor2ego_rotation": [1.0, 0.0, 0.0, 0.0],
        "ego2global_translation": list(map(float, pose_translation)),
        "ego2global_rotation": list(map(float, pose_rotation)),
        "cam_intrinsic": [list(map(float, row)) for row in intrinsics],
        "sensor2lidar_translation": [0.0, 0.0, 0.0],
        "sensor2lidar_rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "timestamp": int(timestamp),
    }


def _iter_frames(
    video_path: str,
    plugin_path: str = "",
    param_path: str = "",
    enable_ffmpeg: bool = False,
    frame_start: int = 0,
    frame_end: int = -1,
) -> Any:
    if plugin_path or param_path:
        try:
            return OtoTesterFrameReader(
                video_path,
                plugin_path=plugin_path,
                param_path=param_path,
                enable_ffmpeg=enable_ffmpeg,
                frame_start=frame_start,
                frame_end=frame_end,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize OTO reader. If this is a TLV/YUV file, "
                "provide the plugin and param that match your dataset format."
            ) from exc
    try:
        from tlv_tools import DataGetter

        class _TLVReader:
            def __init__(self, path: str, frame_start: int = 0, frame_end: int = -1):
                self.getter = DataGetter(path, enable_ffmpeg=enable_ffmpeg)
                self.frame_start = frame_start
                self.frame_end = frame_end

            def __iter__(self):
                total = self.getter.get_frame_count()
                end = self.frame_end if self.frame_end >= 0 else total
                for frame_id in range(self.frame_start, end):
                    hdr, imgs = self.getter.get_header_image(frame_id)
                    yield frame_id, copy.deepcopy(hdr), copy.deepcopy(imgs)

        return _TLVReader(video_path, frame_start=frame_start, frame_end=frame_end)
    except Exception:
        return FFmpegFallbackReader(
            video_path,
            frame_start=frame_start,
            frame_end=frame_end,
        )


def build_custom_infos(
    video_path: str,
    ann_file: str,
    image_root: str,
    camera_map: List[Tuple[str, int]],
    frame_start: int = 0,
    frame_end: int = -1,
    num_cams: int = 6,
    plugin_path: str = "",
    param_path: str = "",
    enable_ffmpeg: bool = False,
    intrinsics: Optional[List[List[List[float]]]] = None,
    fallback_fps: int = 10,
) -> int:
    image_root = Path(image_root)
    image_root.mkdir(parents=True, exist_ok=True)
    for i in range(num_cams):
        (image_root / f"cam_{i:02d}_{camera_map[i][0]}").mkdir(
            parents=True, exist_ok=True
        )

    intrinsics = intrinsics or [np.eye(3).tolist() for _ in range(num_cams)]

    infos = []
    reader = _iter_frames(
        video_path,
        plugin_path=plugin_path,
        param_path=param_path,
        enable_ffmpeg=enable_ffmpeg,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    for frame_id, header, images in reader:
        if not images:
            continue

        pose_t, pose_r = _header_pose_from_odom(header)
        sample_token = f"custom_{frame_id:06d}"
        first_valid = images[0]
        if first_valid is None or not isinstance(first_valid, np.ndarray):
            continue

        cams: Dict[str, Any] = {}
        for cam_name, ch in camera_map[:num_cams]:
            if ch < len(images) and images[ch] is not None and isinstance(
                images[ch], np.ndarray
            ):
                img = images[ch]
            else:
                img = first_valid

            ts = _timestamp_us(
                header,
                channel_idx=ch,
                frame_idx=frame_id,
                fallback_fps=fallback_fps,
            )
            cam_dir = image_root / f"cam_{ch:02d}_{cam_name}"
            img_path = cam_dir / f"{frame_id:06d}.jpg"
            mmcv.imwrite(img, str(img_path))

            cam_info = _build_cam_info(
                sample_token,
                cam_name,
                str(img_path),
                ts,
                intrinsics[min(ch, len(intrinsics) - 1)],
                pose_t,
                pose_r,
            )
            cam_info["data_path"] = str(img_path)
            cams[cam_name] = cam_info

        timestamp = _timestamp_us(
            header,
            channel_idx=camera_map[0][1],
            frame_idx=frame_id,
            fallback_fps=fallback_fps,
        )
        first_cam_name = camera_map[0][0]
        infos.append(
            {
                "token": sample_token,
                "lidar_path": cams[first_cam_name]["data_path"],
                "sweeps": [],
                "timestamp": int(timestamp),
                "lidar2ego_translation": [0.0, 0.0, 0.0],
                "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
                "ego2global_translation": list(pose_t),
                "ego2global_rotation": list(pose_r),
                "cams": cams,
            }
        )

    ann_parent = Path(ann_file).parent
    ann_parent.mkdir(parents=True, exist_ok=True)
    mmcv.dump({"infos": sorted(infos, key=lambda i: i["timestamp"]), "metadata": {"version": "v1.0-custom"}}, ann_file)
    return len(infos)


def _build_test_command(
    config: str,
    checkpoint: str,
    ann_file: str,
    output_file: str,
    show_dir: Optional[str],
    no_eval: bool,
    eval_only_show: bool,
    conda_python: Optional[str] = None,
) -> List[str]:
    command = [conda_python or sys.executable, "tools/test.py", config, checkpoint]

    if no_eval:
        command += ["--out", output_file]
        if show_dir:
            command += ["--show-dir", show_dir]
        command += [
            "--cfg-options",
            f"data.test.ann_file={ann_file}",
            f"data.val.ann_file={ann_file}",
            "data.workers_per_gpu=0",
        ]
    else:
        command += [
            "--out",
            output_file,
            "--eval",
            "bbox",
            "--cfg-options",
            f"data.test.ann_file={ann_file}",
            f"data.val.ann_file={ann_file}",
            f"data.test.version=v1.0-custom",
            f"data.val.version=v1.0-custom",
            "data.workers_per_gpu=0",
        ]
        if show_dir:
            command += ["--show-dir", show_dir]
        if eval_only_show:
            command.append("--eval")

    return command


def _images_to_video(visual_dir: str, video_out: str, fps: int = 10) -> None:
    import cv2

    vis_root = Path(visual_dir)
    if not vis_root.exists():
        return

    img_files = sorted(vis_root.glob("*.jpg"))
    if not img_files:
        return

    first = cv2.imread(str(img_files[0]))
    if first is None:
        return
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(video_out, fourcc, float(fps), (w, h))
    for img_path in img_files:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        writer.write(image)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build custom video infos and run Sparse4D inference"
    )
    parser.add_argument("--video", required=True, help="Path to custom video")
    parser.add_argument(
        "--plugin", default="", help="Path to decoding plugin/dll (for TLV/YUV)"
    )
    parser.add_argument(
        "--param", default="", help="Path to plugin param file (for TLV/YUV)"
    )
    parser.add_argument("--enable-ffmpeg", action="store_true", help="Enable ffmpeg path in OTO reader")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=-1, help="End frame index (exclusive)")
    parser.add_argument(
        "--num-cams",
        type=int,
        default=6,
        help="Number of virtual cameras expected by model input",
    )
    parser.add_argument(
        "--camera-map",
        default=",".join([f"{name}:{idx}" for name, idx in DEFAULT_CAMERA_MAP]),
        help="Camera map in name:index format, comma-separated",
    )
    parser.add_argument(
        "--intrinsics",
        default="",
        help="Optional JSON file with 3x3 matrices (single list or per-camera dict/list)",
    )
    parser.add_argument(
        "--ann-file",
        default="outputs/custom_video/custom_video_infos_test.pkl",
        help="Output info pkl path",
    )
    parser.add_argument(
        "--images-root",
        default="outputs/custom_video/images",
        help="Directory to store extracted frames",
    )
    parser.add_argument(
        "--config",
        default="projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py",
        help="Sparse4D test config path",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Sparse4D checkpoint path",
    )
    parser.add_argument(
        "--out",
        default="outputs/custom_video/results.pkl",
        help="Inference output pickle path",
    )
    parser.add_argument(
        "--show-dir",
        default="outputs/custom_video/visual",
        help="Directory to save inference visualization images",
    )
    parser.add_argument(
        "--video-out",
        default="outputs/custom_video/visual/video.avi",
        help="Path of generated output video",
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Only build infos pkl, do not run tools/test.py",
    )
    parser.add_argument(
        "--fallback-fps",
        type=int,
        default=10,
        help="FPS used when source timestamp is unavailable",
    )
    parser.add_argument(
        "--disable-eval",
        action="store_true",
        help="Skip --eval for datasets without labels and still produce visualization",
    )
    parser.add_argument("--python", default="", help="Python executable to call for inference")

    args = parser.parse_args()

    camera_map = parse_camera_map(args.camera_map, args.num_cams)
    intrinsics = parse_intrinsics(args.intrinsics, args.num_cams)

    num_frames = build_custom_infos(
        video_path=args.video,
        ann_file=args.ann_file,
        image_root=args.images_root,
        camera_map=camera_map,
        frame_start=args.start,
        frame_end=args.end,
        num_cams=args.num_cams,
        plugin_path=args.plugin,
        param_path=args.param,
        enable_ffmpeg=args.enable_ffmpeg,
        intrinsics=intrinsics,
        fallback_fps=args.fallback_fps,
    )
    print(f"[INFO] Wrote {num_frames} frames to {args.ann_file}")

    if args.no_inference:
        return

    show_dir = args.show_dir
    command = _build_test_command(
        config=args.config,
        checkpoint=args.checkpoint,
        ann_file=args.ann_file,
        output_file=args.out,
        show_dir=show_dir,
        no_eval=args.disable_eval,
        eval_only_show=False,
        conda_python=args.python or None,
    )
    print("[INFO] Running:", " ".join(command))
    subprocess.run(command, check=True)

    _images_to_video(show_dir, args.video_out, fps=10)
    print(f"[INFO] Video saved to {args.video_out}")


if __name__ == "__main__":
    main()
