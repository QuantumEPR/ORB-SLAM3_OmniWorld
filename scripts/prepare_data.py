#!/usr/bin/env python3
import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

import numpy as np

from utils import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_METADATA_CSV,
    DEFAULT_OUTPUT_ROOT,
    read_tar_json,
    read_tar_text,
    scene_archive,
    selected_scene_splits,
    split_count,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract an OmniWorld split and convert it to an ORB-SLAM3 RGB-D sequence."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory of the downloaded OmniWorld dataset.",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        default=[],
        help="OmniWorld scene UID, for example 63ad1dbede39.",
    )
    parser.add_argument(
        "--split-idx",
        type=int,
        action="append",
        default=[],
        help="Scene split index to prepare. Repeat for multiple splits. Defaults to all splits.",
    )
    parser.add_argument(
        "--split-list",
        type=Path,
        default=None,
        help="Optional text file with scene_id, scene_id/split_XX, or scene_id,split_XX lines.",
    )
    parser.add_argument(
        "--print-split-count",
        action="store_true",
        help="Print the number of available splits for the requested scene and exit.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the prepared split will be written.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=DEFAULT_METADATA_CSV,
        help="Metadata CSV with per-scene metric scale.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every Nth frame from the requested split.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of exported frames after stride is applied.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip extraction when the requested prepared split already looks complete.",
    )
    parser.add_argument(
        "--depth-scale-factor",
        type=float,
        default=1000.0,
        help="Depth PNG scale used in the exported sequence. 1000.0 means millimeters.",
    )
    parser.add_argument(
        "--th-depth-m",
        type=float,
        default=40.0,
        help="Depth threshold in meters used to generate ORB-SLAM3's RGB-D config.",
    )
    return parser.parse_args()


def load_metric_scale(scene_id: str, metadata_csv: Path) -> float:
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["UID"] == scene_id:
                return float(row["Metric Scale"])
    raise KeyError(f"Scene {scene_id} was not found in {metadata_csv}")


def decode_omniworld_depth(raw_depth: np.ndarray, metric_scale: float) -> np.ndarray:
    depthmap = raw_depth.astype(np.float32) / 65535.0
    near_mask = depthmap < 0.0015
    far_mask = depthmap > (65500.0 / 65535.0)
    near = 1.0
    far = 1000.0
    depthmap = depthmap / (far - depthmap * (far - near)) / 0.004
    valid = ~(near_mask | far_mask)
    depthmap[~valid] = 0.0
    depthmap[valid] *= float(metric_scale)
    return depthmap


def encode_metric_depth(depth_m: np.ndarray, depth_scale_factor: float) -> np.ndarray:
    encoded = np.rint(depth_m * depth_scale_factor)
    encoded = np.clip(encoded, 0, np.iinfo(np.uint16).max)
    return encoded.astype(np.uint16)


def extract_needed_members(
    tar_paths,
    requested_names,
    output_dir: Path,
    depth_mode: bool,
    metric_scale: float,
    depth_scale_factor: float,
):
    import imageio.v2 as imageio

    missing = set(requested_names)
    output_dir.mkdir(parents=True, exist_ok=True)

    for tar_path in tar_paths:
        if not missing:
            break
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar:
                if member.name not in missing:
                    continue
                with tar.extractfile(member) as handle:
                    payload = handle.read()
                output_path = output_dir / Path(member.name).name
                if depth_mode:
                    raw_depth = imageio.imread(io.BytesIO(payload))
                    metric_depth = decode_omniworld_depth(raw_depth, metric_scale)
                    encoded_depth = encode_metric_depth(metric_depth, depth_scale_factor)
                    imageio.imwrite(output_path, encoded_depth)
                else:
                    output_path.write_bytes(payload)
                missing.remove(member.name)

    if missing:
        missing_preview = ", ".join(sorted(list(missing))[:5])
        raise FileNotFoundError(f"Failed to extract {len(missing)} members, including {missing_preview}")


def build_ground_truth(split_frames, timestamps, cam_json, metric_scale: float):
    from scipy.spatial.transform import Rotation as R

    quat_wxyz = np.asarray(cam_json["quats"], dtype=np.float64)
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=1)
    rotations_cw = R.from_quat(quat_xyzw).as_matrix()
    translations_cw = np.asarray(cam_json["trans"], dtype=np.float64)

    rows = []
    for i, frame_id in enumerate(split_frames):
        rotation_cw = rotations_cw[i]
        translation_cw = translations_cw[i]
        rotation_wc = rotation_cw.T
        translation_wc = -rotation_wc @ translation_cw
        translation_wc = translation_wc * metric_scale
        quat_wc_xyzw = R.from_matrix(rotation_wc).as_quat()
        rows.append(
            (
                timestamps[i],
                translation_wc[0],
                translation_wc[1],
                translation_wc[2],
                quat_wc_xyzw[0],
                quat_wc_xyzw[1],
                quat_wc_xyzw[2],
                quat_wc_xyzw[3],
                frame_id,
            )
        )
    return rows


def write_text_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")


def write_settings_yaml(
    path: Path,
    width: int,
    height: int,
    fps: float,
    focal_values,
    cx: float,
    cy: float,
    depth_scale_factor: float,
    th_depth_m: float,
):
    fx = float(np.median(focal_values))
    fy = fx
    fps_int = max(1, int(round(fps)))
    stereo_b = 0.1
    stereo_th_depth = th_depth_m / stereo_b
    content = f"""%YAML:1.0

File.version: "1.0"

Camera.type: "PinHole"
Camera1.fx: {fx:.9f}
Camera1.fy: {fy:.9f}
Camera1.cx: {cx:.9f}
Camera1.cy: {cy:.9f}

Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera1.k3: 0.0

Camera.width: {width}
Camera.height: {height}
Camera.fps: {fps_int}
Camera.RGB: 1

Stereo.ThDepth: {stereo_th_depth:.6f}
Stereo.b: {stereo_b:.6f}
RGBD.DepthMapFactor: {depth_scale_factor:.6f}

ORBextractor.nFeatures: 1500
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""
    path.write_text(content, encoding="utf-8")


def is_prepared_output_complete(output_dir: Path, frame_ids) -> bool:
    sequence_dir = output_dir / "sequence"
    color_dir = sequence_dir / "color"
    depth_dir = sequence_dir / "depth"
    required_paths = [
        output_dir / "manifest.json",
        output_dir / "groundtruth_tum.txt",
        output_dir / "omniworld_rgbd.yaml",
        sequence_dir / "associations.txt",
    ]
    if any(not path.is_file() for path in required_paths):
        return False

    expected_count = len(frame_ids)
    color_count = sum(1 for _ in color_dir.glob("*.png"))
    depth_count = sum(1 for _ in depth_dir.glob("*.png"))
    return color_count == expected_count and depth_count == expected_count


def prepare_one(args, scene_id: str, split_idx: int):
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")

    others_tar = scene_archive(args.dataset_root, scene_id)
    if not others_tar.exists():
        raise FileNotFoundError(f"Missing scene annotations archive: {others_tar}")

    split_info = read_tar_json(others_tar, "split_info.json")
    split_count = len(split_info["split"])
    if args.print_split_count:
        print(split_count)
        return
    if not 0 <= split_idx < split_count:
        raise IndexError(
            f"Split index {split_idx} is out of range for scene {scene_id}; valid range is 0..{split_count - 1}"
        )

    fps_text = read_tar_text(others_tar, "fps.txt")
    cam_json = read_tar_json(others_tar, f"camera/split_{split_idx}.json")
    metric_scale = load_metric_scale(scene_id, args.metadata_csv)

    raw_frames = split_info["split"][split_idx]
    if args.max_frames is not None:
        raw_frames = raw_frames[:: args.frame_stride][: args.max_frames]
    else:
        raw_frames = raw_frames[:: args.frame_stride]
    if not raw_frames:
        raise ValueError("No frames selected after applying split, stride, and max-frame options")

    output_dir = args.output_root / scene_id / f"split_{split_idx:02d}"
    if args.skip_existing and is_prepared_output_complete(output_dir, raw_frames):
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {"prepared_dir": str(output_dir), "skipped": True, **manifest},
                indent=2,
            )
        )
        return

    selected_positions = list(range(0, len(split_info["split"][split_idx]), args.frame_stride))
    if args.max_frames is not None:
        selected_positions = selected_positions[: args.max_frames]

    selected_focals = [cam_json["focals"][i] for i in selected_positions]
    selected_quats = [cam_json["quats"][i] for i in selected_positions]
    selected_trans = [cam_json["trans"][i] for i in selected_positions]
    reduced_cam_json = {
        "focals": selected_focals,
        "quats": selected_quats,
        "trans": selected_trans,
        "cx": cam_json["cx"],
        "cy": cam_json["cy"],
    }

    fps = float(fps_text.split("FPS:")[1].splitlines()[0].strip())
    timestamps = [frame_id / fps for frame_id in raw_frames]

    sequence_dir = output_dir / "sequence"
    color_dir = sequence_dir / "color"
    depth_dir = sequence_dir / "depth"

    color_names = {f"color/{frame_id:06d}.png" for frame_id in raw_frames}
    depth_names = {f"depth/{frame_id:06d}.png" for frame_id in raw_frames}

    rgb_tar_paths = sorted(
        (
            args.dataset_root / "videos" / "OmniWorld-Game" / scene_id
        ).glob(f"{scene_id}_rgb_*.tar.gz")
    )
    depth_tar_paths = sorted(
        (
            args.dataset_root / "annotations" / "OmniWorld-Game" / scene_id
        ).glob(f"{scene_id}_depth_*.tar.gz")
    )
    if not rgb_tar_paths or not depth_tar_paths:
        raise FileNotFoundError("RGB or depth shards are missing for the requested scene")

    extract_needed_members(
        rgb_tar_paths,
        color_names,
        color_dir,
        depth_mode=False,
        metric_scale=metric_scale,
        depth_scale_factor=args.depth_scale_factor,
    )
    extract_needed_members(
        depth_tar_paths,
        depth_names,
        depth_dir,
        depth_mode=True,
        metric_scale=metric_scale,
        depth_scale_factor=args.depth_scale_factor,
    )

    import imageio.v2 as imageio

    first_rgb = imageio.imread(color_dir / f"{raw_frames[0]:06d}.png")
    height, width = first_rgb.shape[:2]

    association_lines = []
    for timestamp, frame_id in zip(timestamps, raw_frames):
        rel_color = f"color/{frame_id:06d}.png"
        rel_depth = f"depth/{frame_id:06d}.png"
        association_lines.append(
            f"{timestamp:.6f} {rel_color} {timestamp:.6f} {rel_depth}"
        )
    write_text_lines(sequence_dir / "associations.txt", association_lines)

    gt_rows = build_ground_truth(raw_frames, timestamps, reduced_cam_json, metric_scale)
    gt_lines = [
        f"{ts:.6f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
        for ts, tx, ty, tz, qx, qy, qz, qw, _ in gt_rows
    ]
    write_text_lines(output_dir / "groundtruth_tum.txt", gt_lines)

    write_settings_yaml(
        output_dir / "omniworld_rgbd.yaml",
        width=width,
        height=height,
        fps=fps,
        focal_values=selected_focals,
        cx=float(cam_json["cx"]),
        cy=float(cam_json["cy"]),
        depth_scale_factor=args.depth_scale_factor,
        th_depth_m=args.th_depth_m,
    )

    manifest = {
        "scene_id": scene_id,
        "split_idx": split_idx,
        "frame_count": len(raw_frames),
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "metric_scale": metric_scale,
        "fps": fps,
        "image_width": width,
        "image_height": height,
        "depth_scale_factor": args.depth_scale_factor,
        "selected_frames": raw_frames,
        "median_focal": float(np.median(selected_focals)),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(json.dumps({"prepared_dir": str(output_dir), **manifest}, indent=2))


def main():
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    if args.print_split_count:
        if len(args.scene_id) != 1:
            raise SystemExit("--print-split-count requires exactly one --scene-id")
        print(split_count(args.dataset_root, args.scene_id[0]))
        return
    jobs = selected_scene_splits(
        args.dataset_root,
        scene_ids=args.scene_id,
        split_indices=args.split_idx,
        split_list=args.split_list,
    )
    if not jobs:
        raise SystemExit("No OmniWorld scenes/splits were found to prepare.")
    print(f"Preparing {len(jobs)} split(s) under {args.output_root}", flush=True)
    for index, (scene_id, split_idx) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {scene_id}/split_{split_idx:02d}", flush=True)
        prepare_one(args, scene_id, split_idx)


if __name__ == "__main__":
    main()
