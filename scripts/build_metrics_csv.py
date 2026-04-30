#!/usr/bin/env python3
"""Build one long-form ORB-SLAM3 metrics CSV with the columns in columns.txt."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from utils import (
    DEFAULT_OUTPUT_ROOT,
    MODE_NAMES,
    MODES,
    REPO_ROOT,
    discover_prepared_splits,
    load_json,
    result_dir,
    split_index,
)

DEFAULT_COLUMNS = REPO_ROOT / "columns.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--columns", type=Path, default=DEFAULT_COLUMNS)
    parser.add_argument("--split-list", type=Path, default=None)
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument("--split-idx", type=int, action="append", default=[])
    parser.add_argument("--modes", nargs="+", choices=MODE_NAMES, default=MODE_NAMES)
    return parser.parse_args()


def load_columns(path: Path) -> list[str]:
    columns: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.search(r'"([^"]+)"', line)
        if match:
            columns.append(match.group(1))
    if not columns:
        raise ValueError(f"No quoted column names found in {path}")
    return columns


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray, R]:
    timestamps = []
    positions = []
    quats = []
    if not path.is_file():
        return np.asarray([]), np.empty((0, 3)), R.from_quat([[0.0, 0.0, 0.0, 1.0]])
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 8:
            timestamps.append(float(parts[0]))
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            quats.append([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
    if not timestamps:
        return np.asarray([]), np.empty((0, 3)), R.from_quat([[0.0, 0.0, 0.0, 1.0]])
    return np.asarray(timestamps), np.asarray(positions), R.from_quat(np.asarray(quats))


def path_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(positions[1:] - positions[:-1], axis=1).sum())


def motion_stats(split_dir: Path) -> dict[str, Any]:
    ts, pos, rot = load_tum(split_dir / "groundtruth_tum.txt")
    if len(ts) == 0:
        return {}
    steps = np.linalg.norm(pos[1:] - pos[:-1], axis=1) if len(pos) >= 2 else np.asarray([])
    dt = np.maximum(1e-12, ts[1:] - ts[:-1]) if len(ts) >= 2 else np.asarray([])
    rot_steps = np.asarray([])
    if len(ts) >= 2:
        rel = rot[:-1].inv() * rot[1:]
        rot_steps = np.abs(rel.magnitude()) * 180.0 / math.pi
    total_path = path_length(pos)
    chord = float(np.linalg.norm(pos[-1] - pos[0])) if len(pos) >= 2 else 0.0
    duration = float(ts[-1] - ts[0]) if len(ts) >= 2 else 0.0
    return {
        "duration_sec": duration,
        "gt_path_length_m": total_path,
        "gt_chord_length_m": chord,
        "gt_straightness": chord / total_path if total_path > 0 else "",
        "gt_median_step_m": float(np.median(steps)) if len(steps) else "",
        "gt_mean_step_m": float(np.mean(steps)) if len(steps) else "",
        "gt_max_step_m": float(np.max(steps)) if len(steps) else "",
        "gt_mean_speed_mps": float(np.mean(steps / dt)) if len(steps) else "",
        "gt_max_speed_mps": float(np.max(steps / dt)) if len(steps) else "",
        "gt_total_rotation_deg": float(np.sum(rot_steps)) if len(rot_steps) else "",
        "gt_mean_rotation_step_deg": float(np.mean(rot_steps)) if len(rot_steps) else "",
        "gt_max_rotation_step_deg": float(np.max(rot_steps)) if len(rot_steps) else "",
        "gt_rotation_per_meter_deg": float(np.sum(rot_steps) / total_path) if total_path > 0 and len(rot_steps) else "",
    }


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    lost = [int(value) for value in re.findall(r"(\d+) Frames set to lost", text)]
    return {
        "fail_to_track_local_map_count": text.count("Fail to track local map!"),
        "track_ref_kf_less_than_15_count": text.count("TRACK_REF_KF: Less than 15 matches!!"),
        "active_map_reset_count": text.count("SYSTEM-> Reseting active map"),
        "new_map_count": text.count("Creation of new map"),
        "stored_map_count": text.count("Stored map with ID"),
        "lost_frame_count_reported": sum(lost),
    }


def trace_ratios(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    state_names = {
        "-1": "SYSTEM_NOT_READY",
        "0": "NO_IMAGES_YET",
        "1": "NOT_INITIALIZED",
        "2": "OK",
        "3": "RECENTLY_LOST",
        "4": "LOST",
        "5": "OK_KLT",
    }
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        next(handle, None)
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                state = parts[2].strip()
                counts[state_names.get(state, state)] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {
        "trace_ok_ratio": (counts["OK"] + counts["OK_KLT"]) / total,
        "trace_recently_lost_ratio": counts["RECENTLY_LOST"] / total,
        "trace_lost_ratio": counts["LOST"] / total,
        "trace_not_initialized_ratio": counts["NOT_INITIALIZED"] / total,
    }


def evo_stats_from_zip(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with zipfile.ZipFile(path) as archive:
        stats = json.loads(archive.read("stats.json").decode("utf-8"))
        errors = np.load(io.BytesIO(archive.read("error_array.npy")))
        timestamps = np.load(io.BytesIO(archive.read("timestamps.npy")))
    stats["p90"] = float(np.percentile(errors, 90)) if len(errors) else ""
    stats["p95"] = float(np.percentile(errors, 95)) if len(errors) else ""
    stats["matched_pose_count"] = int(len(timestamps))
    return stats


def keyframe_stats(split_dir: Path, result_dir: Path) -> dict[str, Any]:
    gt_ts, _, _ = load_tum(split_dir / "groundtruth_tum.txt")
    kf_ts, _, _ = load_tum(result_dir / "KeyFrameTrajectory.txt")
    if len(kf_ts) == 0:
        return {"keyframe_count": 0, "keyframe_time_coverage": ""}
    gt_duration = float(gt_ts[-1] - gt_ts[0]) if len(gt_ts) >= 2 else 0.0
    kf_duration = float(kf_ts[-1] - kf_ts[0]) if len(kf_ts) >= 2 else 0.0
    return {
        "keyframe_count": int(len(kf_ts)),
        "keyframe_time_coverage": kf_duration / gt_duration if gt_duration > 0 else "",
    }


def classify(row: dict[str, Any], metrics: dict[str, Any], status: dict[str, Any]) -> str:
    if truthy(status.get("timed_out")) or truthy(metrics.get("orbslam_timed_out")):
        return "timeout"
    exit_code = as_float(metrics.get("orbslam_exit_code", status.get("exit_code")))
    if exit_code is not None and exit_code != 0:
        return "runner_crash"
    result_dir = Path(row["result_dir"])
    if not (result_dir / "CameraTrajectory.txt").is_file():
        return "missing_trajectory"
    evaluation_success = metrics.get("evaluation_success", status.get("evo_success"))
    has_evaluation = (result_dir / "evo_ape.zip").is_file() or "evaluation_success" in metrics
    if not evaluation_success or not has_evaluation:
        return "empty_or_unmatched_trajectory"
    matched = as_float(row.get("matched_pose_ratio"))
    if matched is not None and matched < 0.5:
        return "low_coverage"
    ok_ratio = as_float(row.get("trace_ok_ratio"))
    if ok_ratio is not None and ok_ratio < 0.25:
        return "tracking_collapse"
    if as_float(row.get("active_map_reset_count")) and as_float(row.get("active_map_reset_count")) > 0:
        return "map_reset"
    if as_float(row.get("stored_map_count")) and as_float(row.get("stored_map_count")) > 0:
        return "multi_map_instability"
    if as_float(row.get("new_map_count")) and as_float(row.get("new_map_count")) > 2:
        return "multi_map_instability"
    ate = as_float(row.get("ate_rmse_m"))
    if ate is not None and ate >= 20.0:
        return "severe_drift"
    if ate is not None and ate >= 5.0:
        return "drift"
    return "usable"


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def truthy(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def metric_value(stats: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in stats and stats[key] not in (None, ""):
            return stats[key]
    return ""


def build_row(split_dir: Path, mode: str, run_id: str, columns: list[str]) -> dict[str, Any]:
    manifest = load_json(split_dir / "manifest.json")
    out_dir = result_dir(split_dir, mode, run_id)
    metrics = load_json(out_dir / "metrics.json")
    status = load_json(out_dir / "benchmark_status.json")
    row: dict[str, Any] = {column: "" for column in columns}
    row.update(
        {
            "scene_id": manifest.get("scene_id", split_dir.parent.name),
            "split_idx": manifest.get("split_idx", split_index(split_dir)),
            "split_dir": str(split_dir),
            "result_dir": str(out_dir),
            "frame_count": manifest.get("frame_count", ""),
            "fps": manifest.get("fps", ""),
            "mode": mode,
            "visual_sampled": "false",
        }
    )
    row.update(motion_stats(split_dir))
    row.update(parse_log(out_dir / MODES[mode].log_name))
    row.update(trace_ratios(out_dir / "tracking_trace.csv"))
    row.update(keyframe_stats(split_dir, out_dir))

    gt_ts, gt_pos, _ = load_tum(split_dir / "groundtruth_tum.txt")
    est_ts, est_pos, _ = load_tum(out_dir / "CameraTrajectory.txt")
    ape = metrics.get("ate") if isinstance(metrics.get("ate"), dict) else evo_stats_from_zip(out_dir / "evo_ape.zip")
    rpe = metrics.get("rpe") if isinstance(metrics.get("rpe"), dict) else {}
    rpe_trans = metrics.get("rpe_trans") if isinstance(metrics.get("rpe_trans"), dict) else evo_stats_from_zip(out_dir / "evo_rpe_trans.zip")
    rpe_rot = metrics.get("rpe_rot") if isinstance(metrics.get("rpe_rot"), dict) else evo_stats_from_zip(out_dir / "evo_rpe_rot.zip")
    matched_pose_count = metrics.get("matched_pose_count", ape.get("matched_pose_count"))
    matched_pose_ratio = metrics.get("matched_pose_ratio", "")
    if matched_pose_ratio == "" and matched_pose_count not in (None, "") and len(gt_ts):
        matched_pose_ratio = float(matched_pose_count) / len(gt_ts)
    aligned_path_ratio = metrics.get("raw_estimate_to_groundtruth_path_ratio", "")
    if aligned_path_ratio == "" and len(gt_pos) and len(est_pos):
        gt_path = path_length(gt_pos)
        aligned_path_ratio = path_length(est_pos) / gt_path if gt_path > 0 else ""
    success = metrics.get("success", status.get("success", status.get("status") == "passed"))
    row.update(
        {
            "success": str(truthy(success)).lower(),
            "ate_rmse_m": metric_value(ape, "rmse", "rmse_m"),
            "ate_median_m": metric_value(ape, "median", "median_m"),
            "ate_p90_m": metric_value(ape, "p90", "p90_m"),
            "ate_p95_m": metric_value(ape, "p95", "p95_m"),
            "ate_max_m": metric_value(ape, "max", "max_m"),
            "rpe_trans_rmse_m": metric_value(rpe_trans, "rmse", "trans_rmse_m") or metric_value(rpe, "trans_rmse_m"),
            "rpe_trans_p95_m": metric_value(rpe_trans, "p95", "trans_p95_m") or metric_value(rpe, "trans_p95_m"),
            "rpe_trans_max_m": metric_value(rpe_trans, "max", "trans_max_m") or metric_value(rpe, "trans_max_m"),
            "rpe_rot_rmse_deg": metric_value(rpe_rot, "rmse", "rot_rmse_deg") or metric_value(rpe, "rot_rmse_deg"),
            "rpe_rot_p95_deg": metric_value(rpe_rot, "p95", "rot_p95_deg") or metric_value(rpe, "rot_p95_deg"),
            "rpe_rot_max_deg": metric_value(rpe_rot, "max", "rot_max_deg") or metric_value(rpe, "rot_max_deg"),
            "matched_pose_ratio": matched_pose_ratio,
            "aligned_path_ratio": aligned_path_ratio,
        }
    )
    row["failure_category"] = classify(row, metrics, status)
    if row["failure_category"] != "usable":
        row["success"] = "false"
    return row


def main() -> None:
    args = parse_args()
    columns = load_columns(args.columns)
    splits = discover_prepared_splits(
        args.output_root,
        split_list=args.split_list,
        scene_ids=args.scene_id,
        split_indices=args.split_idx,
    )
    rows = [
        build_row(split_dir, mode, args.run_id, columns)
        for split_dir in splits
        for mode in args.modes
    ]
    rows.sort(key=lambda row: (str(row["scene_id"]), int(row["split_idx"]), str(row["mode"])))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
