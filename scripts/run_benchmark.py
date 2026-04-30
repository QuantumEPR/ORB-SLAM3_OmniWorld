#!/usr/bin/env python3
"""Run ORB-SLAM3 on prepared OmniWorld splits and save evo evaluations."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils import (
    DEFAULT_OUTPUT_ROOT,
    MODE_NAMES,
    MODES,
    ORB_SLAM3_DIR,
    REPO_ROOT,
    Mode,
    apply_thread_env,
    discover_prepared_splits,
    load_json,
    result_dir,
    split_index,
)


apply_thread_env()


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def orbslam_command(mode: Mode, split_dir: Path, out_dir: Path) -> list[str]:
    sequence_dir = split_dir / "sequence"
    cmd = [
        str(mode.binary),
        str(ORB_SLAM3_DIR / "Vocabulary" / "ORBvoc.txt"),
        str(split_dir / "omniworld_rgbd.yaml"),
        str(sequence_dir),
        str(sequence_dir / "associations.txt"),
        str(out_dir / "CameraTrajectory.txt"),
        str(out_dir / "KeyFrameTrajectory.txt"),
        str(out_dir / "runtime.json"),
    ]
    if mode.trace_csv:
        cmd.append(str(out_dir / "tracking_trace.csv"))
    return cmd


def evo_commands(
    max_timestamp_diff: float,
    mode_config: Mode,
    split_dir: Path,
    out_dir: Path,
) -> dict[str, list[str]]:
    gt = split_dir / "groundtruth_tum.txt"
    est = out_dir / "CameraTrajectory.txt"
    align = ["-a", "-s"] if mode_config.sim3 else ["-a"]
    common = [
        "tum",
        str(gt),
        str(est),
        *align,
        "--t_max_diff",
        str(max_timestamp_diff),
        "--no_warnings",
    ]
    return {
        "ape": [
            tool_path("evo_ape"),
            *common,
            "--save_results",
            str(out_dir / "evo_ape.zip"),
        ],
        "rpe_trans": [
            tool_path("evo_rpe"),
            *common,
            "-r",
            "trans_part",
            "--save_results",
            str(out_dir / "evo_rpe_trans.zip"),
        ],
        "rpe_rot": [
            tool_path("evo_rpe"),
            *common,
            "-r",
            "angle_deg",
            "--save_results",
            str(out_dir / "evo_rpe_rot.zip"),
        ],
    }


def tool_path(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    alongside_python = Path(sys.executable).resolve().parent / name
    return str(alongside_python if alongside_python.exists() else name)


def run_logged(cmd: list[str], log_path: Path, timeout_sec: int | None = None) -> tuple[int, bool, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        try:
            completed = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
            return completed.returncode, False, time.monotonic() - start
        except subprocess.TimeoutExpired:
            log.write(f"\nTimed out after {timeout_sec} seconds.\n")
            return 124, True, time.monotonic() - start


def run_evo(
    max_timestamp_diff: float,
    mode_config: Mode,
    split_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    trajectory = out_dir / "CameraTrajectory.txt"
    if not trajectory.is_file():
        return {"evo_success": False, "evo_failure": "missing_trajectory"}
    if not trajectory.read_text(encoding="utf-8", errors="replace").strip():
        return {"evo_success": False, "evo_failure": "empty_trajectory"}

    exit_codes: dict[str, int] = {}
    with (out_dir / "evo.log").open("w", encoding="utf-8", errors="replace") as log:
        for name, cmd in evo_commands(max_timestamp_diff, mode_config, split_dir, out_dir).items():
            log.write("$ " + " ".join(cmd) + "\n")
            completed = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            exit_codes[name] = completed.returncode
            log.write(f"\nexit_code={completed.returncode}\n\n")
            log.flush()

    return {
        "evo_exit_codes": exit_codes,
        "evo_success": all(code == 0 for code in exit_codes.values()),
    }


def result_complete(out_dir: Path, status: dict[str, Any]) -> bool:
    return (
        status.get("status") == "passed"
        and status.get("success") is True
        and (out_dir / "CameraTrajectory.txt").is_file()
        and (out_dir / "KeyFrameTrajectory.txt").is_file()
        and (out_dir / "runtime.json").is_file()
        and (out_dir / "evo_ape.zip").is_file()
        and (out_dir / "evo_rpe_trans.zip").is_file()
        and (out_dir / "evo_rpe_rot.zip").is_file()
    )


def clear_previous_outputs(out_dir: Path, mode_config: Mode) -> None:
    for name in (
        "CameraTrajectory.txt",
        "KeyFrameTrajectory.txt",
        "runtime.json",
        "tracking_trace.csv",
        "evo.log",
        "evo_ape.zip",
        "evo_rpe_trans.zip",
        "evo_rpe_rot.zip",
        "metrics.json",
        mode_config.log_name,
    ):
        try:
            (out_dir / name).unlink()
        except FileNotFoundError:
            pass


def job_status(exit_code: int, timed_out: bool, evo_success: bool, evo_failure: str | None = None) -> str:
    if timed_out:
        return "timed_out"
    if exit_code != 0:
        return "orbslam_failed"
    if evo_failure in {"missing_trajectory", "empty_trajectory"}:
        return evo_failure
    if not evo_success:
        return "evo_failed"
    return "passed"


def run_job(args: argparse.Namespace, split_dir: Path, mode_name: str) -> dict[str, Any]:
    mode_config = MODES[mode_name]
    out_dir = result_dir(split_dir, mode_name, args.run_id)
    status_path = out_dir / "benchmark_status.json"
    old_status = load_json(status_path)
    if old_status and not args.force and result_complete(out_dir, old_status):
        return {**old_status, "status": "skipped_complete"}

    command = orbslam_command(mode_config, split_dir, out_dir)
    if args.dry_run:
        print("[dry-run] " + " ".join(command), flush=True)
        return job_identity(split_dir, mode_name, out_dir) | {"status": "dry_run", "success": True}

    out_dir.mkdir(parents=True, exist_ok=True)
    clear_previous_outputs(out_dir, mode_config)
    status = job_identity(split_dir, mode_name, out_dir) | {
        "run_id": args.run_id,
        "started_at": now(),
        "orbslam_command": command,
    }
    exit_code, timed_out, wall_time = run_logged(
        command,
        out_dir / mode_config.log_name,
        timeout_sec=args.timeout_sec,
    )
    status.update(
        {
            "ended_at": now(),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "wall_time_sec": wall_time,
        }
    )

    if exit_code == 0 and not timed_out:
        status.update(run_evo(args.max_timestamp_diff, mode_config, split_dir, out_dir))
    else:
        status.update({"evo_success": False, "evo_exit_codes": {}})

    status["status"] = job_status(
        exit_code,
        timed_out,
        status["evo_success"],
        status.get("evo_failure"),
    )
    status["success"] = status["status"] == "passed"

    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def job_identity(split_dir: Path, mode: str, out_dir: Path) -> dict[str, Any]:
    return {
        "scene_id": split_dir.parent.name,
        "split_idx": split_index(split_dir),
        "split_dir": str(split_dir),
        "mode": mode,
        "result_dir": str(out_dir),
    }


def run_jobs(args: argparse.Namespace, jobs: list[tuple[Path, str]]) -> list[dict[str, Any]]:
    if args.jobs == 1:
        results = []
        for index, (split_dir, mode) in enumerate(jobs, start=1):
            print(f"[{index}/{len(jobs)}] {split_dir.parent.name}/{split_dir.name} {mode}", flush=True)
            results.append(run_job(args, split_dir, mode))
        return results

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        future_map = {
            pool.submit(run_job, args, split_dir, mode): (split_dir, mode)
            for split_dir, mode in jobs
        }
        while future_map:
            done, _ = wait(
                future_map,
                timeout=args.progress_interval,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                print(f"progress done={len(results)}/{len(jobs)} in_flight={len(future_map)}", flush=True)
                continue
            for future in done:
                split_dir, mode = future_map.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = job_identity(split_dir, mode, result_dir(split_dir, mode, args.run_id)) | {
                        "status": "exception",
                        "success": False,
                        "error": str(exc),
                    }
                results.append(result)
                print(
                    f"[{len(results)}/{len(jobs)}] {result['scene_id']}/split_{int(result['split_idx']):02d} "
                    f"{result['mode']} {result['status']}",
                    flush=True,
                )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-list", type=Path, default=None)
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument("--split-idx", type=int, action="append", default=[])
    parser.add_argument("--modes", nargs="+", choices=MODE_NAMES, default=MODE_NAMES)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--max-timestamp-diff", type=float, default=0.02)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = discover_prepared_splits(
        args.output_root,
        split_list=args.split_list,
        scene_ids=args.scene_id,
        split_indices=args.split_idx,
    )
    if not splits:
        raise SystemExit("No prepared splits found. Run scripts/prepare_data.py first.")

    jobs = [(split_dir, mode) for split_dir in splits for mode in args.modes]
    print(
        f"run_id={args.run_id} splits={len(splits)} modes={','.join(args.modes)} "
        f"jobs={len(jobs)} parallel={args.jobs}",
        flush=True,
    )

    results = run_jobs(args, jobs)
    exceptions = sum(row["status"] == "exception" for row in results)
    benchmark_failures = sum(row.get("success") is False for row in results)
    print(
        f"Finished run_id={args.run_id}: jobs={len(results)} "
        f"benchmark_failures={benchmark_failures} exceptions={exceptions}",
        flush=True,
    )
    raise SystemExit(1 if exceptions else 0)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    main()
