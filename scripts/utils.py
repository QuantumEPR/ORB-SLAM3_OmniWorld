from __future__ import annotations

import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "omniworld_prepared"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "OmniWorld"
if not DEFAULT_DATASET_ROOT.exists():
    DEFAULT_DATASET_ROOT = Path("/data2/zhewenz/datasets/OmniWorld")
DEFAULT_METADATA_CSV = REPO_ROOT / "OmniWorld" / "omniworld_game_metadata.csv"
ORB_SLAM3_DIR = REPO_ROOT / "third_party" / "ORB-SLAM3"

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class Mode:
    binary: Path
    log_name: str
    trace_csv: bool = False
    sim3: bool = False


MODES = {
    "mono": Mode(
        ORB_SLAM3_DIR / "Examples" / "Monocular" / "mono_omniworld",
        "orbslam3_mono.log",
        trace_csv=True,
        sim3=True,
    ),
    "rgbd": Mode(
        ORB_SLAM3_DIR / "Examples" / "RGB-D" / "rgbd_omniworld",
        "orbslam3.log",
    ),
    "vo": Mode(
        ORB_SLAM3_DIR / "Examples" / "RGB-D" / "rgbd_omniworld_vo",
        "orbslam3_vo.log",
    ),
}
MODE_NAMES = list(MODES)


def apply_thread_env() -> None:
    for name, value in THREAD_ENV.items():
        os.environ.setdefault(name, value)


def split_index(split_dir: Path) -> int:
    return int(split_dir.name.removeprefix("split_"))


def result_dir(split_dir: Path, mode: str, run_id: str) -> Path:
    return split_dir / "results" / "orbslam3" / mode / run_id


def is_prepared_split(split_dir: Path) -> bool:
    return (
        split_dir.is_dir()
        and (split_dir / "manifest.json").is_file()
        and (split_dir / "groundtruth_tum.txt").is_file()
        and (split_dir / "omniworld_rgbd.yaml").is_file()
        and (split_dir / "sequence" / "associations.txt").is_file()
    )


def parse_split_name(value: str) -> int:
    return int(value.strip().removeprefix("split_"))


def scene_archive(dataset_root: Path, scene_id: str) -> Path:
    return (
        dataset_root
        / "annotations"
        / "OmniWorld-Game"
        / scene_id
        / f"{scene_id}_others.tar.gz"
    )


def available_scene_ids(dataset_root: Path) -> list[str]:
    root = dataset_root / "annotations" / "OmniWorld-Game"
    if not root.exists():
        return []
    return sorted(
        scene_dir.name
        for scene_dir in root.iterdir()
        if scene_dir.is_dir() and scene_archive(dataset_root, scene_dir.name).exists()
    )


def read_tar_json(tar_path: Path, member_name: str):
    with tarfile.open(tar_path, "r:gz") as tar:
        with tar.extractfile(member_name) as handle:
            return json.load(handle)


def read_tar_text(tar_path: Path, member_name: str) -> str:
    with tarfile.open(tar_path, "r:gz") as tar:
        with tar.extractfile(member_name) as handle:
            return handle.read().decode("utf-8")


def split_count(dataset_root: Path, scene_id: str) -> int:
    archive = scene_archive(dataset_root, scene_id)
    if not archive.exists():
        raise FileNotFoundError(f"Missing scene annotations archive: {archive}")
    return len(read_tar_json(archive, "split_info.json")["split"])


def read_scene_split_selection(path: Path) -> dict[str, set[int] | None]:
    selected: dict[str, set[int] | None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.split("#", 1)[0].strip()
        if not token:
            continue
        if "," in token:
            scene_id, split_name = [part.strip() for part in token.split(",", 1)]
        else:
            entry = Path(token)
            if not entry.name.startswith("split_"):
                selected[token] = None
                continue
            scene_id, split_name = entry.parent.name, entry.name
        selected.setdefault(scene_id, set())
        assert selected[scene_id] is not None
        selected[scene_id].add(parse_split_name(split_name))
    return selected


def selected_scene_splits(
    dataset_root: Path,
    scene_ids: Iterable[str] = (),
    split_indices: Iterable[int] = (),
    split_list: Path | None = None,
) -> list[tuple[str, int]]:
    selections: dict[str, set[int] | None] = {}
    if split_list is not None:
        selections.update(read_scene_split_selection(split_list))
    requested_splits = set(split_indices)
    for scene_id in scene_ids:
        selections.setdefault(scene_id, requested_splits if requested_splits else None)
    if not selections:
        selections = {scene_id: None for scene_id in available_scene_ids(dataset_root)}

    jobs = []
    for scene_id, selected_splits in sorted(selections.items()):
        indices = range(split_count(dataset_root, scene_id)) if selected_splits is None else sorted(selected_splits)
        jobs.extend((scene_id, split_idx) for split_idx in indices)
    return jobs


def split_paths_from_token(token: str, output_root: Path) -> list[Path]:
    if "," in token:
        scene_id, split_name = [part.strip() for part in token.split(",", 1)]
        return [output_root / scene_id / f"split_{parse_split_name(split_name):02d}"]

    path = Path(token)
    if path.name.startswith("split_"):
        return [path]

    scene_dir = path if path.is_absolute() else output_root / token
    return sorted(scene_dir.glob("split_*")) if scene_dir.is_dir() else [scene_dir]


def read_split_paths(path: Path, output_root: Path) -> list[Path]:
    splits: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            splits.extend(split_paths_from_token(token, output_root))
    return sorted(set(splits))


def discover_prepared_splits(
    output_root: Path,
    split_list: Path | None = None,
    scene_ids: Iterable[str] = (),
    split_indices: Iterable[int] = (),
) -> list[Path]:
    candidates = read_split_paths(split_list, output_root) if split_list else sorted(output_root.glob("*/split_*"))
    scene_filter = set(scene_ids)
    split_filter = set(split_indices)
    return [
        split_dir
        for split_dir in candidates
        if is_prepared_split(split_dir)
        and (not scene_filter or split_dir.parent.name in scene_filter)
        and (not split_filter or split_index(split_dir) in split_filter)
    ]


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
