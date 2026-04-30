# ORB-SLAM3 on OmniWorld

The active benchmark path is intentionally small:

- `scripts/prepare_data.py`: prepare OmniWorld scene splits into ORB-SLAM3 RGB-D sequences.
- `scripts/run_benchmark.py`: run ORB-SLAM3 modes in parallel and evaluate each trajectory with `evo`.
- `scripts/build_metrics_csv.py`: build one long-form CSV using the schema in `columns.txt`.

## Prepare Data

Prepare every locally available OmniWorld scene/split:

```bash
python scripts/prepare_data.py --skip-existing
```

Prepare one split:

```bash
python scripts/prepare_data.py \
  --scene-id 63ad1dbede39 \
  --split-idx 0 \
  --skip-existing
```

Prepare a fixed list:

```bash
python scripts/prepare_data.py --split-list splits.txt --skip-existing
```

`splits.txt` may contain `scene_id`, `scene_id/split_03`, or `scene_id,split_03`.

## Run Benchmark

Run all prepared splits for all three modes, with bounded parallelism:

```bash
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/run_benchmark.py \
  --run-id example_run \
  --modes mono rgbd vo \
  --jobs 8 \
  --progress-interval 30
```

Per-job outputs are written under:

```text
runs/omniworld_prepared/<scene>/split_XX/results/orbslam3/<mode>/<run_id>/
```

Each result directory contains ORB-SLAM3 logs/trajectories, `evo_ape.zip`,
`evo_rpe_trans.zip`, `evo_rpe_rot.zip`, `metrics.json`, and
`benchmark_status.json`.

Dry-run one split:

```bash
python scripts/run_benchmark.py \
  --dry-run \
  --run-id example_run \
  --scene-id 63ad1dbede39 \
  --split-idx 0 \
  --modes rgbd
```

## Build CSV

```bash
python scripts/build_metrics_csv.py \
  --run-id example_run \
  --output-csv analysis/benchmarks/example_run.csv
```

The CSV has one row per split/mode and uses the exact quoted column names from
`columns.txt`, including `mode` as a row label rather than separate mono/RGB-D/VO
metric columns.
