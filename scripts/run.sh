#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-${REPO_ROOT}/.conda/envs/omni-orbslam}"

cd "${REPO_ROOT}"

export CONDA_PREFIX="${ENV_PREFIX}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${REPO_ROOT}/third_party/install/pangolin/lib:${REPO_ROOT}/third_party/ORB_SLAM3/lib:${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
JOBS="${JOBS:-8}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
MODES="${MODES:-mono rgbd vo}"
OUTPUT_CSV="${OUTPUT_CSV:-analysis/benchmarks/${RUN_ID}.csv}"

read -r -a MODE_ARGS <<< "${MODES}"
EXTRA_ARGS=()
if [[ -n "${SCENE_ID:-}" ]]; then
  EXTRA_ARGS+=(--scene-id "${SCENE_ID}")
fi
if [[ -n "${SPLIT_IDX:-}" ]]; then
  EXTRA_ARGS+=(--split-idx "${SPLIT_IDX}")
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi

echo "run_id=${RUN_ID} modes=${MODES} jobs=${JOBS}"
"${ENV_PREFIX}/bin/python" scripts/run_benchmark.py \
  --run-id "${RUN_ID}" \
  --modes "${MODE_ARGS[@]}" \
  --jobs "${JOBS}" \
  --timeout-sec "${TIMEOUT_SEC}" \
  --progress-interval "${PROGRESS_INTERVAL}" \
  "${EXTRA_ARGS[@]}"

"${ENV_PREFIX}/bin/python" scripts/build_metrics_csv.py \
  --run-id "${RUN_ID}" \
  --modes "${MODE_ARGS[@]}" \
  --output-csv "${OUTPUT_CSV}" \
  "${EXTRA_ARGS[@]}"
