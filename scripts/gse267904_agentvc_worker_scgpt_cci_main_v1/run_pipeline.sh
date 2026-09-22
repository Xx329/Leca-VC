#!/usr/bin/env bash
set -euo pipefail

ROOT="${LECAVC_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
SCRIPT="$ROOT/scripts/gse267904_agentvc_worker_scgpt_cci_main_v1"
PY="${COMMOT_PYTHON:?Set COMMOT_PYTHON to the COMMOT environment interpreter}"

export NUMBA_CACHE_DIR=/tmp/agentvc_worker_commot_numba_cache
export MPLCONFIGDIR=/tmp/agentvc_worker_matplotlib
export OMP_NUM_THREADS=1

cd "$ROOT"
if [[ ! -f "$ROOT/outputs/GSE267904_agentvc_worker_scgpt_cci_main_v1/00_protocol/frozen_worker_cci_protocol_v1.yaml" ]]; then
  "$PY" "$SCRIPT/prepare_workers.py" audit-freeze
fi
"$PY" "$SCRIPT/prepare_workers.py" registry
"$PY" "$SCRIPT/prepare_workers.py" smoke
"$PY" "$SCRIPT/benchmark.py" smoke
"$PY" "$SCRIPT/prepare_workers.py" formal
"$PY" "$SCRIPT/benchmark.py" prepare-inputs
"$PY" "$SCRIPT/benchmark.py" commot
"$PY" "$SCRIPT/benchmark.py" evaluate
"$PY" "$SCRIPT/benchmark.py" plot
"$PY" "$SCRIPT/benchmark.py" validate
