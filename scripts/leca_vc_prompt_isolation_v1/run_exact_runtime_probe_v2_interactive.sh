#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${LECAVC_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="$PROJECT_ROOT/scripts/leca_vc_prompt_isolation_v1/run_exact_runtime_probe_v2.py"

cd "$PROJECT_ROOT"
export MPLCONFIGDIR="/tmp/lecavc_exact_runtime_probe_v2_matplotlib"

"$PYTHON_BIN" "$SCRIPT" --preflight-only

read -r -s -p "DeepSeek API key: " DEEPSEEK_API_KEY
printf '\n'
if [[ -z "$DEEPSEEK_API_KEY" ]]; then
  echo "API key is empty; no model call was started." >&2
  exit 2
fi
export DEEPSEEK_API_KEY
trap 'unset DEEPSEEK_API_KEY' EXIT INT TERM

"$PYTHON_BIN" "$SCRIPT"
