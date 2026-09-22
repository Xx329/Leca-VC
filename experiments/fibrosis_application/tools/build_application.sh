#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
SOURCE="$REPO_ROOT/physicell/fibrosis_application"
CONFIG="${2:-configs/local_paths.yaml}"
[[ "${1:-}" == "--config" ]] || { echo "usage: $0 --config FILE"; exit 2; }
PHYSICELL_ROOT="$(python -c 'import sys,yaml; print((yaml.safe_load(open(sys.argv[1])) or {}).get("physicell_root") or "")' "$CONFIG")"
test -f "$PHYSICELL_ROOT/core/PhysiCell.h"
mkdir -p "$REPO_ROOT/build/fibrosis_application"
cp "$SOURCE/custom.cpp" "$SOURCE/custom.h" "$SOURCE/Makefile" "$REPO_ROOT/build/fibrosis_application/"
make -C "$REPO_ROOT/build/fibrosis_application" PHYSICELL_ROOT="$PHYSICELL_ROOT" fibrosis_application_v3
echo "Built $REPO_ROOT/build/fibrosis_application/fibrosis_application_v3 without modifying PhysiCell installation"
