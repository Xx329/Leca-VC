#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
SOURCE="$REPO_ROOT/physicell/fibrosis_application"
CONFIG="${2:-configs/local_paths.yaml}"
[[ "${1:-}" == "--config" ]] || { echo "usage: $0 --config FILE"; exit 2; }
eval "$(python - "$CONFIG" <<'PY'
import sys,yaml,shlex
d=yaml.safe_load(open(sys.argv[1])) or {}
for k,e in [('physicell_root','PHYSICELL_ROOT'),('physicell_binary','PHYSICELL_BINARY')]: print(f'export {e}={shlex.quote(str(d.get(k) or ""))}')
PY
)"
command -v g++ >/dev/null; command -v make >/dev/null
test -f "$PHYSICELL_ROOT/core/PhysiCell.h"
test -f "$SOURCE/custom.cpp"; test -f "$SOURCE/custom.h"; test -f "$SOURCE/PhysiCell_settings_base.xml"
test -z "$PHYSICELL_BINARY" || test -f "$PHYSICELL_BINARY"
echo "PhysiCell source/config checks passed"
