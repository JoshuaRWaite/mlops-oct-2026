#!/usr/bin/env bash
# Send drifted traffic (photos with a grid over them) to module 3's API, keeping a copy of each image.
#   ./send_drift.sh                 100 requests
#   ./send_drift.sh --n 200 --reset
# Uses module 3's Python environment. See lifecycle/send_drift.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M3="$HERE/../03_Full_Cycle_MLOps"
[ -f "$M3/.venv/bin/activate" ] || { echo "Error: finish module 3 first (cd ../03_Full_Cycle_MLOps && ./setup_env.sh)." >&2; exit 1; }
# shellcheck disable=SC1091
source "$M3/.venv/bin/activate"
exec python "$HERE/lifecycle/send_drift.py" "$@"
