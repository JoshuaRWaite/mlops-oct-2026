#!/usr/bin/env bash
# Restart module 3's API so it loads whichever model is champion now (module 3's API reads the champion at startup).
# Module 3's own start_api.sh stops the old API and starts the new one immediately; the old one can still be holding
# port 8000 for a moment, so here we first make sure it has really stopped, then call start_api.sh.
set -euo pipefail
M3="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../03_Full_Cycle_MLOps" && pwd)"
cd "$M3"
# shellcheck disable=SC1091
source ./common.sh                       # module 3's helpers (stop_bg ...)
stop_bg api
for _ in $(seq 30); do
    curl -fsS -o /dev/null http://127.0.0.1:8000/health 2>/dev/null || break   # wait until it stops answering
    sleep 1
done
sleep 1
./start_api.sh
