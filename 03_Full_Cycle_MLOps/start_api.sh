#!/usr/bin/env bash
# Step 6: serve the champion model. Run this after train_model.py has finished.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_venv

if ! curl -fsS -o /dev/null http://127.0.0.1:5000/health 2>/dev/null; then
    echo "Error: MLflow isn't running. Run ./start_services.sh first." >&2
    exit 1
fi

# If the API is already running it keeps its old model; restart it to pick up a new champion.
stop_bg api
echo "Starting API ..."
start_bg api uvicorn fastapi_app:app --host 127.0.0.1 --port 8000

if wait_for_http api http://127.0.0.1:8000/health 90; then
    curl -s http://127.0.0.1:8000/health; echo
else
    echo "---- last lines of logs/api.log ----" >&2
    tail -n 15 "$LOG_DIR/api.log" >&2
    exit 1
fi
