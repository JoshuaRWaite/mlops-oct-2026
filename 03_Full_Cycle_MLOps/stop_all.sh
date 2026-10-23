#!/usr/bin/env bash
# Stop every service started by start_services.sh and start_api.sh.
# Your data (mlflow.db, mlflow-artifacts/, processed_images/) is kept.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Stopping services ..."
for name in api streamlit grafana prometheus mlflow; do
    stop_bg "$name"
done
echo "Done."
