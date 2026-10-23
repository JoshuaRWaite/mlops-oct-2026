#!/usr/bin/env bash
# Step 2: start MLflow, Prometheus, Grafana and the Streamlit upload page.
# Everything listens on 127.0.0.1 only; reach it from your laptop with an SSH tunnel (see README).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_venv

echo "Starting services ..."

# MLflow: experiment tracking + model registry. Data lives in ./mlflow.db and ./mlflow-artifacts.
start_bg mlflow mlflow server \
    --backend-store-uri "sqlite:///$ROOT/mlflow.db" \
    --artifacts-destination "$ROOT/mlflow-artifacts" \
    --host 127.0.0.1 --port 5000

# Prometheus: scrapes the API's /metrics endpoint (prometheus.yml says where).
if [ -x "$TOOLS_DIR/prometheus/prometheus" ]; then
    start_bg prometheus "$TOOLS_DIR/prometheus/prometheus" \
        --config.file="$ROOT/prometheus.yml" \
        --storage.tsdb.path="$RUN_DIR/prometheus-data" \
        --web.listen-address=127.0.0.1:9090
else
    echo "  Skipping Prometheus: run ./install_monitoring.sh first."
fi

# Grafana: dashboards. The Prometheus data source and the demo dashboard are provisioned from
# monitoring/grafana/, so there is nothing to click through. Settings are passed as GF_* variables.
if [ -x "$TOOLS_DIR/grafana/bin/grafana" ]; then
    start_bg grafana env \
        MLOPS_DASHBOARD_DIR="$ROOT/monitoring/grafana/dashboards" \
        GF_PATHS_PROVISIONING="$ROOT/monitoring/grafana/provisioning" \
        GF_PATHS_DATA="$RUN_DIR/grafana-data" \
        GF_PATHS_LOGS="$LOG_DIR/grafana" \
        GF_PATHS_PLUGINS="$RUN_DIR/grafana-plugins" \
        GF_SERVER_HTTP_ADDR=127.0.0.1 \
        GF_SERVER_HTTP_PORT=3000 \
        GF_AUTH_ANONYMOUS_ENABLED=true \
        GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
        GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH="$ROOT/monitoring/grafana/dashboards/mlops-demo.json" \
        GF_ANALYTICS_REPORTING_ENABLED=false \
        GF_ANALYTICS_CHECK_FOR_UPDATES=false \
        "$TOOLS_DIR/grafana/bin/grafana" server --homepath "$TOOLS_DIR/grafana"
else
    echo "  Skipping Grafana: run ./install_monitoring.sh first."
fi

# Streamlit: the zip upload page (settings in .streamlit/config.toml).
start_bg streamlit streamlit run streamlit_app.py

echo
echo "Waiting for services to answer ..."
wait_for_http mlflow http://127.0.0.1:5000/health || true
wait_for_http streamlit http://127.0.0.1:8501 || true
is_running prometheus && { wait_for_http prometheus http://127.0.0.1:9090/-/ready || true; }
is_running grafana && { wait_for_http grafana http://127.0.0.1:3000/api/health 60 || true; }

echo
echo "Services (on this machine):"
echo "  Streamlit upload  http://localhost:8501"
echo "  MLflow            http://localhost:5000"
is_running prometheus && echo "  Prometheus        http://localhost:9090"
is_running grafana && echo "  Grafana           http://localhost:3000   (viewers are anonymous; admin login is admin / admin)"
echo "To open these in your laptop's browser, use the SSH tunnel command from the README."
