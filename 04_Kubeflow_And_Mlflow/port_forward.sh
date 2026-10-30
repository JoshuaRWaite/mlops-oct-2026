#!/usr/bin/env bash
# Make the Kubeflow Pipelines web page reachable on this VM's port 8080.
#   ./port_forward.sh start | stop | status
# It only listens on 127.0.0.1, so reach it from your laptop through SSH port forwarding (VS Code: Ports panel).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need kubectl

PIDFILE="$RUN_DIR/portforward-kfp.pid"
running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-status}" in
    start)
        need_cluster
        if running; then echo "Kubeflow Pipelines: already on http://localhost:8080"; exit 0; fi
        setsid nohup kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8080:80 >"$LOG_DIR/portforward-kfp.log" 2>&1 &
        echo $! > "$PIDFILE"
        sleep 3
        if running; then echo "Kubeflow Pipelines: http://localhost:8080"
        else echo "Port-forward failed:"; cat "$LOG_DIR/portforward-kfp.log"; exit 1; fi ;;
    stop)
        if running; then kill "$(cat "$PIDFILE")" && echo "Stopped."; else echo "Not running."; fi
        rm -f "$PIDFILE" ;;
    status)
        if running; then echo "Running on http://localhost:8080"; else echo "Not running."; fi ;;
    *) echo "Usage: $0 start|stop|status" >&2; exit 1 ;;
esac
