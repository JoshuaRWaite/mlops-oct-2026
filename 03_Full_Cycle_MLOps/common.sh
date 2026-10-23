#!/usr/bin/env bash
# Shared helpers, sourced by the other scripts (not meant to be run directly).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$ROOT/.run"     # PID files and service data (Prometheus, Grafana)
LOG_DIR="$ROOT/logs"     # one log file per service
TOOLS_DIR="$ROOT/tools"  # Prometheus and Grafana binaries (see install_monitoring.sh)
mkdir -p "$RUN_DIR" "$LOG_DIR"
cd "$ROOT"

activate_venv() {
    if [ ! -f "$ROOT/.venv/bin/activate" ]; then
        echo "Error: no virtual environment yet. Run ./setup_env.sh first." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
}

is_running() {  # is_running <name>
    [ -f "$RUN_DIR/$1.pid" ] && kill -0 "$(cat "$RUN_DIR/$1.pid")" 2>/dev/null
}

start_bg() {  # start_bg <name> <command...>   -> runs detached, logs to logs/<name>.log
    local name=$1; shift
    if is_running "$name"; then
        echo "  $name is already running (pid $(cat "$RUN_DIR/$name.pid"))"
        return 0
    fi
    # setsid gives the service its own process group so stop_bg can stop its children too.
    setsid nohup "$@" >"$LOG_DIR/$name.log" 2>&1 &
    echo $! >"$RUN_DIR/$name.pid"
    echo "  started $name (pid $!, log: logs/$name.log)"
}

stop_bg() {  # stop_bg <name>
    local name=$1
    if is_running "$name"; then
        local pid; pid=$(cat "$RUN_DIR/$name.pid")
        kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
        echo "  stopped $name"
    fi
    rm -f "$RUN_DIR/$name.pid"
}

wait_for_http() {  # wait_for_http <name> <url> [seconds]
    local name=$1 url=$2 tries=${3:-30}
    for _ in $(seq "$tries"); do
        if curl -fsS -o /dev/null "$url" 2>/dev/null; then
            echo "  $name is up: $url"
            return 0
        fi
        sleep 1
    done
    echo "  Warning: $name did not answer at $url within ${tries}s. See logs/$name.log" >&2
    return 1
}
