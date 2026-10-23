#!/usr/bin/env bash
# Step 1b: download Prometheus and Grafana into ./tools (no sudo, nothing installed system-wide).
# Override versions with:  PROM_VERSION=3.13.5 GRAFANA_VERSION=13.2.4 ./install_monitoring.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PROM_VERSION="${PROM_VERSION:-3.13.5}"        # a Prometheus long-term-support release
GRAFANA_VERSION="${GRAFANA_VERSION:-13.2.4}"

case "$(uname -m)" in
    x86_64)  ARCH=amd64 ;;
    aarch64) ARCH=arm64 ;;
    *) echo "Unsupported CPU: $(uname -m)" >&2; exit 1 ;;
esac

mkdir -p "$TOOLS_DIR"

fetch() {  # fetch <name> <url> <dest-dir>
    local name=$1 url=$2 dest=$3 tarball
    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "$name already installed in tools/$(basename "$dest"). Delete that folder to reinstall."
        return 0
    fi
    echo "Downloading $name ..."
    tarball="$(mktemp)"
    if ! curl -fL --retry 3 -o "$tarball" "$url"; then
        rm -f "$tarball"
        echo "Error: could not download $url" >&2
        return 1
    fi
    mkdir -p "$dest"
    tar -xzf "$tarball" -C "$dest" --strip-components=1
    rm -f "$tarball"
}

fetch "Prometheus $PROM_VERSION" \
    "https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/prometheus-${PROM_VERSION}.linux-${ARCH}.tar.gz" \
    "$TOOLS_DIR/prometheus"

fetch "Grafana $GRAFANA_VERSION" \
    "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-${ARCH}.tar.gz" \
    "$TOOLS_DIR/grafana"

echo
"$TOOLS_DIR/prometheus/prometheus" --version 2>&1 | head -n 1
"$TOOLS_DIR/grafana/bin/grafana" --version 2>&1 | head -n 1 || true
echo "Done. Next: ./start_services.sh"
