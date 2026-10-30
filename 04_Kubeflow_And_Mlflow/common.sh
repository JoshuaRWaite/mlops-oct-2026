#!/usr/bin/env bash
# Shared helpers, sourced by the other scripts (not meant to be run directly).

# The variables below are used by the scripts that source this file.
# shellcheck disable=SC2034
M4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC2034
REPO_ROOT="$(cd "$M4_DIR/.." && pwd)"
# shellcheck disable=SC2034
SHARED_DIR="$REPO_ROOT/03_Full_Cycle_MLOps"     # module 3: the scripts our image reuses
# shellcheck disable=SC2034
RUN_DIR="$M4_DIR/.run"
# shellcheck disable=SC2034
LOG_DIR="$M4_DIR/logs"
# shellcheck disable=SC2034
MODE_FILE="$RUN_DIR/mode"                       # "gpu" or "cpu": written by 02_create_cluster.sh
# shellcheck disable=SC2034
CLUSTER_NAME="${CLUSTER_NAME:-kubeflow}"
# shellcheck disable=SC2034
IMAGE="${IMAGE:-house-autoencoder:module4}"
mkdir -p "$RUN_DIR" "$LOG_DIR"
cd "$M4_DIR" || exit 1

need() {  # need <command>
    command -v "$1" >/dev/null 2>&1 || { echo "Error: '$1' not found. Run ./01_setup_host.sh first." >&2; exit 1; }
}

need_docker_access() {
    if ! docker info >/dev/null 2>&1; then
        echo "Error: can't talk to Docker. If you just ran 01_setup_host.sh, log out and back in" >&2
        echo "(or run: newgrp docker) so the new 'docker' group takes effect." >&2
        exit 1
    fi
}

need_cluster() {
    if ! kubectl cluster-info >/dev/null 2>&1; then
        echo "Error: kubectl can't reach the cluster. Run ./02_create_cluster.sh first," >&2
        echo "or, if it exists: kind export kubeconfig --name $CLUSTER_NAME" >&2
        exit 1
    fi
}

need_module3() {  # module 4 reuses module 3's scripts, so the whole repo must be here
    local f
    for f in config.py model.py process_images.py train_model.py; do
        if [ ! -f "$SHARED_DIR/$f" ]; then
            echo "Error: $SHARED_DIR/$f is missing." >&2
            echo "Module 4 builds on module 3. Clone the whole repo, not just this folder." >&2
            exit 1
        fi
    done
}

need_module3_ready() {  # module 4 continues from a finished module 3
    need_module3
    if [ ! -f "$SHARED_DIR/.venv/bin/activate" ]; then
        echo "Error: module 3 isn't set up. Finish it first: cd ../03_Full_Cycle_MLOps && ./setup_env.sh" >&2
        exit 1
    fi
    if [ ! -d "$SHARED_DIR/uploaded_images" ] || [ ! -d "$SHARED_DIR/processed_images" ]; then
        echo "Error: no images yet. In module 3, upload your zip and run process_images.py first." >&2
        exit 1
    fi
}

# The address of this VM as seen from inside the kind cluster (the gateway of Docker's 'kind' network).
# Pipeline pods use it to reach module 3's MLflow, which runs on the VM, not in the cluster.
kind_gateway() {
    docker network inspect kind -f '{{range .IPAM.Config}}{{.Gateway}} {{end}}' 2>/dev/null | tr ' ' '\n' | grep -m1 '^[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*$' || true
}

cluster_mode() {  # prints "gpu" or "cpu"
    if [ -f "$MODE_FILE" ]; then cat "$MODE_FILE"; else echo "cpu"; fi
}

# Which PyTorch wheel index to build the image with. Prints nothing for the default PyPI wheel
# (CUDA 13 libraries, needs a recent GPU driver). Override with TORCH_INDEX_URL=<url> or TORCH_INDEX_URL=cpu.
# With no override: a CPU-only cluster gets the small CPU build.
detect_torch_index() {
    case "${TORCH_INDEX_URL:-}" in
        cpu) echo "https://download.pytorch.org/whl/cpu"; return ;;
        "") ;;
        *) echo "$TORCH_INDEX_URL"; return ;;
    esac
    if [ "$(cluster_mode)" = "cpu" ]; then
        echo "https://download.pytorch.org/whl/cpu"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local cuda
        cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\..*/\1/p' | head -n1)"
        if [ -n "$cuda" ] && [ "$cuda" -lt 13 ]; then
            echo "https://download.pytorch.org/whl/cu128"   # older driver: use a CUDA 12 build
        fi
    fi
}

# Write a kind cluster config. kubeadm changed how extraArgs are written between its config versions
# (v1beta3 = a map, v1beta4 = a list), so 02_create_cluster.sh tries both.
# The extraMounts block is how Docker hands the GPU to the kind node; it is harmless without a GPU.
write_kind_config() {  # write_kind_config <file> <v1beta3|v1beta4>
    local args
    if [ "$2" = "v1beta4" ]; then
        args='      extraArgs:
        - name: "service-account-issuer"
          value: "kubernetes.default.svc"
        - name: "service-account-signing-key-file"
          value: "/etc/kubernetes/pki/sa.key"'
    else
        args='      extraArgs:
        "service-account-issuer": "kubernetes.default.svc"
        "service-account-signing-key-file": "/etc/kubernetes/pki/sa.key"'
    fi
    cat > "$1" <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: ClusterConfiguration
    apiServer:
$args
  extraMounts:
  - hostPath: /dev/null
    containerPath: /var/run/nvidia-container-devices/all
EOF
}
