#!/usr/bin/env bash
# Step 2: create the Kubernetes cluster (kind: Kubernetes running inside a Docker container).
# With a GPU on the VM, the GPU is made available inside the cluster too. Takes about 5 minutes.
#   FORCE_CPU=1 ./02_create_cluster.sh    # ignore the GPU
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need docker; need kind; need kubectl
need_docker_access

if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
    echo "Cluster '$CLUSTER_NAME' already exists. To start over:  kind delete cluster --name $CLUSTER_NAME"
    kind export kubeconfig --name "$CLUSTER_NAME"
    exit 0
fi

MODE=cpu
if [ -z "${FORCE_CPU:-}" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
    need nvkind; need helm
    MODE=gpu
fi
echo "==> Creating cluster '$CLUSTER_NAME' (mode: $MODE)"

# kubeadm changed its config format between Kubernetes versions; try the newer one, then the older.
create() {  # create <config-version>
    write_kind_config "$RUN_DIR/kind-config.yaml" "$1"
    if [ "$MODE" = gpu ]; then
        # nvkind wraps `kind` and passes the GPU through to the node container.
        local tmpl="$RUN_DIR/kind-config.tmpl.yaml"
        cp "$RUN_DIR/kind-config.yaml" "$tmpl"
        nvkind cluster create --name "$CLUSTER_NAME" --config-template="$tmpl"
    else
        kind create cluster --name "$CLUSTER_NAME" --config "$RUN_DIR/kind-config.yaml"
    fi
}
if ! create v1beta4; then
    echo "Retrying with the older config format ..."
    kind delete cluster --name "$CLUSTER_NAME" >/dev/null 2>&1 || true
    create v1beta3
fi
echo "$MODE" > "$MODE_FILE"

kubectl wait --for=condition=Ready nodes --all --timeout=300s

if [ "$MODE" = gpu ]; then
    echo "==> Installing the NVIDIA GPU Operator (the driver is already on the VM, so the operator skips it)"
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null
    helm repo update >/dev/null
    helm upgrade --install gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace \
        --set driver.enabled=false --wait --timeout 15m
    echo "==> Waiting for the node to advertise its GPU"
    for _ in $(seq 60); do
        n="$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || true)"
        [ -n "$n" ] && [ "$n" != "0" ] && { echo "Cluster has $n GPU(s)."; break; }
        sleep 10
    done
    if [ -z "${n:-}" ] || [ "$n" = "0" ]; then
        echo "WARNING: the cluster doesn't see a GPU. See 'Troubleshooting' in the README. You can continue on CPU:" >&2
        echo "  echo cpu > .run/mode" >&2
    fi
fi
echo
echo "Cluster is up (mode: $MODE). Next: ./03_install_pipelines.sh"
