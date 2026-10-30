#!/usr/bin/env bash
# Step 3: install Kubeflow Pipelines (standalone): the pipeline engine, its web UI, and MinIO for files.
# Takes about 5-10 minutes. (The full Kubeflow platform is an optional extra: extras/full-kubeflow/)
#   KFP_VERSION=2.17.2 ./03_install_pipelines.sh     # pick another release
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need kubectl; need git
need_cluster

KFP_VERSION="${KFP_VERSION:-2.17.2}"
BASE="github.com/kubeflow/pipelines/manifests/kustomize"

echo "==> Installing Kubeflow Pipelines $KFP_VERSION"
kubectl apply -k "$BASE/cluster-scoped-resources?ref=$KFP_VERSION"
kubectl wait --for condition=established --timeout=120s crd/applications.app.k8s.io
kubectl apply -k "$BASE/env/dev?ref=$KFP_VERSION"

# A pod is fine when it is Completed, or Running with all its containers ready (READY shows 2/2, not 1/2).
not_ready() {
    kubectl get pods -n kubeflow --no-headers 2>/dev/null | awk '
        $3 == "Completed" { next }
        { split($2, r, "/"); if ($3 != "Running" || r[1] != r[2]) bad++ }
        END { print bad + 0 }'
}
echo "==> Waiting for all pods in the 'kubeflow' namespace to be ready (up to 15 minutes) ..."
for _ in $(seq 60); do
    n="$(not_ready)"
    [ "$n" = "0" ] && break
    echo "    $n pod(s) not ready yet ..."
    sleep 15
done
kubectl get pods -n kubeflow
if [ "$(not_ready)" != "0" ]; then
    echo "WARNING: some pods are still not ready. Wait a few minutes and check:  kubectl get pods -n kubeflow" >&2
else
    echo "Kubeflow Pipelines is ready."
fi
echo
echo "Next: ./04_connect.sh"
