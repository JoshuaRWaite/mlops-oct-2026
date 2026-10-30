#!/usr/bin/env bash
# OPTIONAL (larger instances): install the FULL Kubeflow platform (dashboard, Istio, Dex login, notebooks, Katib, ...)
# instead of standalone Pipelines. Takes 15-30 minutes and ~25 GB of images: use g3.large or bigger with a 100 GB+ disk.
# Pick a different release with:  KUBEFLOW_REF=v26.03.0 ./install_full_kubeflow.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../common.sh"
need git; need kubectl
if ! command -v kustomize >/dev/null 2>&1; then
    echo "==> Installing kustomize 5.7.1 (the full install is built with it)"
    curl -fsSL "https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv5.7.1/kustomize_v5.7.1_linux_amd64.tar.gz" | sudo tar -xz -C /usr/local/bin kustomize
    printf 'fs.inotify.max_user_instances = 2280\nfs.inotify.max_user_watches = 1255360\n' | sudo tee /etc/sysctl.d/99-kubeflow.conf >/dev/null
    sudo sysctl --system >/dev/null
fi
need_cluster

REPO_URL="https://github.com/kubeflow/manifests.git"
MANIFESTS_DIR="$M4_DIR/kubeflow-manifests"
MAX_TRIES="${MAX_TRIES:-30}"

# ---- 1. get the manifests at a release tag -------------------------------------
if [ -z "${KUBEFLOW_REF:-}" ]; then
    # Newest tag of the 26.xx releases that isn't a release candidate.
    KUBEFLOW_REF="$(git ls-remote --tags --refs --sort=-v:refname "$REPO_URL" 'v26.*' | sed 's#.*refs/tags/##' | grep -v -- '-' | head -n1 || true)"
fi
if [ -z "$KUBEFLOW_REF" ]; then
    echo "Error: couldn't find a v26.* release tag. Pick one from $REPO_URL/tags and run:" >&2
    echo "  KUBEFLOW_REF=<tag> ./install_full_kubeflow.sh" >&2
    exit 1
fi
if [ ! -d "$MANIFESTS_DIR" ]; then
    echo "==> Downloading Kubeflow manifests $KUBEFLOW_REF"
    git clone --depth 1 --branch "$KUBEFLOW_REF" "$REPO_URL" "$MANIFESTS_DIR"
fi

# ---- 2. apply them, retrying ----------------------------------------------------
# The first attempts fail on purpose: some resources depend on definitions (CRDs) that only exist
# after an earlier part of the same install finishes. Applying again a bit later works.
cd "$MANIFESTS_DIR"
for i in $(seq "$MAX_TRIES"); do
    echo "==> Applying Kubeflow (attempt $i of $MAX_TRIES)"
    if kustomize build example | kubectl apply --server-side --force-conflicts -f -; then
        echo "Everything was applied."
        break
    fi
    [ "$i" = "$MAX_TRIES" ] && { echo "Error: still failing after $MAX_TRIES attempts." >&2; exit 1; }
    echo "Some resources aren't ready for each other yet (expected). Retrying in 20 seconds ..."
    sleep 20
done

# ---- 3. wait until every pod is up ---------------------------------------------
# A pod is fine when it is Completed, or Running with all its containers ready (READY shows 2/2, not 1/2).
not_ready() {
    kubectl get pods -A --no-headers 2>/dev/null | awk '
        $4 == "Completed" { next }
        { split($3, r, "/"); if ($4 != "Running" || r[1] != r[2]) bad++ }
        END { print bad + 0 }'
}
echo "==> Waiting for all pods to become ready (up to 40 minutes) ..."
for _ in $(seq 160); do
    n="$(not_ready)"
    [ "$n" = "0" ] && break
    echo "    $n pod(s) not ready yet ..."
    sleep 15
done
kubectl get pods -A | head -n 100
if [ "$(not_ready)" != "0" ]; then
    echo "WARNING: some pods are still not ready. Give it a few more minutes and check:  kubectl get pods -A" >&2
else
    echo "All pods are ready."
fi

echo
echo "Full Kubeflow is installed. Open it with:  kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80"
echo "Then http://localhost:8080  (login user@example.com / 12341234)."
