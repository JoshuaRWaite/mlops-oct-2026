#!/usr/bin/env bash
# Build the image that every pipeline step (and MLflow) runs from, and load it into the cluster.
# 04_connect.sh runs this for you. Run it again yourself after changing a script in
# 03_Full_Cycle_MLOps. The first build is slow (PyTorch is several GB on a GPU cluster).
#   TORCH_INDEX_URL=cpu ./build_image.sh    # force CPU-only PyTorch
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need docker; need kind
need_docker_access
need_module3

TORCH_INDEX="$(detect_torch_index)"
echo "==> Building $IMAGE (cluster mode: $(cluster_mode); PyTorch from: ${TORCH_INDEX:-the default PyPI wheel})"
# The build context is the repo root, because the image contains scripts from 03_Full_Cycle_MLOps.
docker build -f "$M4_DIR/Dockerfile" --build-arg TORCH_INDEX_URL="$TORCH_INDEX" -t "$IMAGE" "$REPO_ROOT"

echo "==> Loading it into the '$CLUSTER_NAME' cluster (the cluster can't see your local Docker images otherwise)"
kind load docker-image "$IMAGE" --name "$CLUSTER_NAME"
docker builder prune -f >/dev/null 2>&1 || true   # free build cache (disk is tight on a g3.medium)
echo "Done."
