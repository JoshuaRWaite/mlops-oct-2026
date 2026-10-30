#!/usr/bin/env bash
# Step 5: put the images in the cluster's file store (MinIO) so pipelines can use them.
#   ./05_upload_data.sh                    uploads module 3's uploaded_images/ (the photos you already used)
#   ./05_upload_data.sh some-images.zip    uploads a zip you name
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need kubectl
need_cluster
if [ $# -eq 0 ]; then
    need_module3_ready
    ZIP="$RUN_DIR/module3-images.zip"
    rm -f "$ZIP"
    (cd "$SHARED_DIR" && python3 -m zipfile -c "$ZIP" uploaded_images)
    echo "==> Zipped module 3's uploaded_images/ into $(basename "$ZIP")"
else
    ZIP="$(realpath "$1")"
    [ -f "$ZIP" ] || { echo "Error: $ZIP not found." >&2; exit 1; }
fi

# MinIO is Kubeflow Pipelines' file store. Its login is kept in a Kubernetes secret; read it from there.
secret() { kubectl get secret mlpipeline-minio-artifact -n kubeflow -o "jsonpath={.data.$1}" 2>/dev/null | base64 -d; }
MINIO_ACCESS_KEY="$(secret accesskey || true)"; : "${MINIO_ACCESS_KEY:=minio}"   # fall back to Kubeflow's defaults
MINIO_SECRET_KEY="$(secret secretkey || true)"; : "${MINIO_SECRET_KEY:=minio123}"
export MINIO_ACCESS_KEY MINIO_SECRET_KEY

echo "==> Opening a temporary connection to MinIO"
kubectl port-forward svc/minio-service -n kubeflow 9000:9000 >"$LOG_DIR/portforward-minio.log" 2>&1 &
PF_PID=$!
trap 'kill $PF_PID 2>/dev/null || true' EXIT
for _ in $(seq 20); do
    curl -s -o /dev/null http://127.0.0.1:9000/minio/health/live && break
    sleep 1
done

"$M4_DIR/.venv/bin/python" "$M4_DIR/upload_data.py" "$ZIP"
