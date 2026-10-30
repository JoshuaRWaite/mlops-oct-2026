#!/usr/bin/env bash
# Step 4: connect the cluster to module 3. Builds the image from module 3's scripts, lets pipeline pods reach
# shellcheck disable=SC2031   # variables set inside the ( ... ) block below are meant to stay there
# module 3's MLflow (the registry the API serves from), and opens the Kubeflow Pipelines page on port 8080.
# Module 3 itself is not modified: this restarts only its MLflow server, with the same settings but listening on all addresses.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need docker; need kind; need kubectl; need curl
need_cluster
need_module3_ready

"$M4_DIR/build_image.sh"

GW="$(kind_gateway)"
[ -n "$GW" ] || { echo "Error: couldn't find the cluster's network. Is the 'kind' Docker network there? (docker network ls)" >&2; exit 1; }
URI="http://$GW:5000"
echo "$URI" > "$RUN_DIR/mlflow_uri"

echo "==> Making module 3's MLflow reachable from the cluster (at $URI)"
echo "    Only MLflow is restarted, and your data is kept. Module 3 normally starts it listening on this VM only."
(
    cd "$SHARED_DIR"
    source ./common.sh            # module 3's helpers: activate_venv, start_bg, stop_bg, wait_for_http
    activate_venv
    stop_bg mlflow
    # Same command as module 3's start_services.sh, except --host 0.0.0.0 (instead of 127.0.0.1)
    start_bg mlflow mlflow server \
        --backend-store-uri "sqlite:///$ROOT/mlflow.db" \
        --artifacts-destination "$ROOT/mlflow-artifacts" \
        --host 0.0.0.0 --port 5000
    wait_for_http mlflow http://127.0.0.1:5000/health
)

# MLflow has no login. Limit port 5000 to this VM and Docker's own networks, in case the VM's firewall is open.
if command -v iptables >/dev/null 2>&1; then
    if ! sudo iptables -C INPUT -p tcp --dport 5000 -j DROP 2>/dev/null; then
        sudo iptables -I INPUT -p tcp --dport 5000 -j DROP
        sudo iptables -I INPUT -p tcp --dport 5000 -s 172.16.0.0/12 -j ACCEPT
        sudo iptables -I INPUT -p tcp --dport 5000 -s 127.0.0.0/8 -j ACCEPT
        echo "==> Firewall: port 5000 now only accepts this VM and Docker networks"
    fi
else
    echo "WARNING: iptables not found. Make sure port 5000 is NOT open in your Jetstream2 security group." >&2
fi

echo "==> Checking that a pod inside the cluster can reach MLflow"
kubectl delete pod mlflow-check --ignore-not-found >/dev/null
kubectl run mlflow-check --rm -i --restart=Never --image="$IMAGE" --image-pull-policy=Never --command -- \
    python -c "import urllib.request; print('MLflow answered:', urllib.request.urlopen('$URI/health', timeout=10).read().decode())"

"$M4_DIR/port_forward.sh" start
echo
echo "Connected. Next: ./05_upload_data.sh"
