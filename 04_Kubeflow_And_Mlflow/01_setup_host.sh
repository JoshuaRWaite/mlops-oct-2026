#!/usr/bin/env bash
# Step 1: install everything module 4 needs on the VM: Docker, kind, kubectl, helm, a Python venv,
# and (if the VM has a GPU) the NVIDIA container toolkit so Docker can hand the GPU to Kubernetes.
# Needs Ubuntu 24.04 and sudo. Safe to run again. Takes about 5 minutes.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

KIND_VERSION="v0.31.0"
K8S_CHANNEL="1.34/stable"

[ "$(uname -m)" = "x86_64" ] || { echo "Error: this script assumes an x86_64 VM." >&2; exit 1; }
need_module3
HAS_GPU=no
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then HAS_GPU=yes; fi
echo "==> GPU detected on this VM: $HAS_GPU"
[ "$HAS_GPU" = yes ] || echo "    (no GPU: everything still works, the training step just runs on the CPU)"

echo "==> Base packages"
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl git jq unzip python3-venv python3-pip snapd

echo "==> Docker"
if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sudo sh
fi
sudo usermod -aG docker "$USER"

echo "==> kind $KIND_VERSION (Kubernetes inside Docker)"
if ! command -v kind >/dev/null 2>&1; then
    sudo curl -fsSL -o /usr/local/bin/kind "https://kind.sigs.k8s.io/dl/$KIND_VERSION/kind-linux-amd64"
    sudo chmod +x /usr/local/bin/kind
fi

echo "==> kubectl, helm"
command -v kubectl >/dev/null 2>&1 || sudo snap install kubectl --classic --channel="$K8S_CHANNEL"
command -v helm    >/dev/null 2>&1 || sudo snap install helm --classic

echo "==> Python environment (.venv): the KFP SDK to compile pipelines, boto3 to upload data"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

if [ "$HAS_GPU" = yes ]; then
    echo "==> NVIDIA container toolkit (lets Docker, and so kind, use the GPU)"
    if ! command -v nvidia-ctk >/dev/null 2>&1; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        sudo apt-get update -y
        sudo apt-get install -y nvidia-container-toolkit
    fi
    sudo nvidia-ctk runtime configure --runtime=docker --set-as-default --cdi.enabled
    sudo nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true --in-place
    sudo systemctl restart docker

    echo "==> nvkind (kind with GPU support)"
    command -v go >/dev/null 2>&1 || sudo snap install go --classic
    if ! command -v nvkind >/dev/null 2>&1; then
        sudo env "PATH=$PATH:/snap/bin" GOBIN=/usr/local/bin go install github.com/NVIDIA/nvkind/cmd/nvkind@latest
    fi
fi

echo
echo "Done. IMPORTANT: log out and back in (or run 'newgrp docker') so you can use Docker without sudo."
echo "Then check it works:  docker run --rm hello-world"
[ "$HAS_GPU" = yes ] && echo "GPU check:            docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all ubuntu:24.04 nvidia-smi -L"
echo "Next: ./02_create_cluster.sh"
