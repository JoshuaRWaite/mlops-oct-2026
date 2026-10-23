#!/usr/bin/env bash
# Step 1: environment setup. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR=".venv"

# --- 1. Check prerequisites -------------------------------------------------
# Only things that must exist BEFORE the venv: python itself.
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found. Install Python 3.10+ and retry." >&2
    exit 1
fi

# The pinned numpy 2.5.x needs Python 3.12+ (Ubuntu 24.04's default python3).
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
    echo "Error: Python 3.12+ required, found $(python3 --version)." >&2
    echo "On Jetstream2, launch an Ubuntu 24.04 image (22.04 ships Python 3.10)." >&2
    exit 1
fi

# On Ubuntu, venv lives in a separate package.
if ! python3 -c 'import ensurepip, venv' >/dev/null 2>&1; then
    echo "Error: python3-venv is missing. Run: sudo apt install -y python3-venv" >&2
    exit 1
fi

# --- 2. Create venv ----------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# --- 3. Install pinned requirements -----------------------------------------
#   ./setup_env.sh                          -> GPU build matched to the VM's NVIDIA driver
#   TORCH_INDEX_URL=<url> ./setup_env.sh    -> force a specific PyTorch wheel index
#   TORCH_INDEX_URL=cpu ./setup_env.sh      -> CPU-only build (no GPU on this VM)
echo "Installing requirements ..."
python -m pip install --upgrade pip

TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
if [ "$TORCH_INDEX_URL" = "cpu" ]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
elif [ -z "$TORCH_INDEX_URL" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        # "CUDA Version: 12.8" in the nvidia-smi header = newest CUDA the driver supports
        DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -n1)"
        echo "nvidia-smi reports driver CUDA support: ${DRIVER_CUDA:-unknown}"
        DRIVER_MAJOR="${DRIVER_CUDA%%.*}"
        if [ -n "$DRIVER_MAJOR" ] && [ "$DRIVER_MAJOR" -ge 13 ]; then
            : # the default PyPI wheel (CUDA 13 libs bundled) works as-is
        else
            # Driver is older than CUDA 13: use a CUDA 12.x build instead.
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
        fi
    else
        echo "Warning: nvidia-smi not found, so no GPU driver is visible on this VM."
        echo "         Re-run with TORCH_INDEX_URL=cpu if this is intentional."
    fi
fi

if [ -n "$TORCH_INDEX_URL" ]; then
    echo "Installing torch from $TORCH_INDEX_URL"
    python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url "$TORCH_INDEX_URL"
fi
# torch/torchvision are now satisfied, so this won't replace them.
python -m pip install -r requirements.txt

# --- 4. Local project folders -----------------------------------------------
mkdir -p uploaded_images processed_images mlflow-artifacts

# --- 5. Smoke test: do the imports work? ------------------------------------
python - <<'EOF'
import cv2, fastapi, mlflow, numpy, PIL, prometheus_client, streamlit, torch, torchvision, wandb
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"mlflow {mlflow.__version__} | numpy {numpy.__version__} | opencv {cv2.__version__}")
print("All imports OK.")
EOF

echo
echo "Done. Activate with:  source $VENV_DIR/bin/activate"
