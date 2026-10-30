# One image for every pipeline step in module 4.
#
# It builds on module 3: the scripts inside are the ones from ../03_Full_Cycle_MLOps, used as they are.
# Build from the REPO ROOT (the folder holding both 03_Full_Cycle_MLOps and 04_Kubeflow_And_Mlflow).
# build_image.sh does this for you:
#   docker build -f 04_Kubeflow_And_Mlflow/Dockerfile -t house-autoencoder:module4 .
FROM python:3.12-slim

# Which PyTorch wheels to install. Empty = the default PyPI wheel (bundles CUDA libraries).
# build_image.sh fills this in based on the GPU driver on the host (or the CPU-only build if there is no GPU).
ARG TORCH_INDEX_URL=""

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

COPY 04_Kubeflow_And_Mlflow/requirements-image.txt /tmp/requirements-image.txt
RUN if [ -n "$TORCH_INDEX_URL" ]; then \
        pip install torch==2.14.1 torchvision==0.29.1 --index-url "$TORCH_INDEX_URL"; \
    fi \
    && pip install -r /tmp/requirements-image.txt

# The same scripts you ran by hand in module 3 ...
COPY 03_Full_Cycle_MLOps/config.py 03_Full_Cycle_MLOps/model.py 03_Full_Cycle_MLOps/process_images.py 03_Full_Cycle_MLOps/train_model.py ./
# ... plus the one new step module 4 adds: the promotion gate.
COPY 04_Kubeflow_And_Mlflow/lifecycle/promote.py ./
