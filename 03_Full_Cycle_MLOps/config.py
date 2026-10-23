"""Settings shared by every script. Override any of them with environment variables."""
import os

# Where the MLflow tracking server listens (started by start_services.sh).
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "House Autoencoder"

# The model registry entry. The API always serves whichever version holds the alias.
MODEL_NAME = "house-autoencoder"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

# Images are shrunk to IMG_SIZE x IMG_SIZE grayscale before training and serving.
IMG_SIZE = 128
UPLOAD_DIR = "uploaded_images"      # raw images extracted from the zip
PROCESSED_DIR = "processed_images"  # cleaned 128x128 grayscale PNGs
