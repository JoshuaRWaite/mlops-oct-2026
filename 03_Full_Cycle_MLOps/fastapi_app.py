"""Step 5: serve the champion model over HTTP and expose Prometheus metrics.

    uvicorn fastapi_app:app --host 127.0.0.1 --port 8000
"""
import io
import time
from contextlib import asynccontextmanager

import mlflow
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from mlflow import MlflowClient
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from config import IMG_SIZE, MLFLOW_TRACKING_URI, MODEL_ALIAS, MODEL_NAME, MODEL_URI

# --- Prometheus metrics: Prometheus scrapes these from /metrics every few seconds ---------
REQUESTS = Counter("model_requests_total", "Prediction requests by outcome", ["status"])
LATENCY = Histogram("model_latency_seconds", "Time to handle one prediction request")
# How badly the model rebuilt the input. A jump means the model is seeing images unlike its training data.
RECON_ERROR = Histogram(
    "model_reconstruction_error",
    "Mean squared error between the input image and its reconstruction",
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
)
for status in ("ok", "bad_request", "error"):
    REQUESTS.labels(status)  # create each series at 0 so dashboards have data before the first request

state = {}  # filled in at startup: the model, its device, and which version is loaded


def load_model(device):
    """Fetch the version holding the 'champion' alias from the MLflow Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        model = mlflow.pytorch.load_model(MODEL_URI, map_location=device)
        version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    except Exception as exc:
        raise RuntimeError(
            f"Could not load '{MODEL_URI}' from {MLFLOW_TRACKING_URI}. "
            "Is the MLflow server running (./start_services.sh) and has train_model.py finished?"
        ) from exc
    return model.to(device).eval(), str(version)


@asynccontextmanager
async def lifespan(app):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, version = load_model(device)
    state.update(model=model, device=device, version=version)
    print(f"Loaded {MODEL_NAME} version {version} on {device}")
    yield
    state.clear()


app = FastAPI(title="House autoencoder", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "version": state["version"], "device": str(state["device"])}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# A plain `def` (not `async def`): FastAPI runs it in a worker thread, so slow model
# inference doesn't block the server from answering other requests.
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """Send an image, get back the model's reconstruction as a PNG."""
    start = time.perf_counter()
    status = "ok"
    try:
        try:
            image = Image.open(file.file).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BOX)
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail="Upload must be a valid image file.")

        x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None, None].to(state["device"])
        with torch.no_grad():
            y = state["model"](x)
        RECON_ERROR.observe(torch.mean((x - y) ** 2).item())

        pixels = (y.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)  # .cpu() works on GPU or CPU
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")
    except HTTPException:
        status = "bad_request"
        raise
    except Exception:
        status = "error"
        raise
    finally:
        REQUESTS.labels(status).inc()
        LATENCY.observe(time.perf_counter() - start)
