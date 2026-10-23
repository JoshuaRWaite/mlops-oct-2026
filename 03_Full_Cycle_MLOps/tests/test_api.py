"""Smoke tests for the API. They use an untrained model, so no MLflow server is needed.

    pytest
"""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import fastapi_app
from model import AutoEncoder


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(fastapi_app, "load_model", lambda device: (AutoEncoder().to(device).eval(), "test"))
    with TestClient(fastapi_app.app) as c:  # the `with` runs the startup (lifespan) code
        yield c


def png_bytes(size=(300, 200)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 120, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["version"] == "test"


def test_predict_returns_128px_png(client):
    r = client.post("/predict", files={"file": ("house.png", png_bytes(), "image/png")})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(r.content)).size == (128, 128)


def test_bad_upload_is_400(client):
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_metrics_count_requests(client):
    client.post("/predict", files={"file": ("house.png", png_bytes(), "image/png")})
    text = client.get("/metrics").text
    assert 'model_requests_total{status="ok"}' in text
    assert "model_reconstruction_error_count" in text
