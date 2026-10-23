# MLOps Demo: from a zip of photos to a monitored model API

In this demo you take a zip of house photos and, step by step, turn it into a **trained model, served behind an API, with live monitoring**. Every step is one command, and every tool runs on a single machine (built for a Jetstream2 GPU instance, but any Ubuntu box works).

```
 your laptop                          the VM (everything listens on localhost only)
┌───────────┐  SSH tunnel   ┌────────────────────────────────────────────────────────────┐
│  browser  │◄─────────────►│  Streamlit ──► uploaded_images/ ──► process_images.py      │
└───────────┘               │   (upload)                               │                 │
                            │                                          ▼                 │
                            │  MLflow ◄──────────────────────── train_model.py           │
                            │  (tracking + model registry)             │ registers       │
                            │      │ "champion" model                  ▼                 │
                            │      └────────────────────────► FastAPI  /predict          │
                            │                                   │      /metrics          │
                            │  Grafana ◄── Prometheus ◄─────────┘  (scraped every 5 s)   │
                            └────────────────────────────────────────────────────────────┘
```

**The model** is a small convolutional *autoencoder*. It squeezes a 128×128 grayscale image down to 512 numbers and then rebuilds the image from them. Nobody labelled anything: the model is graded on how well it reproduces its own input. The reconstruction error also turns out to be a useful health signal for monitoring (see [step 8](#8-watch-it-in-grafana)).

## The pieces

| Tool | What it is | Why we use it here | Port |
|---|---|---|---|
| **Streamlit** | Turns a Python script into a web page | A friendly "upload your zip" form | 8501 |
| **MLflow** | Experiment tracker and model registry | Records each training run (settings, loss curves, sample images) and stores the model with a version number | 5000 |
| **FastAPI** | Python web framework for APIs | Loads the model and answers `POST /predict` requests | 8000 |
| **Prometheus** | Time-series database that *pulls* metrics | Every 5 s it fetches `/metrics` from the API and stores the numbers | 9090 |
| **Grafana** | Dashboard tool | Draws Prometheus data as graphs | 3000 |
| **Weights & Biases** | Hosted experiment tracker (optional) | A second tracker to compare with MLflow; only used if you set an API key | n/a |

**Experiment tracking vs. model registry (both are MLflow).** *Tracking* is a lab notebook: one entry per training run. The *registry* is the shelf of approved models: each saved model gets a version number (v1, v2, ...) and the alias **`champion`** marks the one in production. The API loads `models:/house-autoencoder@champion`, so deploying a new model means moving an alias. No file copying or run IDs are involved.

## What's in this repo

```
setup_env.sh           create the Python environment and install packages
install_monitoring.sh  download Prometheus and Grafana into ./tools (no sudo)
start_services.sh      start MLflow, Prometheus, Grafana, Streamlit
start_api.sh           start (or restart) the model API
stop_all.sh            stop everything

streamlit_app.py       the upload page
process_images.py      raw photos -> 128x128 grayscale PNGs
model.py               the autoencoder definition
train_model.py         training + MLflow logging + registration
fastapi_app.py         the model API and its Prometheus metrics
generate_traffic.py    sends fake requests so the dashboard has data
make_sample_data.py    makes a fake zip of drawn houses (if you have no data)
config.py              names and settings shared by every script

prometheus.yml         tells Prometheus what to scrape
monitoring/grafana/    pre-built Grafana data source + dashboard
tests/                 API smoke tests (pytest)
```

## Before you start (Jetstream2)

1. **Create an instance** from an **Ubuntu 24.04** image. The pinned packages need Python 3.12+, and Ubuntu 22.04 ships 3.10.
2. **Pick a GPU flavor** such as `g3.medium`. These are slices of an A100 and are plenty for this model. GPU flavors need an allocation on the **Jetstream2-GPU** resource ([flavor list](https://docs.jetstream-cloud.org/general/vmsizes/)). *No GPU allocation?* Everything also works on CPU (see [Troubleshooting](#troubleshooting)).
3. Make sure you can **SSH in** with the key you gave the instance (Jetstream2 images use the username `exouser`).
4. Check the GPU is visible: `nvidia-smi` should print a table.

## Walkthrough

Run these on the VM unless a step says "laptop".

### 0. Get the code onto the VM

```bash
git clone <this-repo-url> mlops_demo     # or copy the folder up with scp
cd 03_Full_Cycle_MLOps
sudo apt update && sudo apt install -y python3-venv curl unzip
chmod +x *.sh                            # make the scripts runnable (copying often drops this)
```

### 1. Set up the environment

```bash
./setup_env.sh            # Python virtual environment + packages (a few GB, several minutes)
./install_monitoring.sh   # Prometheus + Grafana binaries into ./tools
```

`setup_env.sh` looks at `nvidia-smi` and installs the PyTorch build that matches your GPU driver. It finishes with a self-test that prints your PyTorch version and **`cuda available: True`** plus the GPU name. If it prints `False` on a GPU machine, stop and see [Troubleshooting](#troubleshooting).

The `start_*.sh` scripts activate the environment themselves, but for the `python ...` commands below you need to activate it in your shell first: `source .venv/bin/activate`.

### 2. Open an SSH tunnel (on your laptop)

All services listen on `127.0.0.1` only, so nothing is exposed to the internet. To reach them from your laptop's browser, forward their ports over SSH and **leave this running in its own terminal**:

```bash
ssh -N -L 8501:localhost:8501 -L 5000:localhost:5000 -L 3000:localhost:3000 -L 9090:localhost:9090 -L 8000:localhost:8000 exouser@<your-instance-ip>
```

Now `http://localhost:5000` on your laptop is MLflow on the VM, and so on. (`ssh` is built into macOS, Linux and current Windows.)

### 3. Start the services

```bash
./start_services.sh
```

This starts MLflow, Prometheus, Grafana and Streamlit in the background and waits until each answers. Logs are in `logs/`. Open these on your laptop:

| URL | What you should see |
|---|---|
| http://localhost:8501 | "House Image Upload" page |
| http://localhost:5000 | MLflow, empty for now |
| http://localhost:9090/targets | Prometheus. The `fastapi` target is **DOWN** until step 7 |
| http://localhost:3000 | Grafana, with the demo dashboard as the home page (panels are empty for now) |

### 4. Activate the environment
```bash
source .venv/bin/activate
```

### 5. Upload your images

Open http://localhost:8501 and upload your zip. Subfolders inside the zip are fine. The page extracts it into `uploaded_images/` and shows a few thumbnails.

*No data yet?* Make a fake zip of drawn houses and upload that instead:

```bash
python make_sample_data.py          # writes sample_houses.zip
```

Download it to your laptop with `scp exouser@<ip>:mlops_demo/sample_houses.zip .` and upload it in the page. *Zip too big for the browser?* Skip the page: `unzip -q your_images.zip -d uploaded_images`.

### 6. Preprocess

```bash
python process_images.py
```

Each image is converted to grayscale, shrunk to 128×128 and saved as a PNG in `processed_images/`. Corrupt files and macOS zip junk are skipped, and the script reports how many images it kept.

### 7. Train

```bash
python train_model.py              # 10 epochs; try --epochs 30
```

The script prints the loss each epoch and ends with `Registered 'house-autoencoder' version 1 -> alias '@champion'`. The command also accepts `--batch-size` and `--lr`.

Now open MLflow (http://localhost:5000) and look around:

- **Experiments → House Autoencoder → your run**: parameters, the `train_loss` / `val_loss` curves, and under *Artifacts* a `reconstructions/` folder. Each picture shows 8 originals on top and the model's rebuilds below, and you can watch them sharpen from epoch to epoch.
- **Models → house-autoencoder**: version 1, with the `champion` alias.

> **Why a validation set?** 10% of the images are held out and never trained on. If `train_loss` keeps falling but `val_loss` stalls or rises, the model is memorizing its training photos.

### 8. Serve the model

```bash
./start_api.sh
```

It loads whichever version holds `@champion` and prints `{"status":"ok","version":"1",...}` when ready. Try it:

```bash
curl -F "file=@processed_images/000000.png" http://127.0.0.1:8000/predict -o prediction.png
```

`prediction.png` is the model's reconstruction. The API also serves interactive docs at http://localhost:8000/docs (try the upload form there), and its raw metrics at http://localhost:8000/metrics.

### 9. Watch it in Grafana

Generate some traffic:

```bash
python generate_traffic.py                          # 200 normal requests
python generate_traffic.py --noise-fraction 0.3      # 30% of requests are random static
```

Then open Grafana (http://localhost:3000; the dashboard refreshes every 5 s). The panels show:

- **Requests per second**, split by outcome (`ok`, `bad_request`, `error`).
- **Latency** p50 / p95: the time within which 50% / 95% of requests finished.
- **Reconstruction error**: how badly the model rebuilt each input. After the noisy run you'll see it jump. The model has only learned to rebuild houses, so static looks "unlike anything I was trained on". That is a simple **data drift** detector: in production, a rising error tells you the inputs have changed and the model may need retraining.
- **Scrape targets**: whether Prometheus can reach the API.

Also send a bad upload to see the `bad_request` counter move: `curl -F "file=@README.md" http://127.0.0.1:8000/predict`.

In Prometheus (http://localhost:9090) you can run the dashboard's queries yourself, for example `sum(rate(model_requests_total[1m]))`.

### 10. Ship a new model

Retrain, for example with more epochs, then restart the API:

```bash
python train_model.py --epochs 30     # registers version 2 and moves @champion to it
./start_api.sh                        # restart so the API loads the new champion
curl http://127.0.0.1:8000/health     # "version":"2"
```

In MLflow's Models page you can move the alias back to version 1 yourself (a *rollback*). Run `./start_api.sh` again afterwards. Note that `train_model.py` promotes every new model automatically, which keeps the demo short. A production pipeline would first check that the new model beats the current champion.

### 11. Stop everything

```bash
./stop_all.sh
```

Your data (`mlflow.db`, `mlflow-artifacts/`, `processed_images/`) stays in place, and `./start_services.sh` brings everything back.

## Optional: Weights & Biases

If you set an API key, `train_model.py` also logs to [wandb.ai](https://wandb.ai), so you can compare the two trackers side by side:

```bash
export WANDB_API_KEY=<your key>     # from wandb.ai/authorize
python train_model.py
```

Without the variable, W&B is skipped.

## Run the tests

```bash
pytest
```

These check the API (health, prediction, bad uploads, metrics) using an untrained model, so they don't need MLflow running.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Python 3.12+ required` | Launch an Ubuntu 24.04 image. |
| `python3-venv is missing` | `sudo apt install -y python3-venv` |
| `cuda available: False` on a GPU machine | Run `nvidia-smi`. If it errors, the instance has no GPU or driver (check the flavor). If it works, reinstall PyTorch for your driver, e.g. `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./setup_env.sh`. |
| No GPU at all | `TORCH_INDEX_URL=cpu ./setup_env.sh`. Training and serving still work, just on CPU. |
| `start_api.sh`: "Could not load models:/house-autoencoder@champion" | Train first (step 6) and make sure MLflow is running (`./start_services.sh`). |
| Prometheus shows `fastapi` DOWN | The API isn't running. Run `./start_api.sh`. |
| Grafana panels say "No data" | Check the target is UP at http://localhost:9090/targets, then run `generate_traffic.py`. |
| Browser can't open `localhost:5000` etc. | The SSH tunnel isn't running, or a program on your laptop already uses that port. Change the left number in the `-L` options, e.g. `-L 5001:localhost:5000`. |
| Upload fails or stalls | The zip may be too big for the browser. Use `unzip -q file.zip -d uploaded_images` instead. |
| `install_monitoring.sh` can't download | Download the Prometheus and Grafana Linux tarballs by hand from their official sites and unpack each into `tools/prometheus` and `tools/grafana` (the folder should contain the `prometheus` and `bin/grafana` binaries). |
| Something is stuck after a reboot | `./stop_all.sh && ./start_services.sh`. Service logs are in `logs/<name>.log`. |

## A note on security

This setup is for a classroom, not production. Services bind to `127.0.0.1`, so the only way in is your SSH tunnel. That matters because MLflow has no login, and Grafana's admin account is the well-known `admin` / `admin` (anonymous visitors get read-only access). Don't open these ports to the internet.

## Where to go next

- Containerize each piece with Docker (the services map one-to-one onto containers).
- Add an alert in Grafana or Prometheus for when reconstruction error stays high.
- Only promote a new model if its `val_loss` beats the current champion's.
- Replace the autoencoder with a classifier or regressor on labelled data. Only `model.py`, `train_model.py` and the `/predict` handler change.
