# MLOps Demo, part 2: when the data drifts, retrain automatically

> **Start with module 3** ([`../03_Full_Cycle_MLOps`](../03_Full_Cycle_MLOps/README.md)). This module picks up exactly where it ends and uses *its* MLflow, *its* API and *its* Prometheus. Module 3 is not changed: everything new lives in this folder. Keep both folders side by side and finish module 3 first.

Module 3 ended with a model API, a dashboard, and a question: *the error panel shows the data drifting, now what?* In this module you answer it. A **drift check** watches the numbers the API already produces. When the data has moved away from what the model was trained on, it **retrains on the new data**, a **gate** keeps the new model only if it beats the old one, and the API is restarted on the winner.

You build the loop twice:

- **Part A** runs it directly on module 3's stack, with one script and nothing new to install.
- **Part B** runs the same loop as a **Kubeflow pipeline** on a Kubernetes cluster (preprocess, train on a GPU pod, promote).

```
 module 3 (on the VM, unchanged)                                 module 4 adds
┌───────────────────────────────────────────────┐
│ photos ──► API :8000 ──metrics──► Prometheus :9090 ──► Grafana :3000
│              │  ▲                       │
│              │  │ loads "champion"      │ the same number the drift check reads
│              ▼  │ when it starts        ▼
│           MLflow :5000  ◄───── new version ◄──── train
└───────────────▲───────────────────────────────┘      ▲
                │ gate (promote.py): keep or roll back │
   06_check_drift: trigger? ──► original photos + new images ──► preprocess ──► train ──► gate
                                   (Part A: scripts on the VM, Part B: Kubeflow pods)
```

## The pieces

| Piece | What it is | Why we use it here |
|---|---|---|
| **Drift check** (`lifecycle/check_drift.py`) | A script with three triggers | Reads the API's recent reconstruction error from Prometheus, the number you watched in Grafana, and decides whether to retrain |
| **Gate** (`lifecycle/promote.py`) | A script that compares two models | Module 3's training makes every new model the champion right away. The gate scores the new model and the previous champion on the same images and **rolls back** if the new one is worse |
| **Kubeflow Pipelines** | Runs multi-step workflows on Kubernetes | Turns *preprocess → train → promote* into a pipeline with a graph, logs per step and a GPU request on the training step (Part B) |
| **Kubernetes (kind)** | A cluster of containers | Kubeflow needs one. [kind](https://kind.sigs.k8s.io) runs a small cluster inside Docker on the VM |
| **MinIO** | An object store that comes with Kubeflow Pipelines | Holds the zip of images that the pipeline reads |
| **Module 3's MLflow, API, Prometheus, Grafana** | Already running | Everything else is reused as it is |

**The three triggers** are the three retraining strategies from the monitoring lecture. Any one firing means "retrain":

| Trigger | Fires when | In this demo |
|---|---|---|
| **Performance** | The API's error over the last 2 minutes is more than 1.5× the champion's error when it was trained | This is the one that fires |
| **Data volume** | At least 500 new images have reached the API since the champion was trained | Set high on purpose, so it stays quiet |
| **Time** | The champion is more than 24 hours old | Stays quiet too |

**The drift here is synthetic.** `send_drift.sh` puts a bright grid over your photos, like a smudged lens or a new watermark, and sends them to the API. It gives you a real, measurable shift without waiting for the world to change. Unlike random static, a model *can* learn a grid, so retraining really helps. It also keeps a copy of every image it sends in `incoming_images/` (in this folder), standing in for "new data collected in production". The retraining set is your original photos plus those, so the new model learns the new look without forgetting the old one.

## What's in this repo

```
send_drift.sh            make the data drift: send gridded photos to the API
06_check_drift_local.sh  Part A: drift check -> retrain -> gate -> API restarted, on module 3's stack
01_setup_host.sh         Part B: install Docker, kind, kubectl, helm (and the GPU toolkit)
02_create_cluster.sh     Part B: create the Kubernetes cluster
03_install_pipelines.sh  Part B: install Kubeflow Pipelines
04_connect.sh            Part B: build the image, connect the cluster to module 3, open Kubeflow
05_upload_data.sh        Part B: put module 3's photos in the cluster's file store
06_check_drift.sh        Part B: drift check -> run the pipeline -> API restarted
common.sh                helper functions used by the scripts
build_image.sh           builds the container image and loads it into the cluster
port_forward.sh          opens the Kubeflow web page on port 8080

lifecycle/check_drift.py the three triggers
lifecycle/promote.py     the gate (also a pipeline step)
lifecycle/send_drift.py  what send_drift.sh runs
lifecycle/restart_api.sh restarts module 3's API so it loads the current champion
pipelines/               the pipeline (house_pipeline.py + compiled .yaml), run_pipeline.py, smoke test
Dockerfile               the image: module 3's scripts + promote.py
upload_data.py           helper for 05_upload_data.sh
extras/full-kubeflow/    optional: the full Kubeflow platform (needs a bigger machine)
```

## Before you start

1. **Finish module 3 on the same VM.** You need its Python environment, your photos uploaded and processed, a first model trained (`@champion` version 1), and the services and API running (`./start_services.sh`, `./start_api.sh`).
2. **Part B only:** use an **Ubuntu 24.04** instance, ideally a GPU flavor such as `g3.medium` (8 cores, 30 GB RAM, 60 GB disk, a 10 GB slice of an A100; [flavor list](https://docs.jetstream-cloud.org/general/vmsizes/)). The model is tiny, so that slice is plenty. *No GPU?* Everything also works on CPU; the scripts detect it. Don't suspend a GPU instance ([Jetstream2 GPU FAQ](https://docs.jetstream-cloud.org/faq/gpu/)); delete it when you're done.
3. **Open an SSH tunnel** as in module 3, and add port 8080 for Kubeflow (Part B):

```bash
ssh -N -L 8501:localhost:8501 -L 5000:localhost:5000 -L 3000:localhost:3000 -L 9090:localhost:9090 -L 8000:localhost:8000 -L 8080:localhost:8080 exouser@<your-instance-ip>
```

If you use VS Code Remote-SSH, forward the same ports in its Ports panel instead.

## Walkthrough

Run these on the VM unless a step says "laptop". If a script says `Permission denied`, run `chmod +x *.sh lifecycle/*.sh` in that folder.

# Part A: the loop on module 3's stack

### A1. Check that module 3 is running

```bash
cd 03_Full_Cycle_MLOps
./start_services.sh              # skips anything already running
./start_api.sh
curl http://127.0.0.1:8000/health
```

You should see `"version":"1"`. In MLflow (http://localhost:5000) under **Models → house-autoencoder**, version 1 has the `champion` alias.

### A2. Make the data drift

Move to this module and send 100 photos with a grid over them to the API:

```bash
cd ../04_Kubeflow_And_Mlflow
chmod +x *.sh lifecycle/*.sh
./send_drift.sh
```

Open Grafana (http://localhost:3000). The **Reconstruction error** panel jumps: the model has never seen photos like these. Two things were saved that you can't see: Prometheus stored the higher error, and `send_drift.sh` kept a copy of the 100 images in `incoming_images/`.

### A3. Run the drift check, and let it retrain

Do this within a couple of minutes of A2: the check looks at the last 2 minutes of traffic.

```bash
./06_check_drift_local.sh
```

The script does five things and prints each one:

1. **Drift check.** A table of the three triggers. Performance shows `FIRED` with a ratio around 2-3×: the API's recent error compared with the champion's error when it was trained (its final `val_loss` in MLflow). Data volume and time say `ok`. If nothing fires the script stops here. If it says "no traffic", run `./send_drift.sh` and try again.
2. **Preprocess.** Your original photos plus the new images from `incoming_images/`, with module 3's `process_images.py`.
3. **Train.** Module 3's `train_model.py`, 30 epochs. Just like in module 3, the new version becomes the champion as soon as it is registered.
4. **Gate.** `promote.py` scores the new version and the previous champion on the same images. It prints `PROMOTED`, or `REJECTED` and moves the `champion` alias back.
5. **Serve.** Module 3's API reads the champion once, when it starts. So the script restarts it with module 3's `start_api.sh` and ends with `/health` showing `"version":"2"`.

### A4. Look at the result

- **MLflow → Models → house-autoencoder:** version 2 now has the `champion` alias.
- **Grafana:** run `./send_drift.sh` again. This time the error stays low: the model learned the grid.
- **Run the check again** once the earlier traffic has left the 2-minute window: `./06_check_drift_local.sh --window 1m`. Nothing fires.

### A5. Watch the gate say no

A new model is not always a better one. Force a retrain with a deliberately weak model (1 epoch):

```bash
EPOCHS=1 ./06_check_drift_local.sh --force
```

`--force` retrains even though no trigger fired (the champion now handles the grid fine). The gate prints `REJECTED ... '@champion' moved back to version 2`. Version 3 stays in the registry, tagged `gate=rejected`, and `/health` still shows version 2.

### A6. Roll back by hand

Every version stays in the registry. In MLflow, open the model, open the older version, and move the `champion` alias back to it (Aliases → Edit). Then restart the API so it loads that version: `cd ../03_Full_Cycle_MLOps && ./start_api.sh`.

# Part B: the same loop as a Kubeflow pipeline

In Part A the three retraining steps ran as scripts on the VM. Here they run as a pipeline, each step in its own container on a Kubernetes cluster. The scripts are the same: the container image holds module 3's `process_images.py` and `train_model.py`, plus `promote.py`.

### B1. Install the tools

```bash
./01_setup_host.sh
```

This installs Docker, kind, kubectl and helm, plus a small Python environment for talking to Kubeflow. On a GPU machine it also installs the NVIDIA container toolkit so Docker can hand the GPU to Kubernetes. **Then log out and back in** (or run `newgrp docker`) so you can use Docker without `sudo`. Check with `docker run --rm hello-world`.

### B2. Create the cluster

```bash
./02_create_cluster.sh
```

This creates a Kubernetes cluster inside a Docker container. With a GPU it also installs the NVIDIA GPU Operator, so pods can ask for a GPU. Check with `kubectl get nodes` (the node says `Ready`). On a GPU machine the script prints how many GPUs the cluster sees. To ignore the GPU: `FORCE_CPU=1 ./02_create_cluster.sh`.

### B3. Install Kubeflow Pipelines

```bash
./03_install_pipelines.sh
```

This installs Kubeflow Pipelines: the pipeline engine, its web page, and MinIO for files. It waits until every pod is ready. Check with `kubectl get pods -n kubeflow` (everything `Running` or `Completed`).

### B4. Connect the cluster to module 3

```bash
./04_connect.sh
```

The pipeline's steps run in the cluster, but MLflow runs on the VM (module 3). This step bridges them:

- builds the container image from module 3's scripts and loads it into the cluster,
- **restarts MLflow only** (your data is kept, the API and monitoring keep running) so it can be reached from the cluster,
- limits port 5000 with the firewall to this VM and Docker's own networks, because MLflow has no login,
- starts a throwaway pod to prove it can reach MLflow (look for `MLflow answered:`),
- opens the Kubeflow web page on port 8080.

### B5. Upload the photos

```bash
./05_upload_data.sh
```

This zips module 3's `uploaded_images/` and copies it into the cluster's MinIO store, where the pipeline reads it. To use a different zip: `./05_upload_data.sh other.zip`.

### B6. Open Kubeflow and look at the pipeline

On your laptop, copy the compiled pipeline down (or download it in VS Code):

```bash
scp exouser@<your-instance-ip>:<path-to>/04_Kubeflow_And_Mlflow/pipelines/house_pipeline.yaml .
```

Open http://localhost:8080, choose **Pipelines → Upload pipeline**, and select `house_pipeline.yaml` (no GPU: `house_pipeline_cpu.yaml`). The graph has three steps:

| Step | What it runs | Notes |
|---|---|---|
| `preprocess` | module 3's `process_images.py` | unzips the photos, writes 128×128 grayscale PNGs |
| `train` | module 3's `train_model.py` | asks Kubernetes for 1 GPU; logs to module 3's MLflow |
| `promote` | `lifecycle/promote.py` | the gate: keeps the new model or rolls back |

You don't need to start a run from this page; the next step does it for you. (If you do, set `mlflow_uri` to the address in `.run/mlflow_uri`.)

### B7. Drift, detection and retraining, as a pipeline

Make the data drift, as in A2, and run the check right away:

```bash
./send_drift.sh
./06_check_drift.sh
```

It prints the same trigger table as Part A. When a trigger fires it zips your original photos plus the new images, uploads them, starts the pipeline, waits for it to finish, restarts the API on the winner, and prints `/health`, which now shows `"version":"2"`. While it waits, open the run in Kubeflow (**Runs**): you can watch the steps start, click a step and open **Logs**. The `promote` log shows both models' errors and `PROMOTED`.

Retraining uses 30 epochs here, so the new model really learns the grid.

### B8. Watch the pipeline's gate say no

```bash
.venv/bin/python pipelines/run_pipeline.py --epochs 1 --run-name weak --wait
```

This runs the pipeline with a weak 1-epoch model. The `promote` step's log ends with `REJECTED`, and the champion goes back to what it was. The API was never restarted, so it keeps serving the same model.

### B9. Stop and start over

```bash
./port_forward.sh stop                    # close the Kubeflow web page
kind delete cluster --name kubeflow       # remove the cluster (start again from B2)
```

Module 3 keeps working on its own; `../03_Full_Cycle_MLOps/stop_all.sh` stops it.

## How it works

- **Where the drift number comes from.** The API (module 3) records every request's reconstruction error as a Prometheus metric. `check_drift.py` asks Prometheus for the average over the last 2 minutes (`--window` changes that) and divides by the champion's `val_loss` from training. Above 1.5× (`--factor`) means drift.
- **Where the "new data" comes from.** Module 3's API doesn't store what it receives, so `send_drift.sh` keeps a copy of what it sends. The data-volume trigger counts the images in `incoming_images/` that are newer than the champion.
- **The gate.** Module 3's `train_model.py` registers the new version and makes it champion straight away. `promote.py` then loads the newest version and the previous champion, computes each one's mean reconstruction error on the images it was given, and keeps the new one only if its error is no higher. Otherwise it moves `champion` back and tags the version `gate=rejected`. It compares on the data the pipeline just trained on, not on held-out data, which keeps the demo simple.
- **Picking up the new model.** Module 3's API loads `models:/house-autoencoder@champion` once, at startup. `lifecycle/restart_api.sh` stops it, waits until port 8000 is free, and runs module 3's own `start_api.sh`. (In production you would roll out a new serving pod instead, or have the API poll the registry.)
- **Kubernetes can't see local Docker images**, so `build_image.sh` builds the image and loads it into the cluster with `kind load`. If you change a script in module 3, run `./build_image.sh` again before the next pipeline run.
- **What changed in module 3:** nothing. Part B's `04_connect.sh` only restarts MLflow, with the same command and data folders module 3 uses but listening on all addresses.

## Optional: the full Kubeflow platform

Part B installs **Kubeflow Pipelines standalone**, which fits a g3.medium. The full Kubeflow platform adds a central dashboard with login, per-user namespaces, Notebooks and Katib (hyperparameter tuning), but needs a bigger machine (g3.large or larger, 100 GB+ disk). See [`extras/full-kubeflow/README.md`](extras/full-kubeflow/README.md).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `06_*`: "No 'champion' model ... in MLflow" | Finish module 3's training first, and make sure MLflow is running (`../03_Full_Cycle_MLOps/start_services.sh`). |
| Drift check says `no traffic` | Nothing reached the API in the last 2 minutes. Run `./send_drift.sh` and check again straight away. |
| Nothing fires after drift traffic | Send more traffic right before the check, or look further back with `--window 5m`. You can also lower the threshold: `--factor 1.2`. |
| `Can't reach http://127.0.0.1:5000` | Module 3's services aren't running: `cd ../03_Full_Cycle_MLOps && ./start_services.sh && ./start_api.sh`. |
| `send_drift.sh` can't reach the API | `cd ../03_Full_Cycle_MLOps && ./start_api.sh`. |
| `docker: permission denied` | Log out and back in (or `newgrp docker`) after `01_setup_host.sh`. |
| The cluster has no GPU (B2) | Check `nvidia-smi` on the VM. To continue on CPU: `kind delete cluster --name kubeflow && FORCE_CPU=1 ./02_create_cluster.sh`. |
| The `train` pod stays `Pending` | No GPU is free for it. `kubectl describe pod -n kubeflow <pod>` shows why. Upload `house_pipeline_cpu.yaml`, or recreate the cluster with `FORCE_CPU=1`. With one GPU slice, two GPU runs at once queue up. |
| `04_connect.sh`: the pod can't reach MLflow | Re-run `./04_connect.sh`. After a VM reboot, or if you restart MLflow with module 3's `start_services.sh`, MLflow is back to localhost only, so run it again. Check the firewall with `sudo iptables -L INPUT -n`. |
| `ImagePullBackOff` | The image wasn't loaded into the cluster: `./build_image.sh`. |
| A pipeline step fails | In Kubeflow, open the run, click the step, then **Logs**. A quick plumbing check: `.venv/bin/python pipelines/smoke_test.py`, upload `pipelines/smoke_test.yaml` and run it (it checks the image starts, MLflow answers, and a GPU is visible). |
| Disk full / pods evicted | `docker system prune -a` frees space (it also removes the image: run `./build_image.sh` afterwards). |
| Old GPU driver, or PyTorch won't start in the image | `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./build_image.sh` |
| Kubeflow page won't open | `./port_forward.sh start`, and make sure port 8080 is in your SSH tunnel. |
| Pods look stuck after a reboot | Give them a few minutes, then `kubectl get pods -A | grep -v Running`. |

## A note on security

This setup is for a classroom, not production. Module 3's services bind to `127.0.0.1`; the only exception is MLflow during Part B, which listens on all addresses so the cluster can reach it, and `04_connect.sh` adds firewall rules that allow only this VM and Docker's networks. MLflow has no login, so don't open port 5000 in the Jetstream2 security group. Kubeflow's page is reached through your SSH tunnel and has no login in this standalone setup. MinIO uses Kubeflow's published demo credentials. Don't use any of this for real data.

## Where to go next

- Run the drift check on a schedule (a Kubeflow recurring run or a Kubernetes CronJob) instead of by hand.
- Compare models on a held-out test set in the gate, and use a statistical drift test instead of "error is 1.5× higher".
- Try hyperparameter search with Katib, or a larger model and images, on a bigger instance (g3.large or g3.xl).
- Serve the champion model from inside the cluster, with a model-serving tool such as KServe, so a new champion doesn't need an API restart.
- Add an alert in Prometheus or Grafana that fires when the error stays high, and have *it* start the retraining.
