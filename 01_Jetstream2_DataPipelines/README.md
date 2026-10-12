# Module 1: Data Pipelines on Jetstream2

A small video-to-annotation pipeline:

1. **Streamlit app** (`app.py`) takes an uploaded video and saves every frame as a JPEG.
2. **Label Studio** is used to annotate those frames for model training.
3. **Airflow DAG** (`streamlit_labelstudio_dag.py`) launches both containers for you.

You will run it two ways: first with plain Docker (Part 1), then orchestrated by Airflow (Part 2).

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit video frame extractor (OpenCV). Writes frames to `/app/frames` inside the container. |
| `Dockerfile` | Builds the `streamlit-app:latest` image. |
| `requirements.txt` | Python dependencies for the Streamlit image. |
| `streamlit_labelstudio_dag.py` | Airflow DAG: `launch_streamlit` then `launch_label_studio`. Manual trigger only. |
| `requirements-airflow.txt` | Airflow (pinned to 2.10.5) and the Docker SDK, for the host virtual environment. |
| `20708416-uhd_2160_3840_60fps.mp4` | Sample video (about 25 MB) to test the pipeline. |

---

## Ports: how to reach the apps from your computer

> **Read this first.** On Jetstream2, `localhost` means the VM itself, not your laptop. To open an app from your own browser, you must connect to the right **VM port**. Note that Label Studio is **not** on its default port.

| App | URL on the VM | VM (host) port | Container port | Note |
|-----|---------------|----------------|----------------|------|
| Streamlit | `http://localhost:8501` | **8501** | 8501 | Default Streamlit port |
| Label Studio | `http://localhost:8081` | **8081** | 8080 | **Remapped** from its default 8080 so it doesn't collide with Airflow |
| Airflow UI | `http://localhost:8080` | **8080** | n/a (runs on the host) | Printed admin password on first start |

The most common mistake is opening Label Studio at 8080 (that is Airflow's port) instead of **8081**.

### Connecting from outside the VM

**Option A: SSH tunnel (recommended).** Run this on your own computer, replacing `<instance-ip>` with your instance's IP:

```bash
ssh -L 8501:localhost:8501 -L 8081:localhost:8081 -L 8080:localhost:8080 exouser@<instance-ip>
```

Leave it running, then browse to `http://localhost:8501`, `http://localhost:8081`, and `http://localhost:8080` on your own computer. You only need the `-L` flags for the apps you are using.

**Option B: Direct access.** Open the port in your instance's security group, then browse to `http://<instance-ip>:<port>`. Only do this for the ports you need (8501, 8081, 8080), since anything you open is reachable from the internet.

If you are working inside the Jetstream2 **web desktop**, use the `localhost` URLs in the table directly. No tunnel is needed.

---

## Part 1: Run with plain Docker

### One-time setup

```bash
cd ~/01_Jetstream2_DataPipelines

# Let exouser use Docker without sudo (then log out and back in, or run: newgrp docker)
sudo usermod -aG docker $USER

# Label Studio's container runs as uid 1001, so it needs write access to the mounted folders
sudo chown -R 1001:1001 .      # or: chmod -R a+rwX .
```

### Streamlit (port 8501)

```bash
docker build -t streamlit-app:latest .

docker run -d --name streamlit -p 8501:8501 \
  -v "$PWD/frames":/app/frames \
  streamlit-app:latest
```

Open `http://localhost:8501`, upload a video (try the sample), and click **Process Video**. Frames appear in `~/01_Jetstream2_DataPipelines/frames/`.

### Label Studio (port 8081)

```bash
docker run -d --name label-studio -p 8081:8080 \
  -v "$PWD/ls-data":/label-studio/data \
  -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
  -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data \
  heartexlabs/label-studio:latest
```

Note `-p 8081:8080`: host port **8081** maps to Label Studio's internal 8080. Open `http://localhost:8081` and create an account.

### Useful commands

```bash
docker logs -f streamlit              # follow logs (debugging)
docker logs -f label-studio
docker ps                             # what's running
docker rm -f streamlit label-studio   # stop and remove both
```

---

## Part 2: Orchestrate with Airflow

Airflow runs on the VM itself (not in a container) and starts the same two containers through the Docker SDK.

> **Stop the Part 1 containers first** (`docker rm -f streamlit label-studio`). The DAG uses the same container names and ports. It also removes leftover containers of the same name when it runs, so re-triggering is safe.

### Install Airflow 2.x in a virtual environment

```bash
cd ~/01_Jetstream2_DataPipelines
sudo apt-get update && sudo apt-get install -y python3-venv
python3 -m venv af-venv && source af-venv/bin/activate

# Pin Airflow 2.x with its constraints file. The DAG uses Airflow 2 style imports, which break on Airflow 3.
AF=2.10.5
PY=$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
pip install "apache-airflow==${AF}" docker \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AF}/constraints-${PY}.txt"
```

### Build the image, install the DAG, start Airflow

```bash
docker build -t streamlit-app:latest .

export AIRFLOW_HOME=~/airflow
mkdir -p $AIRFLOW_HOME/dags
cp streamlit_labelstudio_dag.py $AIRFLOW_HOME/dags/

airflow standalone      # prints the admin password; UI at http://localhost:8080
```

### Run the DAG

1. Open the Airflow UI (`http://localhost:8080`) and log in as `admin` with the printed password.
2. Find `streamlit_labelstudio_pipeline` and trigger it (the play button). It has no schedule.
3. When both tasks are green, use the ports above: Streamlit on **8501**, Label Studio on **8081**.

The DAG's host paths (`PROJECT_DIR`, `frames/`, `ls-data/`) and ports are constants at the top of `streamlit_labelstudio_dag.py`.

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---------|----------------------|
| Label Studio page won't load | You're on port 8080 (Airflow). Use **8081**. |
| `permission denied` running `docker` | Add yourself to the docker group (one-time setup), then log out and back in or run `newgrp docker`. |
| Label Studio crashes or can't write data | Its folder isn't writable by uid 1001: `sudo chown -R 1001:1001 ls-data` (or `chmod -R a+rwX ls-data`). |
| `container name "/streamlit" is already in use` | `docker rm -f streamlit label-studio`, then retry. |
| Port already allocated | Another container or process holds the port. Check `docker ps`. |
| Frames don't appear in `frames/` | The `-v` mount path must match the app's output directory (`/app/frames`). Mounts are fixed when the container starts, so remove and re-run it. |
| DAG fails to import with Airflow errors | You have Airflow 3. Recreate the venv with the pinned 2.10.5 install above. |
| Streamlit app not reachable from your laptop | Use the SSH tunnel (or open the port in your security group). `localhost` on your laptop is not the VM. |

## Notes

- Airflow is a teaching scaffold here: the DAG only launches two containers. It becomes more useful as later steps (deduplication, pre-annotation, training triggers) are added.
- Label Studio only sees files under its mounted `ls-data/` folder. To annotate extracted frames, the frames need to be reachable from that folder (for example, copy or move them into `ls-data/`), then import them through Label Studio's local-files storage.
