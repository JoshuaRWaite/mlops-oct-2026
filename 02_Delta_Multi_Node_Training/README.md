# MNIST Training on Delta with PyTorch Lightning and Weights & Biases

This demo trains a small MNIST classifier on NCSA Delta GPUs and tracks the run in Weights & Biases (W&B).

| File | Purpose |
| --- | --- |
| `multi_node_delta.py` | Training script (PyTorch Lightning + W&B logging) |
| `multi_node_job.sh` | Slurm job script for Delta |
| `README.md` | This guide |

## Quick start: the main steps on Delta

1. **Log in** to Delta (VS Code or SSH) -> [Section 1](#1-log-in-to-delta)
2. **Set up the environment** once (load the PyTorch module + W&B login) -> [Section 2](#2-set-up-the-environment-one-time)
3. **Submit the job** with `sbatch multi_node_job.sh` -> [Section 3](#3-run-the-demo-on-delta)
4. **Monitor it** with `squeue -u $USER` and watch the run live on wandb.ai
5. **Check results** in `multi_node_op.txt` and in your W&B project

---

## 1. Log in to Delta

### Before you start

1. Find your **username**. It's in the email from *University of Illinois NCSA Account* (NCSA-Alloc@illinois.edu), and in your ACCESS profile under "Resource Provider Site Usernames".
2. **Set your NCSA password** using the link in that email.
3. **Set up Duo MFA** (Duo app on your phone) when prompted.

### Option A: VS Code (recommended)

1. Install the **Remote - SSH** extension.
2. Add this to `~/.ssh/config` (on Windows: `C:\Users\<you>\.ssh\config`):

   ```
   Host delta
       HostName login.delta.ncsa.illinois.edu
       User <username>
       PreferredAuthentications keyboard-interactive,password
   ```

3. Press `Ctrl/Cmd+Shift+P`, choose **Remote-SSH: Connect to Host...**, and pick `delta`.
4. Enter your NCSA password and approve Duo when asked.
5. Use **File > Open Folder** to open your Delta home or project directory.

### Option B: Terminal

```bash
ssh <username>@login.delta.ncsa.illinois.edu
```

Typing the password shows nothing, which is normal. At the Duo prompt, enter `1` for a push notification or the 6-digit passcode.

If login fails, try:

```bash
ssh -o PreferredAuthentications=keyboard-interactive,password <username>@login.delta.ncsa.illinois.edu
```

### Option C: Open OnDemand (if SSH isn't working)

Go to https://openondemand.delta.ncsa.illinois.edu and log in with your NCSA username, password and Duo. It gives you a file browser and a terminal in the browser.

More in the Delta docs: https://docs.ncsa.illinois.edu

---

## 2. Set up the environment (one time)

Run these on the Delta login node.

### Python environment

Delta's modules change over time, so check what is currently available:

```bash
module avail pytorch
```

`pytorch-conda/2.12` has worked on Delta recently:

```bash
module load pytorch-conda/2.12
pip install --user pytorch_lightning wandb
```

If a module name is no longer available, load the newest `pytorch-conda` version listed and use that in `multi_node_job.sh` too. Quick check that everything imports:

```bash
python -c "import torch, torchvision, pytorch_lightning, wandb; print(torch.__version__, torch.cuda.is_available())"
```

(`torch.cuda.is_available()` prints `False` on the login node because it has no GPU; that is expected.)

### Weights & Biases

1. Create a free account at https://wandb.ai and copy your API key from https://wandb.ai/authorize.
2. Log in. This saves your key to `~/.netrc`, so batch jobs can use it:

   ```bash
   wandb login
   ```

3. Check that it worked:

   ```bash
   wandb verify
   ```

---

## 3. Run the demo on Delta

### Submit the job

```bash
sbatch multi_node_job.sh
```

Slurm prints a job ID, such as `Submitted batch job 123456`. The job script requests 1 node with 4 A100 GPUs (`gpuA100x4` partition, `bcjh-delta-gpu` account).

### Monitor the job

| Command | What it does |
| --- | --- |
| `squeue -u $USER` | Show your queued and running jobs |
| `scancel <jobid>` | Cancel a job |
| `sacct -j <jobid>` | Show a finished job's status and runtime |
| `tail -f multi_node_op.txt` | Follow the job's output as it runs |

Errors go to `multi_node_error.e`.

### Watch the run in W&B

Open https://wandb.ai, go to the **mnist-mlops** project, and open the run named `delta-<jobid>`. You should see:

- Training and validation loss, and validation accuracy
- Gradient and weight histograms
- GPU utilization and memory under the **System** tab
- A table of sample predictions

If Delta compute nodes can't reach the internet, set `WANDB_MODE=offline` in `multi_node_job.sh`. After the job finishes, upload the results from the login node:

```bash
wandb sync wandb_logs/wandb/offline-run-*
```

---

## 4. Try it first without Slurm

On an interactive GPU node, or on a laptop (it runs on CPU if no GPU is found):

```bash
python multi_node_delta.py --epochs 3 --run-name test
```

Useful options: `--epochs`, `--batch-size`, `--lr`, `--hidden`, `--devices`, `--num-nodes`, `--project`, `--run-name`. Run it twice with different settings and compare the two runs in W&B.