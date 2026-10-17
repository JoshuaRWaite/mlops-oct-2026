#!/bin/bash
# Copy/paste this job script into a text file and submit with the command:
#    sbatch multi_node_job.sh
#SBATCH --time=1:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1  # number of nodes
#SBATCH --mem=64g
#SBATCH --account=bjbq-delta-gpu 
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4 # one task per GPU (Lightning DDP expects this)
#SBATCH --job-name="multi_node"
#SBATCH --mail-user=[UPDATE WITH YOUR EMAIL ADDRESS HERE]   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="multi_node_op.txt" # job standard output file
#SBATCH --error="multi_node_error.e" # job standard error file
# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module list    # records what is loaded (good job metadata and reproducibility)
echo "job is starting on $(hostname)"
 
# This job uses the conda environment that was active when you ran sbatch
# (Delta docs: submit from within the environment, and do not add `conda activate` here).
echo "python: $(which python)"
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: torch not found. Activate the mlops environment, then run sbatch again (see README.md)." >&2
    exit 1
fi


# --- Weights & Biases ---
# Run `wandb login` once on the login node (see installation_instructions.md);
# the job then picks up your key from ~/.netrc. Do not paste the key in this file.
export WANDB_MODE=online        # if compute nodes can't reach the internet, use: offline
export WANDB_DIR="$PWD/wandb_logs"

srun python multi_node_delta.py --epochs 5 --run-name "delta-${SLURM_JOB_ID}"

# If you ran offline, upload the results from the login node afterward:
#   wandb sync wandb_logs/wandb/offline-run-*
