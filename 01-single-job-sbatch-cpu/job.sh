#!/bin/bash
# ==========================================================================
# 01 — single-job sbatch (CPU). The minimal Slurm batch script: "hello world".
# Submit with:  sbatch job.sh
# ==========================================================================

#SBATCH --account=def-YOURPI        # REQUIRED: your allocation. Replace def-YOURPI.
#SBATCH --job-name=moons-cpu        # Name shown by squeue/sacct.
#SBATCH --time=00:05:00             # Wall-clock limit HH:MM:SS. Job is killed after this.
#SBATCH --cpus-per-task=1           # CPU cores for the job.
#SBATCH --mem=2G                    # RAM for the job.
#SBATCH --output=slurm-%x-%j.out    # Stdout/stderr file (%x=job-name, %j=job-id).
#SBATCH --mail-user=YOUR_EMAIL      # OPTIONAL: notifications. Replace or delete.
#SBATCH --mail-type=END,FAIL        # OPTIONAL: when to email.

set -euo pipefail

# --- Environment -----------------------------------------------------------
# Compute nodes have NO internet. We only ACTIVATE the venv that was built
# once on the login node (see top-level README "One-time setup"); nothing is
# installed here, so no internet is needed.
module load python/3.11
source ~/.venv_ai/bin/activate

# --- Paths on $SCRATCH -----------------------------------------------------
# $SCRATCH is large and fast but purged ~every 60 days: perfect for data and
# results, wrong for anything you want to keep forever.
DATA_DIR="$SCRATCH/moons-tutorial/data"
OUT_DIR="$SCRATCH/moons-tutorial/results/$SLURM_JOB_NAME-$SLURM_JOB_ID"

# --- Run -------------------------------------------------------------------
python ../common/train.py \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR" \
    --epochs 50 --lr 0.05 --seed 0 \
    --device cpu

echo "Results written to: $OUT_DIR"
