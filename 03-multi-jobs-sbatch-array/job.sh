#!/bin/bash
# ==========================================================================
# 03 — multi-jobs via Slurm job ARRAY. ONE `sbatch` submission launches
# FIVE independent jobs, each with a different seed. This is the canonical
# "how do I launch many jobs at once" answer on Alliance clusters.
# Submit with:  sbatch job.sh
# ==========================================================================

#SBATCH --account=def-YOURPI        # REQUIRED: your allocation. Replace def-YOURPI.
#SBATCH --job-name=moons-array
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-4                  # NEW: run 5 tasks, index 0..4. Each is its own job.
#SBATCH --output=slurm-%x-%A_%a.out  # %A=array job id, %a=this task's index.

set -euo pipefail

module load python/3.11
source ~/.venv_ai/bin/activate

# $SLURM_ARRAY_TASK_ID is 0,1,2,3,4 in the five tasks. Use it to vary a
# hyperparameter — here the random seed — and to give each task its own folder.
SEED=$SLURM_ARRAY_TASK_ID

DATA_DIR="$SCRATCH/moons-tutorial/data"
OUT_DIR="$SCRATCH/moons-tutorial/results/$SLURM_JOB_NAME-$SLURM_ARRAY_JOB_ID/seed-$SEED"

python ../common/train.py \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR" \
    --epochs 50 --lr 0.05 --seed "$SEED" \
    --device cpu

echo "Task $SLURM_ARRAY_TASK_ID (seed=$SEED) results -> $OUT_DIR"
