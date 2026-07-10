#!/bin/bash
# ==========================================================================
# 02 — single-job sbatch (GPU). Same as 01, plus one GPU.
# Submit with:  sbatch job.sh
# ==========================================================================

#SBATCH --account=def-YOURPI        # REQUIRED: your allocation. Replace def-YOURPI.
#SBATCH --job-name=moons-gpu
#SBATCH --time=00:05:00
#SBATCH --gpus-per-node=1           # NEW vs 01: request 1 GPU (Narval = A100 40GB).
#SBATCH --cpus-per-task=4           # Narval has ~12 cores per GPU; 4 is plenty here.
#SBATCH --mem=16G                   # Match the resources to the GPU you asked for.
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

module load python/3.11
source ~/.venv_ai/bin/activate

# Prove a GPU is actually attached to this job (handy sanity check).
echo "===== nvidia-smi ====="
nvidia-smi
echo "===== torch sees CUDA? ====="
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

DATA_DIR="$SCRATCH/moons-tutorial/data"
OUT_DIR="$SCRATCH/moons-tutorial/results/$SLURM_JOB_NAME-$SLURM_JOB_ID"

# --device cuda: use the GPU. train.py still falls back to CPU if none is found,
# so the SAME script runs unchanged on CPU (01) and GPU (02).
python ../common/train.py \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR" \
    --epochs 50 --lr 0.05 --seed 0 \
    --device cuda

echo "Results written to: $OUT_DIR"
