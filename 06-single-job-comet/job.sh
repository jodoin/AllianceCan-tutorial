#!/bin/bash
# ==========================================================================
# 06 — single-job with Comet.ml training-curve logging.
#
# Two modes (pick ONE — see README):
#   OFFLINE (default, recommended): job writes a Comet archive to $SCRATCH;
#           you `comet upload` it from the login node afterward. No internet
#           needed on the compute node.
#   ONLINE: load the httpproxy module so the compute node can reach Comet live.
#
# Submit with:  sbatch job.sh
# ==========================================================================

#SBATCH --account=def-YOURPI        # REQUIRED: your allocation. Replace def-YOURPI.
#SBATCH --job-name=moons-comet
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-user=YOUR_EMAIL      # OPTIONAL. Replace or delete.
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module load python/3.11
source ~/.venv_ai/bin/activate

# Comet API key: NEVER hardcode it. Read it from the environment (or from
# ~/.comet.config). Set it before submitting, e.g.  export COMET_API_KEY=...
export COMET_API_KEY="${COMET_API_KEY:-YOUR_COMET_API_KEY}"

DATA_DIR="$SCRATCH/moons-tutorial/data"
OUT_DIR="$SCRATCH/moons-tutorial/results/$SLURM_JOB_NAME-$SLURM_JOB_ID"
COMET_OFFLINE_DIR="$SCRATCH/moons-tutorial/comet-offline/$SLURM_JOB_ID"

# --- OFFLINE mode (default) ------------------------------------------------
python ../common/train.py \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR" \
    --epochs 50 --lr 0.05 --device cpu \
    --comet --comet-offline-dir "$COMET_OFFLINE_DIR"

echo "Comet offline archive in: $COMET_OFFLINE_DIR"
echo "Upload it from the LOGIN node with:  comet upload $COMET_OFFLINE_DIR/*.zip"

# --- ONLINE mode (alternative) ---------------------------------------------
# To stream curves live instead, comment out the OFFLINE block above and use:
#   module load httpproxy
#   python ../common/train.py --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" \
#       --epochs 50 --lr 0.05 --device cpu --comet
