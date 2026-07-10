#!/bin/bash
# ==========================================================================
# 04 — single-job driven by a Hydra config.yaml.
# The GPU variant: config has device=cuda, so we ALSO request a GPU below.
# Submit with:  sbatch job.sh
# ==========================================================================

#SBATCH --account=def-YOURPI        # REQUIRED: your allocation. Replace def-YOURPI.
#SBATCH --job-name=moons-hydra
#SBATCH --time=00:05:00
#SBATCH --gpus-per-node=1           # Must match device=cuda in config.yaml (see README).
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

module load python/3.11
source ~/.venv_ai/bin/activate

DATA_DIR="$SCRATCH/moons-tutorial/data"
OUT_DIR="$SCRATCH/moons-tutorial/results/$SLURM_JOB_NAME-$SLURM_JOB_ID"

# --config-dir "$PWD" tells Hydra to read config.yaml from THIS folder.
# data_dir/out_dir are overridden here so they resolve to $SCRATCH at run time.
# To run on CPU instead, set device=cpu AND delete the --gpus-per-node line above.
python ../common/train_hydra.py \
    --config-dir "$PWD" \
    data_dir="$DATA_DIR" \
    out_dir="$OUT_DIR" \
    device=cuda

echo "Results written to: $OUT_DIR"
