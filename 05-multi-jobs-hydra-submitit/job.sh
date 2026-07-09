#!/bin/bash
# ==========================================================================
# 05 — multi-jobs sweep via Hydra --multirun + submitit-slurm launcher.
#
# UNLIKE the other examples, you do NOT `sbatch` this file. You RUN it on the
# LOGIN node:   bash job.sh
# Hydra + submitit then submit one Slurm job per swept value for you.
# (That's why there are no #SBATCH lines here — submitit adds them from the
#  launcher config in config.yaml.)
# ==========================================================================

set -euo pipefail

module load python/3.11
source ~/.venv_ai/bin/activate

DATA_DIR="$SCRATCH/moons-tutorial/data"
OUT_DIR="$SCRATCH/moons-tutorial/results/moons-sweep"

# --multirun + lr=0.001,0.01,0.1  ->  three Slurm jobs, one per learning rate.
python ../common/train_hydra.py \
    --config-dir "$PWD" \
    --multirun \
    data_dir="$DATA_DIR" \
    out_dir="$OUT_DIR" \
    device=cpu \
    lr=0.001,0.01,0.1

echo "Submitted the sweep. Watch it with:  sq"
