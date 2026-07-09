#!/bin/bash
# ==========================================================================
# 07 — single-job that survives being killed, and resumes on resubmission.
#
# The --time is deliberately SHORT so you can watch checkpoint/resume happen.
# The runtime folder uses a STABLE name (the job name, not the job id) so that
# resubmitting this exact script finds the previous checkpoint and continues.
# Submit with:  sbatch job.sh    (then scancel it, then sbatch job.sh again)
# ==========================================================================

#SBATCH --account=def-YOURPI        # REQUIRED: your allocation. Replace def-YOURPI.
#SBATCH --job-name=moons-resume
#SBATCH --time=00:02:00             # SHORT on purpose so you can see it get cut off.
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --requeue                   # let Slurm auto-resubmit us if preempted.
#SBATCH --signal=B:TERM@30          # send SIGTERM to THIS script 30s before the kill.
#SBATCH --mail-user=YOUR_EMAIL      # OPTIONAL. Replace or delete.
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module load python/3.11
source ~/.venv_ai/bin/activate

DATA_DIR="$SCRATCH/moons-tutorial/data"
OUT_DIR="$SCRATCH/moons-tutorial/results/$SLURM_JOB_NAME"
# STABLE runtime folder (job NAME, not job id) => resubmits resume from here.
RUNTIME_DIR="$SCRATCH/moons-tutorial/runtime/$SLURM_JOB_NAME"

# Forward Slurm's SIGTERM to the Python process so train.py can checkpoint
# cleanly before the node is taken away (the "automatic" resume path).
term_handler() {
    echo "[trap] SIGTERM received; forwarding to training PID $PID"
    kill -TERM "$PID"
    wait "$PID"
}
trap term_handler TERM

# Many epochs + slow-ish so the 2-minute wall time can't finish in one go.
python ../common/train.py \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR" \
    --checkpoint-dir "$RUNTIME_DIR" \
    --epochs 2000 --lr 0.01 --device cpu \
    --checkpoint-every 50 &
PID=$!
wait "$PID"

echo "Runtime folder (log + checkpoint): $RUNTIME_DIR"
