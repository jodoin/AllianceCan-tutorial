# 01 — Single-job, sbatch, CPU

**Type:** single-job (one Slurm job from one submission). **Runs on:** a Narval CPU compute node.

This is the Slurm "hello world": take the exact training run you did locally in
[00](../00-single-job-local-cpu/) and hand it to the scheduler with `sbatch`.

## Prerequisites (do once, see top [README](../README.md))

1. Build the shared venv `~/.venv_ai` on the **login node**.
2. Stage the data on `$SCRATCH`:
   ```bash
   python common/prepare_data.py     # writes $SCRATCH/moons-tutorial/data
   ```

## Submit

Edit `job.sh` and replace `def-YOURPI` (and optionally `YOUR_EMAIL`), then:

```bash
cd 01-single-job-sbatch-cpu
sbatch job.sh
```

`sbatch` prints a job id. The job waits in the queue, then runs on a compute node.

## What each `#SBATCH` line does

`job.sh` is commented line-by-line. The essentials for *any* job:

- `--account` — **required**; your PI's allocation (`def-YOURPI`).
- `--time` — wall-clock limit; the job is killed when it expires.
- `--cpus-per-task`, `--mem` — the resources you're asking for.
- `--output` — where stdout/stderr are written (`slurm-<name>-<id>.out`).

## Monitor and collect

```bash
sq                       # your queued/running jobs (Alliance shortcut)
squeue -u $USER          # same, standard Slurm
sacct -j <jobid>         # accounting once it finishes
cat slurm-moons-cpu-<jobid>.out
```

**Output lands in** `$SCRATCH/moons-tutorial/results/moons-cpu-<jobid>/`
(`metrics.json`, `model.pt`, and a `runtime/` folder with `train.log` +
`checkpoint.pt`). Final test accuracy ≈ 0.85–0.90.

Next: [02 — add a GPU](../02-single-job-sbatch-gpu/).
