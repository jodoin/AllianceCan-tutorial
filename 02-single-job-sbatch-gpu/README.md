# 02 — Single-job, sbatch, GPU

**Type:** single-job. **Runs on:** a Narval GPU compute node (one A100 40 GB).

Identical to [01](../01-single-job-sbatch-cpu/) except it requests a GPU and
runs the model on it. For a model this tiny the GPU is not actually faster —
the point is to learn the **GPU request syntax**, which is what you'll reuse for
real models.

## What changed vs 01

| | 01 (CPU) | 02 (GPU) |
|---|---|---|
| GPU directive | — | `#SBATCH --gpus-per-node=1` |
| CPUs | `--cpus-per-task=1` | `--cpus-per-task=4` |
| Memory | `--mem=2G` | `--mem=16G` |
| `train.py` device | `--device cpu` | `--device cuda` |

The job also runs `nvidia-smi` and `torch.cuda.is_available()` so you can *see*
the GPU in the `.out` file.

> **Narval GPU facts:** 4× NVIDIA A100 (40 GB) per node, 48 CPU cores, ~510 GB
> RAM — so roughly 12 cores / 124 GB per GPU. Ask for one GPU with
> `--gpus-per-node=1`. See the Alliance
> [Narval](https://docs.alliancecan.ca/wiki/Narval/en) and
> [Using GPUs with Slurm](https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm) docs.

## Submit

```bash
cd 02-single-job-sbatch-gpu
sbatch job.sh          # after replacing def-YOURPI
```

## Expected output

In `slurm-moons-gpu-<jobid>.out` you should see an `nvidia-smi` table, then
`cuda available: True`, then `[info] Using device: cuda ...`, then the per-epoch
lines. Results land in `$SCRATCH/moons-tutorial/results/moons-gpu-<jobid>/`.

Next: [03 — launch many jobs with a job array](../03-multi-jobs-sbatch-array/).
