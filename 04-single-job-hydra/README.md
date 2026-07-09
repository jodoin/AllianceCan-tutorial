# 04 — Single-job, Hydra config

**Type:** single-job. **Runs on:** a Narval GPU node (or CPU, see below).

[Hydra](https://hydra.cc) moves your hyperparameters into a `config.yaml` and
lets you override any of them on the command line — no editing Python. This is
the foundation for the sweeps in [05](../05-multi-jobs-hydra-submitit/).

Same `common/train.py` as before; the only new piece is the thin
`common/train_hydra.py` wrapper, which reads the config and calls straight into
`train.py`. There is still a single source of truth for the model.

## The config

[`config.yaml`](config.yaml) holds `epochs`, `lr`, `seed`, `hidden`, and an
explicit `device` field. `data_dir`/`out_dir` are left as `???` and filled in by
`job.sh` so they point at `$SCRATCH` at run time.

## Overriding on the command line

```bash
# change one value
python ../common/train_hydra.py --config-dir "$PWD" data_dir=/tmp/moons/data out_dir=/tmp/out lr=0.1

# force CPU vs GPU
python ../common/train_hydra.py --config-dir "$PWD" data_dir=... out_dir=... device=cpu
python ../common/train_hydra.py --config-dir "$PWD" data_dir=... out_dir=... device=cuda
```

## ⚠️ The one thing beginners forget

The Slurm resource request and the Hydra `device` are **two separate places**:

- `config.yaml` → `device: cuda`
- `job.sh`      → `#SBATCH --gpus-per-node=1`

**They must agree.** `device=cuda` without a GPU request means the job runs on
CPU anyway (train.py falls back and warns); `--gpus-per-node=1` with
`device=cpu` wastes a GPU. To run this example on CPU: set `device=cpu` in the
config (or `device=cpu` on the CLI) **and** delete the `--gpus-per-node=1` line
in `job.sh`.

## Submit

```bash
cd 04-single-job-hydra
sbatch job.sh          # after replacing def-YOURPI
```

Results land in `$SCRATCH/moons-tutorial/results/moons-hydra-<jobid>/`. Hydra
also writes its own run log under `outputs/` in this folder.

Next: [05 — sweep many jobs with Hydra + submitit](../05-multi-jobs-hydra-submitit/).
