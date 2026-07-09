# 05 — Multi-jobs, Hydra `--multirun` + submitit

**Type:** multi-jobs (one command → several Slurm jobs). **Runs from:** the login node.

This is the "fancy" multi-jobs option: a Hydra sweep where the
**submitit-slurm launcher** turns each swept value into its own Slurm job. You
run one command on the login node and submitit submits the jobs for you.

## ⚠️ Check the wheelhouse first

`hydra-submitit-launcher` is **not guaranteed** to be in the Alliance
wheelhouse. Before relying on this example, check:

```bash
module load python/3.11
source ~/.venv_ai/bin/activate
avail_wheels hydra-core hydra-submitit-launcher submitit
```

- **If they're available**, install into the shared venv:
  ```bash
  pip install --no-index hydra-core hydra-submitit-launcher
  ```
- **If `hydra-submitit-launcher` is NOT available** (only in the wheelhouse as a
  regular PyPI package, or missing entirely): don't fight it. Use the
  [job array in example 03](../03-multi-jobs-sbatch-array/) instead — it's the
  **recommended, always-available way to launch many jobs on Narval** and does
  the same thing. You can request it be added via `support@tech.alliancecan.ca`.

## Run the sweep

Replace `def-YOURPI` in [`config.yaml`](config.yaml) (under `hydra.launcher`),
then, **on the login node**:

```bash
cd 05-multi-jobs-hydra-submitit
bash job.sh
```

`lr=0.001,0.01,0.1` produces **three Slurm jobs**. The launcher settings in
`config.yaml` (`account`, `timeout_min`, `cpus_per_task`, `mem_gb`) become the
`#SBATCH` directives submitit uses.

## GPU sweep

In `config.yaml`: set `device: cuda` and uncomment `gpus_per_node: 1` under
`hydra.launcher`. (Same "keep device and GPU request in sync" rule as
[04](../04-single-job-hydra/), just expressed in the launcher config.)

## Monitor

```bash
sq                       # the submitted sweep jobs
```

Per-run outputs and Hydra logs land under
`$SCRATCH/moons-tutorial/results/moons-sweep/multirun/`.

> Docs: [Hydra submitit launcher](https://hydra.cc/docs/plugins/submitit_launcher/)
> and Alliance [job arrays](https://docs.alliancecan.ca/wiki/Job_arrays) for the
> fallback.

Next: [06 — track training curves with Comet.ml](../06-single-job-comet/).
