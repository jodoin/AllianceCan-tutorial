# 03 — Multi-jobs, sbatch job array

**Type:** multi-jobs (one submission → several Slurm jobs). **Runs on:** Narval CPU nodes.

When you want to run the same script many times with different settings — sweep
a seed, a learning rate, a data fold — a **job array** is the right tool. One
`sbatch job.sh` launches N independent jobs; the scheduler runs them whenever
resources are free (often in parallel).

## How it works

- `#SBATCH --array=0-4` creates **5 tasks**, indexed 0 through 4.
- Inside each task, `$SLURM_ARRAY_TASK_ID` holds that task's index.
- We use it as the `--seed`, so the five jobs train with seeds 0,1,2,3,4, each
  writing to its own subfolder:
  `$SCRATCH/moons-tutorial/results/moons-array-<arrayid>/seed-<n>/`.

To sweep a **learning rate** instead, map the index to a value, e.g.:

```bash
LRS=(0.001 0.005 0.01 0.05 0.1)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
```

## Submit

```bash
cd 03-multi-jobs-sbatch-array
sbatch job.sh          # after replacing def-YOURPI
```

You get **one** array job id (e.g. `12345`); the tasks appear as `12345_0` …
`12345_4`. Output files are `slurm-moons-array-12345_0.out`, etc.

## Make it a GPU array

Add the GPU directives from [02](../02-single-job-sbatch-gpu/) to this script —
`#SBATCH --gpus-per-node=1`, bump `--cpus-per-task`/`--mem`, and switch the run
to `--device cuda`. Each array task then gets its own GPU. (Be mindful this
requests 5 GPUs total across the array.)

## Monitor

```bash
sq                        # shows the array tasks
sacct -j <arrayid>        # per-task accounting
```

> Job arrays are the **recommended way to launch many jobs on Narval**. Example
> [05](../05-multi-jobs-hydra-submitit/) shows a fancier Hydra+submitit sweep,
> but arrays are simpler and always available. See the Alliance
> [job arrays](https://docs.alliancecan.ca/wiki/Job_arrays) doc.

Next: [04 — config-driven runs with Hydra](../04-single-job-hydra/).
