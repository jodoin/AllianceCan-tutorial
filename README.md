# Launching PyTorch training jobs on Narval (Alliance Canada)

A beginner-friendly, runnable tutorial for launching PyTorch training jobs on a
[Digital Research Alliance of Canada](https://docs.alliancecan.ca) cluster,
using **Narval** as the concrete example. No prior HPC/Slurm experience assumed.

The machine-learning part is deliberately trivial — a **2-layer logistic
regression** on the `make_moons` toy dataset — so that all your attention goes
to the thing that's actually new: **how you launch jobs on a cluster**. Every
example trains the exact same tiny model; what changes is *how it's launched*.

## The examples (run them in order)

Three local warm-ups (`00a`/`00b`/`00c`, no Slurm) then seven cluster examples:

| # | Folder | Single- or multi-jobs | Teaches |
|---|--------|----------------------|---------|
| 00a | [`00a-single-job-local-cpu`](00a-single-job-local-cpu/) | single-job | Run it on your laptop first, no Slurm |
| 00b | [`00b-single-job-local-cpu`](00b-single-job-local-cpu/) | single-job | Checkpoint & resume, locally |
| 00c | [`00c-single-job-local-cpu`](00c-single-job-local-cpu/) | single-job | Comet.ml logging, locally |
| 01 | [`01-single-job-sbatch-cpu`](01-single-job-sbatch-cpu/) | single-job | The minimal `sbatch` script (CPU) |
| 02 | [`02-single-job-sbatch-gpu`](02-single-job-sbatch-gpu/) | single-job | Add a GPU (`--gpus-per-node=1`) |
| 03 | [`03-multi-jobs-sbatch-array`](03-multi-jobs-sbatch-array/) | multi-jobs | Job **array**: many jobs from one submit |
| 04 | [`04-single-job-hydra`](04-single-job-hydra/) | single-job | Config-driven runs with Hydra |
| 05 | [`05-multi-jobs-hydra-submitit`](05-multi-jobs-hydra-submitit/) | multi-jobs | Hydra `--multirun` sweep via submitit |
| 06 | [`06-single-job-comet`](06-single-job-comet/) | single-job | Training curves with Comet.ml |
| 07 | [`07-single-job-checkpoint-resume`](07-single-job-checkpoint-resume/) | single-job | Survive a kill and resume |

> **"single-job" vs "multi-jobs"** is used consistently throughout: *single-job*
> means one Slurm job comes out of one submission; *multi-jobs* means one
> submission launches several jobs (03 and 05).

## Repository layout

```
.
├── README.md                      <- you are here
├── requirements.txt
├── CLAUDE.md
├── common/                        <- the ONLY place the model/training code lives
│   ├── model.py                   <- the 2-layer logistic regression
│   ├── prepare_data.py            <- make_moons -> .npz (run on the LOGIN node)
│   ├── train.py                   <- CLI training loop (composes the two below)
│   ├── recovery.py                <- opt-in checkpoint/resume, used by 07
│   ├── comet.py                   <- opt-in Comet.ml logging, used by 06
│   └── train_hydra.py             <- thin Hydra wrapper around train.py (04, 05)
├── 00a/ 00b/ 00c/                 <- local (no-Slurm) intros: plain, recovery, comet
└── 01.. 07/                       <- cluster examples: a README + launch files
```

The numbered folders contain **only launch files** (`job.sh` / `config.yaml`) and
a README (the 00 folders are local walkthroughs, so README only). Every `job.sh`
calls `../common/train.py`, so there is a single source of truth for the model.

---

## Step 1 — Log in

```bash
ssh USER@narval.alliancecan.ca      # replace USER with your Alliance username
```

## Step 2 — Clone this repo

Clone it into your `$HOME` (or `$PROJECT`):

```bash
git clone <this-repo-url> AllianceCan-tutorial
cd AllianceCan-tutorial
```

## Step 3 — One-time setup: build `~/.venv_ai` on the login node

**This is the single most important cluster concept in this tutorial:** compute
nodes have **no internet**. So you install everything **once, on the login
node**, into a persistent virtual environment, using the Alliance **wheelhouse**
(`--no-index` = "don't reach out to PyPI, use the local prebuilt wheels").

```bash
module load python/3.11
virtualenv --no-download ~/.venv_ai
source ~/.venv_ai/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch scikit-learn numpy
# optional, only for the examples that use them (check first with avail_wheels):
pip install --no-index hydra-core hydra-submitit-launcher   # examples 04, 05
pip install --no-index comet-ml                             # example 06
```

Every `job.sh` then just does `module load python/3.11 && source
~/.venv_ai/bin/activate` — it only **activates** the venv, installs nothing, and
so needs no internet at run time.

> **Why `~/.venv_ai` in `$HOME`?** `$HOME` is small (~50 GB) but **not purged**,
> which is exactly right for a venv you build once and keep. It's the *wrong*
> place for data or results (too small, and you don't want big files there).
>
> Check what's in the wheelhouse with `avail_wheels <name>` (e.g. `avail_wheels
> torch`). If a package is missing, request it at `support@tech.alliancecan.ca`.
> Docs: [Python](https://docs.alliancecan.ca/wiki/Python) ·
> [Available wheels](https://docs.alliancecan.ca/wiki/Available_Python_wheels).

## Step 4 — Stage the data on `$SCRATCH`

Run the data prep **on the login node** (this is where you'd download a real
dataset, since compute nodes can't):

```bash
python common/prepare_data.py       # writes to $SCRATCH/moons-tutorial/data
```

> **Use `$SCRATCH` for data and results.** It's large and fast, but **purged
> periodically (~60 days on Narval)** — so it's for temporary/large files, not
> permanent storage. Copy anything you want to keep back to your own machine
> (Step 7) or to `$PROJECT`. Docs:
> [Storage and file management](https://docs.alliancecan.ca/wiki/Storage_and_file_management).

## Step 5 — Work through examples 01 → 07

Each folder's README has the exact command to run and what to expect. Before any
`sbatch`, open the `job.sh` and replace the **UPPERCASE placeholders**:

- `def-YOURPI` → your allocation account (**required**, e.g. `def-smith`)
- `YOUR_EMAIL` → your email for notifications (optional; delete the mail lines
  if you don't want them)
- `YOUR_COMET_API_KEY` → only for example 06

## Step 6 — Monitor your jobs

```bash
sq                       # Alliance shortcut: your queued + running jobs
squeue -u $USER          # standard Slurm equivalent
sacct -j <jobid>         # accounting/details once a job has run
scancel <jobid>          # cancel a job (used in example 07)
```

## Step 7 — Get your results back

Results live under `$SCRATCH/moons-tutorial/results/...` on the cluster. Copy
them to your own machine (run these **from your laptop**):

```bash
# single folder
scp -r USER@narval.alliancecan.ca:'~/scratch/moons-tutorial/results' ./results

# or mirror with rsync
rsync -av USER@narval.alliancecan.ca:'~/scratch/moons-tutorial/results/' ./results/
```

---

## Cluster facts this tutorial relies on

- **Narval GPUs:** 4× NVIDIA **A100 (40 GB)** per node, 48 CPU cores, ~510 GB
  RAM. Request one GPU with `#SBATCH --gpus-per-node=1`.
  ([Narval docs](https://docs.alliancecan.ca/wiki/Narval/en))
- **`--account` is required** on every job (`def-YOURPI`).
- **Compute nodes have no internet** → build the venv on the login node with
  `--no-index`; use `module load httpproxy` only when a job genuinely needs
  internet (example 06 online mode).
- **`$HOME`** (~50 GB, not purged) → venv. **`$SCRATCH`** (large, purged ~60
  days) → data, results, checkpoints.

Some Narval-specific details (exact wheel availability, current purge window,
Comet's internet policy, requeue behaviour) can change — where it matters, the
per-example READMEs point at the relevant page on
<https://docs.alliancecan.ca>. When in doubt, check there.
