# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A beginner-oriented **tutorial** teaching how to launch PyTorch training jobs on
the Alliance Canada (Digital Research Alliance of Canada) cluster **Narval**. The
ML is deliberately trivial (a 2-layer logistic regression on `make_moons`) so the
focus stays on *launching jobs*. It is not a library — it's runnable teaching
material, so clarity and correct cluster mechanics matter more than cleverness.

## Architecture: one source of truth + launch-only folders

- **`common/` holds the ONLY copy of the model and training code.** Do not
  duplicate model/training logic into the numbered folders.
  - `model.py` — `MoonsNet` (Linear→ReLU→Linear, 2 classes).
  - `prepare_data.py` — generates `make_moons`, writes `train.npz`/`test.npz`.
    Meant to run on the **login node** (compute nodes have no internet).
  - `train.py` — the CLI training loop + device selection. Composes the two
    optional modules below; all examples call this.
  - `recovery.py` — checkpoint/resume ("recovery"). **Opt-in via
    `--checkpoint-dir`**; a `Recovery` object no-ops when disabled. Turned on by
    **00b** (local intro) and **07** (on-cluster); every other example does plain
    training. Also owns the per-epoch `train.log` and the SIGTERM handler.
  - `comet.py` — optional Comet.ml logging. **Opt-in via `--comet`**; a
    `CometLogger` no-ops when disabled or if `comet_ml` isn't installed. Turned
    on by **00c** (local intro) and **06** (on-cluster).
  - `train_hydra.py` — thin Hydra wrapper that translates a `config.yaml` into
    `train.py`'s argparse args and calls `train.main()`. Used by 04 and 05.
  - Both optional modules expose `add_arguments(parser)` so `train.py` stays the
    single place that builds the CLI. Keep the training loop free of
    `if enabled:` checks — the no-op objects handle that.
- **Each numbered folder contains only a `README.md` plus its launch files**
  (`job.sh` and/or `config.yaml`). The `00a`/`00b`/`00c` folders are local
  (no-Slurm) walkthroughs, so README-only. Every `job.sh` calls
  `../common/train.py`. When adding/changing behaviour, edit `common/`, not the
  folders.
- Terminology is deliberate: **"single-job"** = one Slurm job per submission;
  **"multi-jobs"** = one submission launches several (examples 03 and 05). Keep
  this wording consistent in folder names, prose, and READMEs.

## The examples

Local (no Slurm): 00a plain · 00b recovery · 00c Comet. Cluster: 01 sbatch CPU ·
02 sbatch GPU · 03 job **array** (multi-jobs) · 04 Hydra config · 05 Hydra
`--multirun` + submitit (multi-jobs) ·
06 Comet.ml · 07 checkpoint/resume.

## Narval mechanics that must stay correct

These are the whole point of the tutorial — don't regress them:

- **`--account=def-YOURPI` is required** on every job; it's an obvious UPPERCASE
  placeholder the reader replaces (as are `YOUR_EMAIL`, `YOUR_COMET_API_KEY`).
- **Compute nodes have no internet.** The env is built **once on the login node**
  at `~/.venv_ai` from the wheelhouse (`pip install --no-index ...`); job scripts
  only *activate* it (`module load python/3.11 && source ~/.venv_ai/bin/activate`).
  Never install packages inside a `job.sh`.
- **Storage:** `$HOME` (~50 GB, not purged) → venv only. `$SCRATCH` (large,
  purged ~60 days on Narval) → data, results, runtime/checkpoints.
- **GPU request syntax:** `#SBATCH --gpus-per-node=1` (Narval = A100 40 GB, ~12
  cores / ~124 GB per GPU). The **Hydra `device` field and the Slurm GPU request
  are two separate places that must stay in sync** (called out in 04/05).
- **Comet (06):** compute nodes are offline, so the default is **offline mode**
  (write archive to `$SCRATCH`, `comet upload` from the login node afterward);
  **online mode requires `module load httpproxy`**. API key comes from
  `$COMET_API_KEY`/`~/.comet.config`, never hardcoded.
- **Checkpoint/resume (07 only):** the **runtime-folder pattern**, implemented in
  `recovery.py` and enabled only when `--checkpoint-dir` is passed. A folder on
  `$SCRATCH` keyed on the **job name** (stable across resubmissions) holds
  `train.log` (appended + flushed every epoch) and `checkpoint.pt` (model +
  optimizer + epoch + RNG state, saved every N epochs, written atomically). On
  startup `Recovery.resume_epoch` resumes from the checkpoint if present. `job.sh` adds
  `#SBATCH --requeue` + `--signal=B:TERM@30` + a SIGTERM trap for the automatic
  (preemption) path; `recovery.py`'s SIGTERM handler checkpoints then exits.
- Where a Narval detail is uncertain (exact wheel availability, purge window,
  Comet policy, requeue behaviour) the READMEs point to
  <https://docs.alliancecan.ca>. Prefer verifying there over guessing. Some
  wheels (`hydra-submitit-launcher`, `comet-ml`) may be absent — the READMEs tell
  readers to `avail_wheels <name>` and fall back (05 → job array; 06 →
  `--find-links`).

## Running locally (how to verify changes)

`train.py`/`train_hydra.py` run unchanged on a laptop (CPU, local folders). To
test without a cluster:

```bash
python3 -m venv /tmp/venv && /tmp/venv/bin/pip install torch scikit-learn numpy hydra-core
/tmp/venv/bin/python common/prepare_data.py --data-dir /tmp/moons/data
/tmp/venv/bin/python common/train.py --data-dir /tmp/moons/data --out-dir /tmp/moons/out \
    --checkpoint-dir /tmp/moons/rt --epochs 12 --lr 0.05
# resume: rerun with --epochs 24; it resumes from epoch 12 and appends to train.log
```

`torch.load(..., weights_only=False)` is used on purpose in
`recovery.Recovery.resume_epoch` (the checkpoint stores numpy/torch RNG state,
and we wrote the file ourselves).
