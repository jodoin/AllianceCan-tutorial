# 00c — Single-job, local CPU: Comet.ml logging

**Type:** single-job. **Where it runs:** your laptop, on CPU, no Slurm.

Same training run again, with the last local idea added: **experiment tracking**
with [Comet.ml](https://www.comet.com). Comet records live training curves
(loss/accuracy per epoch) to a web dashboard. Doing it on your laptop first is
easy because a laptop **has internet** — on the cluster, compute nodes don't,
which is the whole complication that example
[06](../06-single-job-comet/) exists to solve.

Prerequisites: the venv and data from [00a](../00a-single-job-local-cpu/).

## 1. Install Comet and set your API key

```bash
source ~/.venv_ai/bin/activate
pip install comet-ml          # laptop has internet, so a normal install
```

Get your API key from your Comet account and export it — **never hardcode it**:

```bash
export COMET_API_KEY=YOUR_COMET_API_KEY
```

`train.py` reads the key from this environment variable (or `~/.comet.config`).

## 2. Train with `--comet` (online)

Recovery is off here (no `--checkpoint-dir`) so we stay focused on one new thing:

```bash
python ../common/train.py \
    --device cpu \
    --data-dir /tmp/moons/data \
    --out-dir /tmp/moons/out \
    --epochs 50 --lr 0.05 \
    --comet --comet-project moons-tutorial
```

`train.py` prints a Comet experiment URL. Open it and watch `train_loss` and
`test_acc` update per epoch. `--comet` is opt-in; without it, everything runs
exactly as in [00a](../00a-single-job-local-cpu/).

> If `comet_ml` isn't installed or `--comet` is omitted, logging is silently
> skipped — the run still trains normally.

## 3. Offline mode (a preview of the cluster pattern)

You can also write the experiment to disk and upload it later — this is the
**default on Narval** because compute nodes are offline:

```bash
python ../common/train.py --device cpu \
    --data-dir /tmp/moons/data --out-dir /tmp/moons/out \
    --epochs 50 --lr 0.05 \
    --comet --comet-offline-dir /tmp/moons/comet-offline
comet upload /tmp/moons/comet-offline/*.zip     # send it up afterward
```

On the cluster you'd write the offline archive to `$SCRATCH` inside the job and
`comet upload` from the login node. That's exactly what
[06 — single-job Comet](../06-single-job-comet/) does, plus the `httpproxy`
option for true online logging.

That's the end of the local (00) series. Move on to the cluster:
[01 — single-job sbatch CPU](../01-single-job-sbatch-cpu/).
