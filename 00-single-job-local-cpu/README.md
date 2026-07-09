# 00 — Single-job, local CPU (no Slurm)

**Type:** single-job. **Where it runs:** your laptop (or a login node), on CPU, with no Slurm at all.

Start here. Before you touch the cluster, prove to yourself that the training
script works on your own machine. Everything you learn here — the data folder,
the output folder, the runtime folder that enables resume — works *identically*
on the cluster; the only thing the later examples add is Slurm to launch the job
on a compute node.

## 1. Make a local environment

Use the same venv location as on the cluster — `~/.venv_ai` — so your commands
are identical everywhere. On a laptop it's an ordinary virtualenv (you have
internet, so no `--no-index`):

```bash
python3 -m venv ~/.venv_ai
source ~/.venv_ai/bin/activate
pip install torch scikit-learn numpy
```

> On the cluster you build this same `~/.venv_ai` once on the login node with
> `--no-index` — see the top-level [README](../README.md#one-time-setup).

## 2. Prepare the data

`make_moons` is synthetic, but we still stage it to disk first (this mirrors the
cluster pattern where data is prepared on the login node):

```bash
python ../common/prepare_data.py --data-dir /tmp/moons/data
```

## 3. Train

```bash
python ../common/train.py \
    --device cpu \
    --data-dir /tmp/moons/data \
    --out-dir /tmp/moons/out \
    --checkpoint-dir /tmp/moons/runtime \
    --epochs 50 --lr 0.05
```

**Expected output:** one line per epoch (`epoch N | train_loss ... | test_acc ...`)
and a final `final test accuracy = ...` around 0.85–0.90. You get:

- `/tmp/moons/out/metrics.json` — final metrics
- `/tmp/moons/out/model.pt` — trained weights
- `/tmp/moons/runtime/train.log` — one CSV line per epoch
- `/tmp/moons/runtime/checkpoint.pt` — latest checkpoint

## 4. See resume work locally

Kill it with `Ctrl-C` partway through, then re-run the **exact same command**.
It prints `[resume] ... resuming from epoch N` and continues where it left off,
appending to `train.log`. This is the same mechanism example
[07](../07-single-job-checkpoint-resume/) demonstrates against a real Slurm kill.

Once this works locally, move on to [01 — single-job sbatch CPU](../01-single-job-sbatch-cpu/).
