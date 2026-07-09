# 00b — Single-job, local CPU: checkpoint & resume

**Type:** single-job. **Where it runs:** your laptop, on CPU, no Slurm.

Same training run as [00a](../00a-single-job-local-cpu/), with one idea added:
**recovery** — making a run survive being killed and pick up where it left off.
On the cluster a job dies for many reasons (wall-time limit, preemption, node
failure, your own `scancel`); here you'll simulate that with `Ctrl-C` on your
laptop. The mechanism is identical — example
[07](../07-single-job-checkpoint-resume/) is this exact feature under real Slurm.

Prerequisites: the venv and data from [00a](../00a-single-job-local-cpu/)
(`~/.venv_ai` activated, data staged at `/tmp/moons/data`).

## How recovery works

Recovery is **opt-in**: it turns on only when you pass `--checkpoint-dir`. That
folder — the "runtime folder" — outlives any single run and holds:

- `train.log` — one CSV line per epoch, flushed **every** epoch (so progress
  survives a sudden kill; stdout can be lost).
- `checkpoint.pt` — model + optimizer + epoch + RNG state, saved every N epochs
  (`--checkpoint-every`, default 2), written atomically.

On startup `train.py` checks the runtime folder for `checkpoint.pt`; if it's
there, it resumes from `epoch + 1` and **appends** to `train.log` instead of
starting over.

## 1. Start a run with a runtime folder

Use lots of epochs so you have time to interrupt it:

```bash
source ~/.venv_ai/bin/activate
python ../common/train.py \
    --device cpu \
    --data-dir /tmp/moons/data \
    --out-dir /tmp/moons/out \
    --checkpoint-dir /tmp/moons/runtime \
    --epochs 2000 --lr 0.01 --checkpoint-every 50
```

## 2. Kill it partway through

Press **`Ctrl-C`** after a few hundred epochs. Look at what's already on disk:

```bash
tail /tmp/moons/runtime/train.log      # epochs logged so far
ls   /tmp/moons/runtime/               # train.log + checkpoint.pt
```

## 3. Re-run the exact same command → it resumes

```bash
python ../common/train.py \
    --device cpu \
    --data-dir /tmp/moons/data \
    --out-dir /tmp/moons/out \
    --checkpoint-dir /tmp/moons/runtime \
    --epochs 2000 --lr 0.01 --checkpoint-every 50
```

You'll see `[resume] Found checkpoint ... resuming from epoch N` and training
continues from where it stopped; `train.log` keeps growing rather than being
overwritten.

> To force a fresh start instead of resuming, add `--no-resume` (it ignores any
> existing checkpoint).

Next: [00c — track curves with Comet.ml, locally](../00c-single-job-local-cpu/).
