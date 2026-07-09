# 07 — Single-job checkpoint & resume

**Type:** single-job. **Runs on:** a Narval CPU node.

Jobs *will* get killed — you hit the wall-time limit, the scheduler preempts
you, a node fails, or you `scancel` on purpose. A well-behaved training job must
be able to pick up where it left off. This example demonstrates the
**runtime-folder pattern** end to end.

## The pattern

A folder on `$SCRATCH` that outlives any single job:

```
$SCRATCH/moons-tutorial/runtime/moons-resume/
├── train.log       # one CSV line per epoch, flushed every epoch
└── checkpoint.pt   # model + optimizer + epoch + RNG state, saved every N epochs
```

- `train.log` is flushed **every epoch**, so progress survives a sudden kill
  (stdout in the Slurm `.out` file only flushes periodically and is easy to
  lose).
- On startup `train.py` checks for `checkpoint.pt`; if it exists, it resumes
  from `epoch + 1` and **appends** to `train.log` instead of overwriting.
- The runtime folder is keyed on the **job name** (`moons-resume`), not the job
  id, so resubmitting the same `job.sh` reuses it.

## Hands-on demo (manual restart)

`job.sh` uses a short `#SBATCH --time=00:02:00` and 2000 epochs so it *can't*
finish in one shot.

```bash
cd 07-single-job-checkpoint-resume
sbatch job.sh                       # note the <jobid>

sq                                  # watch it start
# let it run ~20-30 s, then kill it mid-training:
scancel <jobid>

# the runtime folder is already populated on $SCRATCH:
tail $SCRATCH/moons-tutorial/runtime/moons-resume/train.log
ls   $SCRATCH/moons-tutorial/runtime/moons-resume/

# resubmit the EXACT same script:
sbatch job.sh
```

In the second job's `.out` file you'll see
`[resume] Found checkpoint ... resuming from epoch N` — it continues rather than
restarting from 0, and `train.log` keeps growing from where it stopped.

## The automatic version (`--requeue` + SIGTERM trap)

`job.sh` also wires up the *automatic* path for scheduler preemption:

- `#SBATCH --requeue` — lets Slurm put the job back in the queue if preempted.
- `#SBATCH --signal=B:TERM@30` — Slurm sends `SIGTERM` to the batch script 30 s
  before it's killed.
- A `trap term_handler TERM` in `job.sh` forwards that signal to the Python
  process; `train.py`'s SIGTERM handler saves a final checkpoint and exits
  cleanly, so the requeued run resumes with almost no lost work.

This is optional polish — the manual restart above already gives you full
resume. For the complete signal-handling reference see the Alliance
[checkpointing / points de contrôle](https://docs.alliancecan.ca/wiki/Points_de_contr%C3%B4le)
docs.

## On a laptop

The same mechanism works with no Slurm — that's exactly what
[00b](../00b-single-job-local-cpu/) walks through: run it, `Ctrl-C` partway, and
re-run — it resumes from the local runtime folder.

That's the last example — back to the top [README](../README.md).
