# 06 — Single-job with Comet.ml curves

**Type:** single-job. **Runs on:** a Narval CPU node.

[Comet.ml](https://www.comet.com) records live training curves (loss/accuracy
per epoch) to a web dashboard. The catch on Narval: **compute nodes have no
internet by default**, so there are two supported patterns. The Comet logging
itself lives in `common/train.py` behind the `--comet` flag — the folder only
holds the launch script.

## Install

```bash
module load python/3.11
source ~/.venv_ai/bin/activate
avail_wheels comet_ml comet-ml          # check what's in the wheelhouse
pip install --no-index comet-ml         # or: pip install --no-index --find-links <wheelhouse> comet_ml
```

If `comet-ml` isn't a plain wheelhouse package, use the `--find-links
<wheelhouse-path>` form; if it's missing entirely, request it via
`support@tech.alliancecan.ca`.

## API key (never hardcode it)

`train.py` reads the key from the `COMET_API_KEY` environment variable (or
`~/.comet.config`). Set it before submitting:

```bash
export COMET_API_KEY=YOUR_COMET_API_KEY
```

## Two modes

### Offline (default, recommended)

Matches the wandb-offline pattern used elsewhere on Alliance clusters. The job
writes a Comet archive to `$SCRATCH`; **no internet needed on the compute node**.
After the job finishes, upload from the **login node** (which has internet):

```bash
cd 06-single-job-comet
sbatch job.sh
# when it's done:
comet upload $SCRATCH/moons-tutorial/comet-offline/<jobid>/*.zip
```

### Online (`httpproxy`)

To stream curves live during the run, give the compute node internet with the
`httpproxy` module. Edit `job.sh`: comment out the offline block and uncomment
the online block (`module load httpproxy` + `--comet` without
`--comet-offline-dir`).

> **Why offline is the default:** Narval compute nodes are offline by design, so
> the offline-then-upload flow always works; `httpproxy` is the opt-in exception.
> See the Alliance docs on
> [Weights & Biases / offline logging](https://docs.alliancecan.ca/wiki/Weights_%26_Biases_(wandb))
> (Comet follows the same shape) and
> [httpproxy](https://docs.alliancecan.ca/wiki/Technical_documentation).

## A note on logging frequency

Comet recommends logging metrics at low frequency (≈ ≤ 1/minute) for long runs
to avoid throttling. Our model is tiny so per-epoch logging is fine; for real
training, log every N steps or every minute.

Next: [07 — checkpoint & resume after a kill](../07-single-job-checkpoint-resume/).
