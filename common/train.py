"""Train + test the tiny MoonsNet, with checkpoint/resume built in.

This ONE script is used by every example (00-07). The numbered folders only
contain launch files (job.sh / config.yaml) that call this script with
different arguments — there is a single source of truth for the model and the
training loop.

Key features a beginner should notice:

* Device handling: `--device auto` (default) picks CUDA if available, else CPU.
  You can force `--device cpu` or `--device cuda`; if you ask for cuda but none
  is available, we warn and fall back to CPU. The SAME script therefore runs on
  a laptop, a CPU job, or a GPU job with no code changes.

* Runtime folder (checkpoint + log): see `--checkpoint-dir`. Every N epochs we
  save `checkpoint.pt` (model + optimizer + epoch + RNG state) and we append one
  line per epoch to `train.log`. If a checkpoint already exists when the script
  starts, we RESUME from it instead of starting over. This is what makes a job
  survive a wall-time kill, preemption, node failure, or your own `scancel`.

* Outputs: final metrics go to `<out-dir>/metrics.json` and the trained weights
  to `<out-dir>/model.pt`.

Run `python common/train.py --help` for all options.
"""

import argparse
import csv
import json
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# Make `common` importable whether run as `python common/train.py` or from
# inside a numbered folder as `python ../common/train.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MoonsNet  # noqa: E402


# Set by the SIGTERM handler so the training loop can checkpoint then exit
# cleanly when Slurm (or the user) asks the job to stop.
_STOP_REQUESTED = False


def _handle_sigterm(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("[signal] SIGTERM received — will checkpoint after this epoch and exit.",
          flush=True)


def setup_comet(args):
    """Return a Comet experiment, or None if Comet is disabled/unavailable.

    Online mode needs internet — on Narval compute nodes that means loading the
    `httpproxy` module in job.sh first. Offline mode (recommended default) writes
    an archive you `comet upload` from the login node afterward. The API key is
    read from the COMET_API_KEY env var / ~/.comet.config, never hardcoded.
    """
    if not args.comet:
        return None
    try:
        import comet_ml
    except ImportError:
        print("[warn] --comet set but comet_ml is not installed; skipping Comet logging.",
              flush=True)
        return None

    common = dict(project_name=args.comet_project)
    if args.comet_offline_dir:
        os.makedirs(args.comet_offline_dir, exist_ok=True)
        exp = comet_ml.OfflineExperiment(offline_directory=args.comet_offline_dir, **common)
        print(f"[comet] OFFLINE mode; archive -> {args.comet_offline_dir}", flush=True)
    else:
        exp = comet_ml.Experiment(**common)  # ONLINE; key from env / ~/.comet.config
        print("[comet] ONLINE mode.", flush=True)
    exp.log_parameters({"lr": args.lr, "epochs": args.epochs,
                        "hidden": args.hidden, "seed": args.seed})
    return exp


def resolve_device(requested: str) -> torch.device:
    """Turn a --device string into a real torch.device, with CPU fallback."""
    if requested in ("auto", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] --device cuda requested but no GPU is visible; using CPU.",
              flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def load_split(data_dir: str, name: str, device: torch.device):
    path = os.path.join(data_dir, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run `python common/prepare_data.py --data-dir {data_dir}` "
            "first (on the LOGIN node when on the cluster)."
        )
    d = np.load(path)
    X = torch.from_numpy(d["X"]).to(device)
    y = torch.from_numpy(d["y"]).to(device)
    return X, y


def evaluate(model, X, y) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item()


def save_checkpoint(path: str, model, optimizer, epoch: int):
    """Atomically save model + optimizer + epoch + RNG state."""
    tmp = path + ".tmp"
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        ckpt["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(ckpt, tmp)
    os.replace(tmp, path)  # atomic: a kill mid-write can't corrupt checkpoint.pt


def maybe_resume(path: str, model, optimizer, device) -> int:
    """If a checkpoint exists, load it and return the next epoch to run."""
    if not os.path.exists(path):
        return 0
    # weights_only=False because our checkpoint also stores numpy/torch RNG
    # state (not just tensors). Safe here: we wrote this file ourselves.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    torch.set_rng_state(ckpt["torch_rng_state"])
    np.random.set_state(ckpt["numpy_rng_state"])
    if "cuda_rng_state" in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    next_epoch = ckpt["epoch"] + 1
    print(f"[resume] Found checkpoint at {path}; resuming from epoch {next_epoch}.",
          flush=True)
    return next_epoch


def main():
    parser = argparse.ArgumentParser(description="Train MoonsNet with checkpoint/resume.")
    parser.add_argument("--data-dir", required=True,
                        help="Folder containing train.npz / test.npz (from prepare_data.py).")
    parser.add_argument("--out-dir", required=True,
                        help="Where to write metrics.json and model.pt.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Runtime folder for train.log + checkpoint.pt. "
                             "Defaults to <out-dir>/runtime. This folder is what "
                             "lets a killed job resume.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="auto = cuda if available else cpu.")
    parser.add_argument("--checkpoint-every", type=int, default=2,
                        help="Save a checkpoint every N epochs.")
    parser.add_argument("--resume", action="store_true",
                        help="(Default behaviour anyway.) Explicitly resume from a "
                             "checkpoint if one exists in --checkpoint-dir.")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Ignore any existing checkpoint and start fresh.")
    parser.set_defaults(resume=True)
    # --- Comet.ml (optional, used by example 06) ---------------------------
    parser.add_argument("--comet", action="store_true",
                        help="Log training curves to Comet.ml (needs comet_ml installed).")
    parser.add_argument("--comet-offline-dir", default=None,
                        help="If set, use Comet OFFLINE mode and write the experiment "
                             "archive here (e.g. on $SCRATCH), to upload later from the "
                             "login node. If unset, --comet uses ONLINE mode.")
    parser.add_argument("--comet-project", default="moons-tutorial",
                        help="Comet project name.")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    device = resolve_device(args.device)
    print(f"[info] Using device: {device} "
          f"(torch.cuda.is_available()={torch.cuda.is_available()})", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    runtime_dir = args.checkpoint_dir or os.path.join(args.out_dir, "runtime")
    os.makedirs(runtime_dir, exist_ok=True)
    ckpt_path = os.path.join(runtime_dir, "checkpoint.pt")
    log_path = os.path.join(runtime_dir, "train.log")

    # Seed everything for reproducibility (RNG state is later restored on resume).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X_train, y_train = load_split(args.data_dir, "train", device)
    X_test, y_test = load_split(args.data_dir, "test", device)

    model = MoonsNet(in_features=X_train.shape[1], hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch = maybe_resume(ckpt_path, model, optimizer, device) if args.resume else 0
    if not args.resume and os.path.exists(ckpt_path):
        print("[info] --no-resume set; ignoring existing checkpoint.", flush=True)

    experiment = setup_comet(args)

    # Append if resuming (log already has earlier epochs), else start a fresh log.
    log_is_new = start_epoch == 0 or not os.path.exists(log_path)
    log_file = open(log_path, "a" if start_epoch > 0 else "w", newline="")
    writer = csv.writer(log_file)
    if log_is_new:
        writer.writerow(["epoch", "train_loss", "test_acc", "timestamp"])
        log_file.flush()

    if start_epoch >= args.epochs:
        print(f"[info] Checkpoint epoch ({start_epoch}) already >= --epochs "
              f"({args.epochs}); nothing to do.", flush=True)

    test_acc = evaluate(model, X_test, y_test)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        test_acc = evaluate(model, X_test, y_test)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        # Log to BOTH stdout (ends up in the Slurm .out file, flushed periodically)
        # and train.log (flushed every epoch, so it survives a sudden kill).
        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | test_acc {test_acc:.4f}",
              flush=True)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{test_acc:.6f}", ts])
        log_file.flush()

        # Comet wants metrics logged at low frequency (~<=1/min for long runs) to
        # avoid throttling; for this tiny model per-epoch is fine.
        if experiment is not None:
            experiment.log_metric("train_loss", train_loss, step=epoch)
            experiment.log_metric("test_acc", test_acc, step=epoch)

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(ckpt_path, model, optimizer, epoch)

        if _STOP_REQUESTED:
            save_checkpoint(ckpt_path, model, optimizer, epoch)
            print(f"[signal] Checkpointed at epoch {epoch}; exiting so the job can "
                  "requeue/resume later.", flush=True)
            log_file.close()
            sys.exit(0)

    log_file.close()

    # Persist final artefacts for downstream inspection / copying back with scp.
    metrics = {
        "final_test_accuracy": test_acc,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden": args.hidden,
        "seed": args.seed,
        "device": str(device),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))

    if experiment is not None:
        experiment.log_metric("final_test_accuracy", test_acc)
        experiment.end()

    print(f"[done] final test accuracy = {test_acc:.4f}", flush=True)
    print(f"[done] metrics -> {os.path.join(args.out_dir, 'metrics.json')}", flush=True)
    print(f"[done] model   -> {os.path.join(args.out_dir, 'model.pt')}", flush=True)


if __name__ == "__main__":
    main()
