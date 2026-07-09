"""Train + test the tiny MoonsNet.

This ONE script is used by every example (00-07). The numbered folders only
contain launch files (job.sh / config.yaml) that call this script with
different arguments — there is a single source of truth for the model and the
training loop.

Two optional, self-contained features live in their own modules and are wired in
here without cluttering the loop:

* Checkpoint/resume — `recovery.py`. Opt-in via `--checkpoint-dir` (example 07).
  When off (examples 00-06), the Recovery object no-ops.
* Comet.ml logging — `comet.py`. Opt-in via `--comet` (example 06). When off,
  the CometLogger no-ops.

Device handling: `--device auto` (default) picks CUDA if available, else CPU.
`--device cuda` with no GPU warns and falls back to CPU, so the SAME script runs
on a laptop, a CPU job, or a GPU job with no code changes.

Run `python common/train.py --help` for all options.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Make sibling modules importable whether run as `python common/train.py` or
# from inside a numbered folder as `python ../common/train.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MoonsNet          # noqa: E402
import recovery as recovery_mod     # noqa: E402
import comet as comet_mod           # noqa: E402


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


def build_parser():
    parser = argparse.ArgumentParser(description="Train MoonsNet.")
    parser.add_argument("--data-dir", required=True,
                        help="Folder containing train.npz / test.npz (from prepare_data.py).")
    parser.add_argument("--out-dir", required=True,
                        help="Where to write metrics.json and model.pt.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="auto = cuda if available else cpu.")
    # Optional features contribute their own flags.
    recovery_mod.add_arguments(parser)
    comet_mod.add_arguments(parser)
    return parser


def main():
    args = build_parser().parse_args()

    device = resolve_device(args.device)
    print(f"[info] Using device: {device} "
          f"(torch.cuda.is_available()={torch.cuda.is_available()})", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)

    # Seed everything for reproducibility (RNG state is later restored on resume).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X_train, y_train = load_split(args.data_dir, "train", device)
    X_test, y_test = load_split(args.data_dir, "test", device)

    model = MoonsNet(in_features=X_train.shape[1], hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Optional features (both no-op unless enabled by their flags).
    recovery = recovery_mod.Recovery.from_args(args)
    recovery.install_signal_handler()
    comet = comet_mod.make_comet_logger(args)

    start_epoch = recovery.resume_epoch(model, optimizer, device)
    recovery.open_log(start_epoch)

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

        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | test_acc {test_acc:.4f}",
              flush=True)
        recovery.log_epoch(epoch, train_loss, test_acc)
        # Comet recommends low-frequency logging (~<=1/min) for long runs; per
        # epoch is fine for this tiny model.
        comet.log_metric("train_loss", train_loss, step=epoch)
        comet.log_metric("test_acc", test_acc, step=epoch)

        recovery.checkpoint_if_due(model, optimizer, epoch, args.epochs)

        if recovery.stop_requested():
            recovery.checkpoint_now(model, optimizer, epoch)
            print(f"[signal] Checkpointed at epoch {epoch}; exiting so the job can "
                  "requeue/resume later.", flush=True)
            recovery.close_log()
            comet.end()
            sys.exit(0)

    recovery.close_log()

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

    comet.log_metric("final_test_accuracy", test_acc)
    comet.end()

    print(f"[done] final test accuracy = {test_acc:.4f}", flush=True)
    print(f"[done] metrics -> {os.path.join(args.out_dir, 'metrics.json')}", flush=True)
    print(f"[done] model   -> {os.path.join(args.out_dir, 'model.pt')}", flush=True)


if __name__ == "__main__":
    main()
