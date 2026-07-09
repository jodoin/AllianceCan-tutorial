"""Checkpoint/resume ("recovery"), isolated from the training loop (example 07).

Jobs get killed — wall-time limit, preemption, node failure, or your own
`scancel`. Recovery makes a run resumable via a **runtime folder** that outlives
any single job (on the cluster, on $SCRATCH):

    <checkpoint-dir>/
    ├── train.log       # one CSV line per epoch, flushed every epoch
    └── checkpoint.pt   # model + optimizer + epoch + RNG state, saved every N epochs

Recovery is OPT-IN: it does nothing unless `--checkpoint-dir` is given. Examples
00-06 leave it off (plain training); only 07-single-job-checkpoint-resume turns
it on. `train.py` always calls into a `Recovery` object; when disabled, every
method is a no-op, so the training loop stays clean.
"""

import csv
import os
import signal
import time

import numpy as np
import torch


def add_arguments(parser):
    """Register recovery CLI flags on an argparse parser."""
    group = parser.add_argument_group("recovery (checkpoint/resume, example 07)")
    group.add_argument("--checkpoint-dir", default=None,
                       help="Runtime folder for train.log + checkpoint.pt. If omitted, "
                            "checkpoint/resume is DISABLED (examples 00-06). Set it to "
                            "enable resume-after-kill (example 07).")
    group.add_argument("--checkpoint-every", type=int, default=2,
                       help="Save a checkpoint every N epochs (needs --checkpoint-dir).")
    group.add_argument("--resume", action="store_true",
                       help="(Default when --checkpoint-dir is set.) Resume from an "
                            "existing checkpoint if one is present.")
    group.add_argument("--no-resume", dest="resume", action="store_false",
                       help="Ignore any existing checkpoint and start fresh.")
    parser.set_defaults(resume=True)


class Recovery:
    """Runtime-folder checkpoint/resume + per-epoch log. No-ops when disabled."""

    def __init__(self, checkpoint_dir=None, checkpoint_every=2, resume=True):
        self.enabled = checkpoint_dir is not None
        self.every = checkpoint_every
        self.resume = resume
        self._stop_requested = False
        self._log_file = None
        self._writer = None
        if self.enabled:
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            self.log_path = os.path.join(checkpoint_dir, "train.log")

    @classmethod
    def from_args(cls, args):
        return cls(args.checkpoint_dir, args.checkpoint_every, args.resume)

    # --- signal handling (the "automatic" preemption path) ----------------
    def install_signal_handler(self):
        """Trap SIGTERM so we can checkpoint before Slurm kills the job."""
        if self.enabled:
            signal.signal(signal.SIGTERM, self._on_sigterm)

    def _on_sigterm(self, signum, frame):
        self._stop_requested = True
        print("[signal] SIGTERM received — will checkpoint after this epoch and exit.",
              flush=True)

    def stop_requested(self) -> bool:
        return self._stop_requested

    # --- resume -----------------------------------------------------------
    def resume_epoch(self, model, optimizer, device) -> int:
        """Return the next epoch to run: 0 fresh, or epoch+1 from a checkpoint."""
        if not self.enabled:
            return 0
        if not self.resume:
            if os.path.exists(self.ckpt_path):
                print("[info] --no-resume set; ignoring existing checkpoint.", flush=True)
            return 0
        if not os.path.exists(self.ckpt_path):
            return 0
        # weights_only=False because the checkpoint also stores numpy/torch RNG
        # state (not just tensors). Safe: we wrote this file ourselves.
        ckpt = torch.load(self.ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        np.random.set_state(ckpt["numpy_rng_state"])
        if "cuda_rng_state" in ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        next_epoch = ckpt["epoch"] + 1
        print(f"[resume] Found checkpoint at {self.ckpt_path}; resuming from epoch "
              f"{next_epoch}.", flush=True)
        return next_epoch

    # --- per-epoch log ----------------------------------------------------
    def open_log(self, start_epoch: int):
        if not self.enabled:
            return
        is_new = start_epoch == 0 or not os.path.exists(self.log_path)
        # Append when resuming (keep earlier epochs); else start a fresh log.
        self._log_file = open(self.log_path, "a" if start_epoch > 0 else "w", newline="")
        self._writer = csv.writer(self._log_file)
        if is_new:
            self._writer.writerow(["epoch", "train_loss", "test_acc", "timestamp"])
            self._log_file.flush()

    def log_epoch(self, epoch: int, train_loss: float, test_acc: float):
        if not self.enabled:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self._writer.writerow([epoch, f"{train_loss:.6f}", f"{test_acc:.6f}", ts])
        self._log_file.flush()  # flush every epoch so a sudden kill can't lose it

    def close_log(self):
        if self._log_file is not None:
            self._log_file.close()

    # --- checkpointing ----------------------------------------------------
    def _save(self, model, optimizer, epoch: int):
        tmp = self.ckpt_path + ".tmp"
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
        os.replace(tmp, self.ckpt_path)  # atomic: a kill mid-write can't corrupt it

    def checkpoint_if_due(self, model, optimizer, epoch: int, total_epochs: int):
        if self.enabled and ((epoch + 1) % self.every == 0 or epoch == total_epochs - 1):
            self._save(model, optimizer, epoch)

    def checkpoint_now(self, model, optimizer, epoch: int):
        if self.enabled:
            self._save(model, optimizer, epoch)
