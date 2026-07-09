"""Thin Hydra wrapper around common/train.py (used by examples 04 and 05).

Hydra lets you drive the run from a `config.yaml` and override any field on the
command line, e.g. `python ../common/train_hydra.py lr=0.1 device=cuda`. With
`--multirun` (example 05) plus the submitit launcher, Hydra submits one Slurm
job per value in a sweep like `lr=0.001,0.01,0.1`.

This wrapper does NOT reimplement training. It translates the Hydra config into
the exact same arguments common/train.py already understands and calls into it,
so there remains a single source of truth for the model and training loop.

The config directory is passed on the command line from job.sh via
`--config-dir "$PWD"` (the numbered example folder), so this file does not hard
-code where config.yaml lives.
"""

import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as train_module  # noqa: E402


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig):
    print("[hydra] Resolved config:\n" + OmegaConf.to_yaml(cfg), flush=True)

    # Rebuild argv for common/train.py's argparse, then reuse its main().
    argv = [
        "train.py",
        "--data-dir", str(cfg.data_dir),
        "--out-dir", str(cfg.out_dir),
        "--epochs", str(cfg.epochs),
        "--lr", str(cfg.lr),
        "--seed", str(cfg.seed),
        "--hidden", str(cfg.hidden),
        "--device", str(cfg.device),
    ]
    if "checkpoint_dir" in cfg and cfg.checkpoint_dir:
        argv += ["--checkpoint-dir", str(cfg.checkpoint_dir)]

    sys.argv = argv
    train_module.main()


if __name__ == "__main__":
    main()
