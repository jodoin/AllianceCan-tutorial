"""Optional Comet.ml logging, isolated from the training loop (example 06).

`train.py` always calls into a `CometLogger`; when Comet is disabled (the common
case, examples 00-05 and 07) or `comet_ml` isn't installed, every method is a
no-op, so the training loop needs no `if experiment is not None` checks.

On Narval, compute nodes have no internet:
  * OFFLINE mode (recommended default) writes an experiment archive to $SCRATCH,
    which you `comet upload` from the login node afterward.
  * ONLINE mode needs `module load httpproxy` in job.sh first.
The API key is read from the COMET_API_KEY env var / ~/.comet.config, never
hardcoded. See 06-single-job-comet/README.md.
"""


def add_arguments(parser):
    """Register Comet CLI flags on an argparse parser.

    Args:
        parser: the argparse.ArgumentParser to add the Comet flags to.
    """
    group = parser.add_argument_group("comet (optional, example 06)")
    group.add_argument("--comet", action="store_true",
                       help="Log training curves to Comet.ml (needs comet_ml installed).")
    group.add_argument("--comet-offline-dir", default=None,
                       help="If set, use Comet OFFLINE mode and write the experiment "
                            "archive here (e.g. on $SCRATCH), to upload later from the "
                            "login node. If unset, --comet uses ONLINE mode.")
    group.add_argument("--comet-project", default="moons-tutorial",
                       help="Comet project name.")


class CometLogger:
    """Thin wrapper around a comet_ml experiment; no-ops when disabled."""

    def __init__(self, experiment=None):
        """Wrap a comet_ml experiment, or None to make every method a no-op.

        Args:
            experiment: a comet_ml Experiment/OfflineExperiment, or None to disable.
        """
        self._exp = experiment

    @property
    def enabled(self) -> bool:
        """True if a real Comet experiment is attached."""
        return self._exp is not None

    def log_parameters(self, params: dict):
        """Log run hyperparameters to Comet (no-op when disabled).

        Args:
            params: mapping of hyperparameter name -> value.
        """
        if self._exp is not None:
            self._exp.log_parameters(params)

    def log_metric(self, name: str, value, step=None):
        """Log a single metric value at an optional step (no-op when disabled).

        Args:
            name: metric name, e.g. "train_loss".
            value: the metric value to record.
            step: optional step/epoch index the value belongs to.
        """
        if self._exp is not None:
            self._exp.log_metric(name, value, step=step)

    def end(self):
        """Finalize the experiment so Comet flushes/uploads it (no-op when disabled)."""
        if self._exp is not None:
            self._exp.end()


def make_comet_logger(args) -> CometLogger:
    """Build a CometLogger from parsed args, or a disabled one if unavailable.

    Args:
        args: parsed args carrying comet (bool), comet_offline_dir, comet_project,
            plus lr/epochs/hidden/seed which are logged as parameters.
    """
    if not getattr(args, "comet", False):
        return CometLogger(None)

    try:
        import comet_ml
    except ImportError:
        print("[warn] --comet set but comet_ml is not installed; skipping Comet logging.",
              flush=True)
        return CometLogger(None)

    import os
    common = dict(project_name=args.comet_project)
    if args.comet_offline_dir:
        os.makedirs(args.comet_offline_dir, exist_ok=True)
        exp = comet_ml.OfflineExperiment(offline_directory=args.comet_offline_dir, **common)
        print(f"[comet] OFFLINE mode; archive -> {args.comet_offline_dir}", flush=True)
    else:
        exp = comet_ml.Experiment(**common)  # ONLINE; key from env / ~/.comet.config
        print("[comet] ONLINE mode.", flush=True)

    logger = CometLogger(exp)
    logger.log_parameters({"lr": args.lr, "epochs": args.epochs,
                           "hidden": args.hidden, "seed": args.seed})
    return logger
