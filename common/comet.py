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
    """Register Comet CLI flags on an argparse parser."""
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
        self._exp = experiment

    @property
    def enabled(self) -> bool:
        return self._exp is not None

    def log_parameters(self, params: dict):
        if self._exp is not None:
            self._exp.log_parameters(params)

    def log_metric(self, name: str, value, step=None):
        if self._exp is not None:
            self._exp.log_metric(name, value, step=step)

    def end(self):
        if self._exp is not None:
            self._exp.end()


def make_comet_logger(args) -> CometLogger:
    """Build a CometLogger from parsed args, or a disabled one if unavailable."""
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
