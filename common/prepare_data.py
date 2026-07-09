"""Generate the toy dataset and stage it on disk.

RUN THIS ON THE LOGIN NODE (or your laptop) — never inside a compute-node job.

Why? Alliance Canada *compute* nodes have no internet access. `make_moons` is
synthetic so it needs no download, but we still follow the real-world pattern:
any data preparation that might touch the internet (downloading a dataset,
pulling from a hub, etc.) must happen on the LOGIN node, which *does* have
internet. The compute node then only reads the already-staged files.

On the cluster we write the .npz files to $SCRATCH (large, fast, but purged
~every 60 days). On a laptop, just pass any local --data-dir.

Usage (cluster, after activating ~/.venv_ai):
    python common/prepare_data.py                 # -> $SCRATCH/moons-tutorial/data
    python common/prepare_data.py --data-dir /some/other/dir

Usage (laptop):
    python common/prepare_data.py --data-dir /tmp/moons/data
"""

import argparse
import os

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def default_data_dir() -> str:
    """Default to $SCRATCH on the cluster, else a local folder."""
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return os.path.join(scratch, "moons-tutorial", "data")
    return os.path.join(os.getcwd(), "moons-data")


def main():
    parser = argparse.ArgumentParser(description="Generate make_moons and save train/test .npz")
    parser.add_argument("--data-dir", default=default_data_dir(),
                        help="Where to write train.npz / test.npz "
                             "(default: $SCRATCH/moons-tutorial/data)")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # --- This is where you'd download/stage a REAL dataset on the login node. ---
    # e.g. torchvision.datasets.CIFAR10(root=..., download=True)
    # For real datasets, do the download here (login node has internet), then
    # compute-node jobs read the files without needing the internet.
    X, y = make_moons(n_samples=args.n_samples, noise=args.noise, random_state=args.seed)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    train_path = os.path.join(args.data_dir, "train.npz")
    test_path = os.path.join(args.data_dir, "test.npz")
    np.savez(train_path, X=X_train, y=y_train)
    np.savez(test_path, X=X_test, y=y_test)

    print(f"Wrote {len(X_train)} train samples -> {train_path}")
    print(f"Wrote {len(X_test)} test samples  -> {test_path}")


if __name__ == "__main__":
    main()
