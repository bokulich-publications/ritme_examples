"""Compute the adaptive TPE warm-up length for the B1 figure marker.

Replays ritme's `_adaptive_n_startup_trials` (n_startup = max(20, 5 x
effective search-space dims)) with the same inputs the B1 runs get, and
writes the value to a JSON file. Importing ritme is heavy, so run this
inside a small SLURM job (launched by `benchmarking.launch_b1`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.launch_models import REPO_ROOT, USECASES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--usecase", default="u1")
    p.add_argument("--model-type", default="xgb")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ritme.tune_models import _adaptive_n_startup_trials

    spec = USECASES[args.usecase]
    config_dir = REPO_ROOT / spec["use_case_dir"] / "config"
    base_config = json.loads(
        (config_dir / f"{spec['config_prefix']}_base_tpe.json").read_text()
    )
    train_val = pd.read_pickle(REPO_ROOT / spec["data_splits"] / "train_val.pkl")
    tax = pd.read_csv(REPO_ROOT / spec["path_tax"], sep="\t", index_col=0)

    # Mirrors ritme's own short-circuit: an explicit n_startup_trials in the
    # config wins over the adaptive value.
    model_hyperparameters = base_config.get("model_hyperparameters", {})
    n_startup = model_hyperparameters.get("n_startup_trials") or (
        _adaptive_n_startup_trials(
            args.model_type, train_val, tax, model_hyperparameters
        )
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "usecase": args.usecase,
                "model_type": args.model_type,
                "n_startup_trials": int(n_startup),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"n_startup_trials={n_startup} written to {out}")


if __name__ == "__main__":
    main()
