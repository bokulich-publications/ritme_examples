"""Percentile-bootstrap 95% CIs for held-out test metrics.

Library function :func:`bootstrap_metrics` resamples the (true, pred)
pairs of a held-out test set ``n_resamples`` times, recomputes
RMSE / R² / Pearson on each resample, and returns the point estimate
plus the 95% percentile interval (``ci_low``, ``ci_high``). Predictions
are computed once on the full test set; the bootstrap operates on
those frozen pairs (the standard "test-set bootstrap"), so cost is
dominated by the metric recomputations rather than by the model.

Run as a CLI to bootstrap a ritme experiment's tuned model::

    python -m src.bootstrap_metrics \\
        <experiment_dir> <model_type> <splits_dir> [--n-resamples 1000]

Loads ``<model_type>_best_model.pkl`` from ``experiment_dir``, reads
the target column from ``experiment_dir/experiment_config.json``, scores
``<splits_dir>/test.pkl``, and writes ``bootstrap_test_metrics.csv``
into ``experiment_dir`` with one row per metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, root_mean_squared_error


def _metrics_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse_test": root_mean_squared_error(y_true, y_pred),
        "r2_test": r2_score(y_true, y_pred),
        "pearson_corr_test": pearsonr(y_true, y_pred).statistic,
    }


def bootstrap_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 12,
) -> pd.DataFrame:
    """Return a DataFrame of point + bootstrap-CI per metric.

    Columns: ``metric, point, mean, ci_low, ci_high, n_resamples, ci_pct``.
    Bootstrap pairs are sampled with replacement; degenerate resamples
    where Pearson is undefined (zero variance in either array) are
    skipped before the percentile is taken.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be matching 1D arrays; "
            f"got {y_true.shape} vs {y_pred.shape}"
        )
    n = y_true.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 samples to bootstrap; got {n}.")

    point = _metrics_row(y_true, y_pred)

    rng = np.random.default_rng(seed)
    rmse_b: list[float] = []
    r2_b: list[float] = []
    pcorr_b: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        rmse_b.append(root_mean_squared_error(yt, yp))
        r2_b.append(r2_score(yt, yp))
        if yt.std() > 0 and yp.std() > 0:
            pcorr_b.append(pearsonr(yt, yp).statistic)

    alpha = (1.0 - ci) / 2.0
    rows = []
    for name, samples in [
        ("rmse_test", rmse_b),
        ("r2_test", r2_b),
        ("pearson_corr_test", pcorr_b),
    ]:
        arr = np.asarray(samples, dtype=float)
        rows.append(
            {
                "metric": name,
                "point": point[name],
                "mean": float(arr.mean()) if arr.size else float("nan"),
                "ci_low": float(np.quantile(arr, alpha)) if arr.size else float("nan"),
                "ci_high": (
                    float(np.quantile(arr, 1 - alpha)) if arr.size else float("nan")
                ),
                "n_resamples": int(arr.size),
                "ci_pct": ci,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_ritme_experiment(
    experiment_dir: Path,
    model_type: str,
    splits_dir: Path,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    from ritme.evaluate_tuned_models import get_predictions, load_best_model

    cfg = json.loads((experiment_dir / "experiment_config.json").read_text())
    target = cfg["target"]
    test = pd.read_pickle(splits_dir / "test.pkl")
    tmodel = load_best_model(model_type, str(experiment_dir))
    preds = get_predictions(test, tmodel, target, "test")
    return bootstrap_metrics(
        preds["true"].astype(float).to_numpy(),
        preds["pred"].astype(float).to_numpy(),
        n_resamples=n_resamples,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Ritme experiment directory containing <model_type>_best_model.pkl.",
    )
    parser.add_argument(
        "model_type",
        help="Model class to bootstrap (matches `ls_model_types[0]` of the config).",
    )
    parser.add_argument(
        "splits_dir",
        type=Path,
        help=(
            "Directory holding train_val.pkl/test.pkl produced by "
            "`ritme split-train-test`."
        ),
    )
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()

    df = _bootstrap_ritme_experiment(
        args.experiment_dir,
        args.model_type,
        args.splits_dir,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    out = args.experiment_dir / "bootstrap_test_metrics.csv"
    df.to_csv(out, index=False)
    print(f"Bootstrap metrics written to {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
