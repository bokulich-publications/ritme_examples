"""Percentile-bootstrap 95% CIs for held-out test metrics.

Library functions :func:`bootstrap_metrics` and
:func:`bootstrap_classification_metrics` resample the (true, pred) pairs of a
held-out test set ``n_resamples`` times, recompute the relevant metrics on
each resample, and return the point estimate plus the 95% percentile
interval (``ci_low``, ``ci_high``). Predictions are computed once on the
full test set; the bootstrap operates on those frozen pairs (the standard
"test-set bootstrap"), so cost is dominated by the metric recomputations
rather than by the model.

Run as a CLI to bootstrap a ritme experiment's tuned model::

    python -m src.bootstrap_metrics \\
        <experiment_dir> <model_type> <splits_dir> [--n-resamples 1000]

Loads ``<model_type>_best_model.pkl`` from ``experiment_dir``, reads the
target column from ``experiment_dir/experiment_config.json``, scores
``<splits_dir>/test.pkl``, and writes ``bootstrap_test_metrics.csv`` into
``experiment_dir`` with one row per metric. The CLI auto-dispatches to the
regression or classification metric set based on ``model_type``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from ritme.evaluate_tuned_models import get_predictions, load_best_model

# Binary-classifier model classes that the CLI routes through
# `bootstrap_classification_metrics`. `nn_corn` is excluded because ritme's
# CORN head is ordinal-regression — bootstrapping it as a binary classifier
# would mis-resolve the positive class. Run CORN through the regression path.
CLASSIFICATION_MODELS = frozenset({"logreg", "rf_class", "xgb_class", "nn_class"})


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
    """Return a DataFrame of point + bootstrap-CI per regression metric.

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


def _classification_metrics_row(
    y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    return {
        "auroc_test": roc_auc_score(y_true, y_proba),
        "accuracy_test": accuracy_score(y_true, y_pred),
        "f1_test": f1_score(y_true, y_pred, zero_division=0),
        "precision_test": precision_score(y_true, y_pred, zero_division=0),
        "recall_test": recall_score(y_true, y_pred, zero_division=0),
    }


def bootstrap_classification_metrics(
    y_true: Iterable[int],
    y_proba: Iterable[float],
    y_pred: Iterable[int] | None = None,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 12,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return a DataFrame of point + bootstrap-CI per binary-classification metric.

    Metrics: AUROC, accuracy, F1, precision, recall (positive class = 1).
    Columns: ``metric, point, mean, ci_low, ci_high, n_resamples, ci_pct`` —
    same shape as the regression sibling for downstream aggregation parity.

    ``y_proba`` is the positive-class probability in [0, 1]. ``y_pred`` is
    derived from ``y_proba >= threshold`` if not supplied. Bootstrap
    resamples that end up with only one class are skipped for AUROC (which
    is undefined on a single class), so the ``n_resamples`` column for
    ``auroc_test`` may be smaller than the requested count on heavily
    imbalanced test sets — inspect that column when interpreting the CI.
    """
    if y_pred is not None and threshold != 0.5:
        raise ValueError(
            "Pass either `y_pred` (already thresholded) or a non-default "
            "`threshold` (we derive `y_pred`); not both."
        )
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_pred is None:
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred, dtype=int)
    if not (y_true.shape == y_proba.shape == y_pred.shape) or y_true.ndim != 1:
        raise ValueError(
            "y_true, y_proba, y_pred must be matching 1D arrays; got shapes "
            f"{y_true.shape} / {y_proba.shape} / {y_pred.shape}"
        )
    n = y_true.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 samples to bootstrap; got {n}.")
    if set(np.unique(y_true).tolist()) - {0, 1}:
        raise ValueError(
            f"bootstrap_classification_metrics expects binary 0/1 labels; "
            f"got unique values {np.unique(y_true).tolist()}"
        )
    if np.unique(y_true).size < 2:
        raise ValueError(
            "Cannot bootstrap AUROC on a single-class test set "
            f"(all labels = {int(y_true[0])})."
        )
    if np.isnan(y_proba).any():
        raise ValueError("y_proba contains NaN; refusing to bootstrap.")
    if not ((y_proba >= 0).all() and (y_proba <= 1).all()):
        raise ValueError(
            "y_proba must lie in [0, 1]; got range "
            f"[{float(y_proba.min())}, {float(y_proba.max())}]."
        )
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1]; got {threshold}.")

    point = _classification_metrics_row(y_true, y_proba, y_pred)

    rng = np.random.default_rng(seed)
    auroc_b: list[float] = []
    accuracy_b: list[float] = []
    f1_b: list[float] = []
    precision_b: list[float] = []
    recall_b: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, yp_proba, yp = y_true[idx], y_proba[idx], y_pred[idx]
        if np.unique(yt).size > 1:
            auroc_b.append(roc_auc_score(yt, yp_proba))
        accuracy_b.append(accuracy_score(yt, yp))
        f1_b.append(f1_score(yt, yp, zero_division=0))
        precision_b.append(precision_score(yt, yp, zero_division=0))
        recall_b.append(recall_score(yt, yp, zero_division=0))

    alpha = (1.0 - ci) / 2.0
    rows = []
    for name, samples in [
        ("auroc_test", auroc_b),
        ("accuracy_test", accuracy_b),
        ("f1_test", f1_b),
        ("precision_test", precision_b),
        ("recall_test", recall_b),
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


MULTICLASS_METRICS = [
    "accuracy_test",
    "f1_macro_test",
    "roc_auc_macro_ovr_test",
    "log_loss_test",
    "balanced_accuracy_test",
    "mcc_test",
]


def _multiclass_metrics_row(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray, present: np.ndarray
) -> dict[str, float]:
    """ritme's classification metric set plus accuracy, over a fixed class set.

    ``present`` is held fixed across bootstrap resamples so every resample
    estimates the same quantity; a resample that loses one of those classes
    makes its one-vs-rest AUROC undefined and yields ``nan``, which the caller
    excludes.
    """
    y_pred = classes[y_proba.argmax(axis=1)]
    keep = np.isin(classes, present)
    proba = y_proba[:, keep]
    mass = proba.sum(axis=1, keepdims=True)
    proba = proba / np.where(mass > 0, mass, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        auc = roc_auc_score(
            y_true, proba, multi_class="ovr", average="macro", labels=present
        )
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy_test": float(accuracy_score(y_true, y_pred)),
        "f1_macro_test": float(f1_macro),
        "roc_auc_macro_ovr_test": float(auc),
        "log_loss_test": float(log_loss(y_true, proba, labels=present)),
        "balanced_accuracy_test": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc_test": float(matthews_corrcoef(y_true, y_pred)),
    }


def bootstrap_multiclass_metrics(
    y_true: Iterable,
    y_proba: np.ndarray,
    classes: Iterable,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 12,
) -> pd.DataFrame:
    """Return point + bootstrap-CI per metric for a multi-class classifier.

    Same column schema as the binary sibling. The class set is frozen to the
    classes present in the full ``y_true``; resamples in which one of them
    disappears are excluded from the AUROC percentiles, so that metric's
    ``n_resamples`` can be lower than requested -- inspect that column.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    classes = np.asarray(list(classes))
    if y_proba.ndim != 2 or y_proba.shape != (y_true.shape[0], classes.shape[0]):
        raise ValueError(
            f"y_proba must be (n_samples, n_classes); got {y_proba.shape} for "
            f"{y_true.shape[0]} samples and {classes.shape[0]} classes."
        )
    if np.isnan(y_proba).any():
        raise ValueError("y_proba contains NaN; refusing to bootstrap.")
    seen = set(y_true.tolist())
    present = np.array([c for c in classes if c in seen])
    if present.size < 2:
        raise ValueError(f"Need at least 2 classes in y_true; got {present.tolist()}.")

    point = _multiclass_metrics_row(y_true, y_proba, classes, present)
    samples: dict[str, list[float]] = {name: [] for name in MULTICLASS_METRICS}
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        row = _multiclass_metrics_row(y_true[idx], y_proba[idx], classes, present)
        for name in MULTICLASS_METRICS:
            samples[name].append(row[name])

    alpha = (1.0 - ci) / 2.0
    rows = []
    for name in MULTICLASS_METRICS:
        arr = np.asarray(samples[name], dtype=float)
        arr = arr[np.isfinite(arr)]
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


def _positive_class_proba(
    proba: np.ndarray, classes: list, target_dtype: np.dtype
) -> np.ndarray:
    """Pick the column of ``proba`` that matches the positive class (==1).

    ``target_dtype`` must be an integer subtype — we coerce ``classes``
    through it so equality against ``1`` works regardless of whether ritme
    returned the class labels as numpy ints or Python ints. Cast the target
    column to int upstream before calling this; non-integer dtypes raise.
    """
    if not np.issubdtype(np.dtype(target_dtype), np.integer):
        raise TypeError(
            "U3 classification expects an integer target dtype; got "
            f"{target_dtype}. Cast the target column to int upstream."
        )
    classes_arr = np.asarray(classes).astype(target_dtype)
    positive = np.array(1).astype(target_dtype)
    matches = np.where(classes_arr == positive)[0]
    if matches.size != 1:
        raise ValueError(
            f"Expected exactly one positive class (==1) in classes={classes} "
            f"(coerced to {classes_arr.tolist()}); found {matches.size} match(es). "
            f"target_dtype={target_dtype}."
        )
    return proba[:, matches[0]]


def _bootstrap_ritme_experiment(
    experiment_dir: Path,
    model_type: str,
    splits_dir: Path,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    cfg = json.loads((experiment_dir / "experiment_config.json").read_text())
    target = cfg["target"]
    test = pd.read_pickle(splits_dir / "test.pkl")
    tmodel = load_best_model(model_type, str(experiment_dir))

    if cfg.get("task_type", "regression") == "classification":
        proba, classes = tmodel.predict_proba(test, "test")
        if len(classes) > 2:
            return bootstrap_multiclass_metrics(
                test[target].to_numpy(),
                proba,
                classes,
                n_resamples=n_resamples,
                seed=seed,
            )
        y_true = test[target].astype(int).to_numpy()
        y_proba = _positive_class_proba(proba, classes, test[target].dtype)
        return bootstrap_classification_metrics(
            y_true, y_proba, n_resamples=n_resamples, seed=seed
        )

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
