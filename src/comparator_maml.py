"""mAML search-space comparator arm (classification only, use case 3).

Runs mAML's published scaler x classifier grid (see
:mod:`src.comparator_maml_space`) against the ritme train/test split, so the
result sits in the same column space as the ritme and auto-sklearn arms.

This is *not* the upstream mAML CLI. That CLI has no held-out-test path -- it
scores with ``cross_val_predict`` over the whole input -- and it computes its
prevalence filter from the labels of the full dataset. Two deliberate
deviations follow:

- the prevalence filter is fitted on ``train_val`` only and the resulting
  column mask applied to ``test``;
- model selection scores ``roc_auc`` rather than upstream's ``accuracy``
  default, matching what use case 3 reports everywhere else.

mRMR selection and SMOTE over-sampling stay off, matching the upstream CLI
defaults (``--mrmr_n 0``, no ``--over_sampling``); both are supervised steps
that would reintroduce leakage.

Some classifiers in the grid (``LinearSVC``, ``SGDClassifier`` under a hinge
loss) expose only ``decision_function``. When one of those wins, it is refit
inside a ``CalibratedClassifierCV`` so the reported metric set -- which
includes ``log_loss`` -- can be produced in the same column space as the other
arms.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.comparator_common import load_xy, write_configs, write_metrics
from src.comparator_maml_space import ALL_CLASSIFIERS, SCALERS
from src.eval_automl import get_metrics_n_roc_curve
from src.launch_models import USECASES

N_FOLDS = 5
DEFAULT_PREVALENCE = 0.2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--usecase", default="u3")
    p.add_argument("--task", default="classification", choices=["classification"])
    p.add_argument("--data-splits-folder", required=True)
    p.add_argument("--path-to-features", required=True)
    p.add_argument("--path-to-md", required=True)
    p.add_argument("--target", required=True)
    p.add_argument(
        "--enrich-with",
        dest="enrich_with",
        action="append",
        default=[],
        help="Metadata column to append to the feature table, repeatable.",
    )
    p.add_argument("--prevalence", type=float, default=DEFAULT_PREVALENCE)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--out-dir", default="comparators")
    p.add_argument(
        "--subset-for-smoke",
        action="store_true",
        help="Use only the first two scalers and classifiers, for a smoke run.",
    )
    # Accepted for launcher symmetry; mAML's grid has no time budget.
    p.add_argument("--total-time-s", type=int, default=None)
    return p.parse_args()


def prevalence_mask(X: pd.DataFrame, y: pd.Series, prevalence: float) -> pd.Index:
    """Columns whose maximum within-class presence rate exceeds ``prevalence``.

    Upstream rule (``filter_low_prevalence_features``), but fitted on the
    training split alone.
    """
    if prevalence == 0:
        return X.columns
    rates = [
        (X[y.values == label] > 0).sum(axis=0) / int((y.values == label).sum())
        for label in np.unique(y.values)
    ]
    return X.columns[pd.DataFrame(rates).max() > prevalence]


def main() -> None:
    args = parse_args()
    if USECASES[args.usecase]["task"] != "classification":
        raise ValueError(f"mAML is classification-only; {args.usecase!r} is not.")

    X_train, y_train, X_test, y_test, _ = load_xy(
        args.path_to_features,
        args.path_to_md,
        args.data_splits_folder,
        args.target,
        args.task,
        enrich_with=args.enrich_with,
    )
    print(f"Enriched with {args.enrich_with}")

    keep = prevalence_mask(X_train, y_train, args.prevalence)
    X_train, X_test = X_train[keep], X_test[keep]
    print(f"Prevalence filter at {args.prevalence}: kept {len(keep)} features")
    print("X_train.shape", X_train.shape)
    print("X_test.shape", X_test.shape)

    scalers = SCALERS[:2] if args.subset_for_smoke else SCALERS
    classifiers = ALL_CLASSIFIERS[:2] if args.subset_for_smoke else ALL_CLASSIFIERS
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=args.seed)

    rows = []
    best = {"score": -np.inf, "estimator": None, "scaler": None, "clf": None}
    for scaler_name, scaler in scalers:
        for clf, param_grid in classifiers:
            clf_name = type(clf).__name__
            pipeline = Pipeline([("scl", scaler), ("clf", clf)])
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=args.n_jobs,
                error_score=np.nan,
            )
            try:
                search.fit(X_train, y_train)
            except Exception as e:  # one bad combination must not end the grid
                print(f"{scaler_name:28s} {clf_name:28s} FAILED: {e}")
                rows.append(
                    {
                        "scaler": scaler_name,
                        "classifier": clf_name,
                        "params": "",
                        "mean_cv_roc_auc": np.nan,
                    }
                )
                continue
            for params, mean_score in zip(
                search.cv_results_["params"], search.cv_results_["mean_test_score"]
            ):
                rows.append(
                    {
                        "scaler": scaler_name,
                        "classifier": clf_name,
                        "params": str(params),
                        "mean_cv_roc_auc": mean_score,
                    }
                )
            print(
                f"{scaler_name:28s} {clf_name:28s} "
                f"best_cv_roc_auc={search.best_score_:.4f}"
            )
            if np.isfinite(search.best_score_) and search.best_score_ > best["score"]:
                best = {
                    "score": search.best_score_,
                    "estimator": search.best_estimator_,
                    "scaler": scaler_name,
                    "clf": clf_name,
                }

    if best["estimator"] is None:
        raise RuntimeError("No pipeline in the mAML grid produced a valid score.")
    print(
        f"Best: {best['scaler']} + {best['clf']} "
        f"(mean CV ROC-AUC {best['score']:.4f})"
    )

    final = best["estimator"]
    calibrated = not hasattr(final, "predict_proba")
    if calibrated:
        print(f"{best['clf']} has no predict_proba; refitting calibrated")
        final = CalibratedClassifierCV(final, cv=N_FOLDS, method="sigmoid")
        final.fit(X_train, y_train)

    metrics, fig = get_metrics_n_roc_curve(final, X_train, y_train, X_test, y_test)
    metrics["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    metrics["probability_calibrated"] = calibrated
    metrics["best_scaler"] = best["scaler"]
    metrics["best_classifier"] = best["clf"]
    metrics["n_configs_evaluated"] = len(rows)
    metrics["n_features_after_prevalence"] = len(keep)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = write_metrics(args.out_dir, args.usecase, "maml", metrics)
    configs_path = write_configs(args.out_dir, args.usecase, "maml", pd.DataFrame(rows))
    fig_path = os.path.join(args.out_dir, f"{args.usecase}_maml_best_roc.png")
    fig.savefig(fig_path, bbox_inches="tight")

    for label, path in [
        ("Metrics", metrics_path),
        ("Configs", configs_path),
        ("Plot", fig_path),
    ]:
        print(f"{label} written to {path}")


if __name__ == "__main__":
    main()
