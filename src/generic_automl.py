"""automl implementation for usecase regression and binary-classification tasks."""

import argparse
import os
import subprocess
from pprint import pprint

from functools import partial

import autosklearn.classification
import autosklearn.regression
import pandas as pd
from autosklearn.ensembles import SingleBest
from autosklearn.metrics import make_scorer, roc_auc, root_mean_squared_error
from sklearn.metrics import roc_auc_score as sk_roc_auc_score

from src.comparator_common import load_xy
from src.eval_automl import (
    get_metrics_n_roc_curve,
    get_metrics_n_scatterplot,
)

REGRESSION_MODELS = [
    "ard_regression",
    "gradient_boosting",
    "mlp",
    "random_forest",
    "sgd",
]

CLASSIFICATION_MODELS = [
    "gradient_boosting",
    "liblinear_svc",
    "mlp",
    "random_forest",
    "sgd",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Auto-Sklearn regression or classification"
    )
    p.add_argument("--total-time-s", type=int, required=True)
    p.add_argument("--usecase", required=True)
    p.add_argument(
        "--task",
        required=True,
        choices=["regression", "classification"],
        help="Auto-Sklearn estimator family.",
    )
    p.add_argument("--data-splits-folder", required=True)
    p.add_argument("--path-to-features", required=True)
    p.add_argument("--path-to-md", required=True)
    p.add_argument("--target", required=True, help="Target column in metadata")
    p.add_argument(
        "--restricted-model",
        dest="restricted_models",
        nargs="+",
        default=[],
        help=(
            "Space-separated list of estimators to include. Names follow "
            "auto-sklearn's regressor / classifier vocabulary (mlp, "
            "random_forest, gradient_boosting are valid for both)."
        ),
    )
    p.add_argument(
        "--single-best",
        action="store_true",
        help=("If set, restrict Auto-Sklearn to single best model - no ensembles."),
    )
    p.add_argument(
        "--enrich-with",
        dest="enrich_with",
        action="append",
        default=[],
        help=(
            "Metadata column to append to the feature table, repeatable. "
            "Mirrors ritme's `data_enrich_with`; see src/launch_automl.py."
        ),
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="auto-sklearn worker processes; cap on wide tables (memory).",
    )
    p.add_argument(
        "--memory-limit-mb",
        type=int,
        default=24000,
        help="auto-sklearn per-worker memory limit (MB).",
    )
    p.add_argument(
        "--keep-tmp-folder",
        default=None,
        help="Directory for auto-sklearn's SMAC output, kept after the run "
        "(default: a temp dir SLURM deletes). Set it on shared storage so "
        "per-run failure statuses survive for post-mortems.",
    )
    return p.parse_args()


_CONVERTER_ENV = "ritme_usecases"


def _read_split(data_splits_folder: str, name: str) -> pd.DataFrame:
    """Read a train/test split frame, converting pkl -> parquet on the fly when needed.

    Pickled DataFrames embed NumPy module paths (`numpy.core.*` for NumPy 1.x,
    `numpy._core.*` for NumPy 2.x) and so cannot cross the NumPy 1<->2 boundary.
    This module runs in the autosklearn env (NumPy 1.x), but splits may have been
    written by the ritme env (NumPy 2.x) -- in which case the bare pickle is
    unreadable here (`ModuleNotFoundError: numpy._core`).

    Resolution order:
      1. If `<name>.parquet` exists, read it (parquet has no Python module refs,
         so it round-trips cleanly across NumPy major versions).
      2. Else try `<name>.pkl`. If that works (e.g. the pickle was written under
         the matching NumPy major version), return it.
      3. Else assume a NumPy major-version mismatch on the pickle and shell out to
         the `{_CONVERTER_ENV}` env -- which has NumPy 2.x and can read both pickle
         flavors -- to write `<name>.parquet` next to the pkl. Then read parquet.

    The conversion is one-shot: the resulting parquet stays on disk, so subsequent
    reads hit step (1) directly.
    """
    parquet_path = os.path.join(data_splits_folder, f"{name}.parquet")
    pkl_path = os.path.join(data_splits_folder, f"{name}.pkl")
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    try:
        return pd.read_pickle(pkl_path)
    except (ModuleNotFoundError, ImportError):
        # NumPy major-version mismatch on the pickle. Convert via the env that
        # wrote it -- that env can read pickles of both NumPy flavors.
        print(
            f"Pickle {pkl_path!r} unreadable in this env "
            f"(NumPy 1<->2 mismatch). Converting to parquet via "
            f"{_CONVERTER_ENV!r} env...",
            flush=True,
        )
        convert_code = (
            "import pandas as pd; "
            f"pd.read_pickle({pkl_path!r}).to_parquet({parquet_path!r}, index=True)"
        )
        subprocess.run(
            ["mamba", "run", "-n", _CONVERTER_ENV, "python", "-c", convert_code],
            check=True,
        )
        return pd.read_parquet(parquet_path)


def main():
    args = parse_args()

    X_train, y_train, X_test, y_test, _ = load_xy(
        args.path_to_features,
        args.path_to_md,
        args.data_splits_folder,
        args.target,
        args.task,
        enrich_with=args.enrich_with,
    )
    print(f"Enriched with {args.enrich_with}")
    print("X_train.shape", X_train.shape)
    print("X_test.shape", X_test.shape)

    if args.task == "classification":
        default_models = CLASSIFICATION_MODELS
        if y_train.nunique() <= 2:
            metric = roc_auc
        else:
            # auto-sklearn's roc_auc is binary-only; mirror ritme's
            # macro-OvR objective for multi-class targets.
            metric = make_scorer(
                "roc_auc_macro_ovr",
                partial(sk_roc_auc_score, multi_class="ovr", average="macro"),
                needs_proba=True,
            )
        estimator_key = "classifier"
        Estimator = autosklearn.classification.AutoSklearnClassifier
    else:
        default_models = REGRESSION_MODELS
        metric = root_mean_squared_error
        estimator_key = "regressor"
        Estimator = autosklearn.regression.AutoSklearnRegressor

    common_kwargs = dict(
        time_left_for_this_task=args.total_time_s,
        # every worker holds its own copy of the data; on wide tables the
        # worker count, not the core count, bounds memory
        n_jobs=args.n_jobs,
        metric=metric,
        memory_limit=args.memory_limit_mb,
    )
    if args.keep_tmp_folder:
        common_kwargs["tmp_folder"] = args.keep_tmp_folder
        common_kwargs["delete_tmp_folder_after_terminate"] = False
    if args.single_best:
        print("No ensembles - only single best model.")
        common_kwargs["ensemble_class"] = SingleBest

    models = args.restricted_models or default_models
    invalid = [m for m in args.restricted_models if m not in default_models]
    if invalid:
        # auto-sklearn raises a less-helpful error deeper inside fit(); flag early.
        raise ValueError(
            f"Restricted models {invalid} are not valid for task={args.task!r}. "
            f"Valid options: {default_models}."
        )
    print(f"Using auto-sklearn {args.task} with models: {models}")
    automl = Estimator(include={estimator_key: list(models)}, **common_kwargs)

    automl.fit(X_train, y_train)
    try:
        stats = pd.Series(
            [str(v.status) for v in automl.automl_.runhistory_.data.values()]
        ).value_counts()
        print("SMAC run statuses:\n", stats.to_string())
    except Exception as e:  # diagnostic only; never fail the run on it
        print(f"(could not summarise run statuses: {e})")
    print("Print model leaderboard:")
    print(automl.leaderboard())

    print("Final ensemble:")
    pprint(automl.show_models(), indent=4)

    # evaluate
    if args.task == "classification":
        metrics, fig = get_metrics_n_roc_curve(automl, X_train, y_train, X_test, y_test)
    else:
        metrics, fig = get_metrics_n_scatterplot(
            automl, X_train, y_train, X_test, y_test
        )

    out_dir = "automl"
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, f"{args.usecase}_metrics.csv")
    metrics.reset_index(names="model", inplace=True)
    metrics.to_csv(metrics_path, index=False)

    fig_suffix = "roc" if args.task == "classification" else "true_vs_pred"
    fig_path = os.path.join(out_dir, f"{args.usecase}_best_{fig_suffix}.png")
    fig.savefig(fig_path, bbox_inches="tight")

    print(f"Metrics written to {metrics_path}")
    print(f"Plot written to {fig_path}")


if __name__ == "__main__":
    main()
