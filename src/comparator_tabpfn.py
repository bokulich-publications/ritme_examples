"""TabPFN comparator arm for the usecase regression and classification tasks.

Runs TabPFN, Prior Labs' pretrained tabular foundation model, against the same
train/test split and metadata enrichment as the ritme and AutoML arms. TabPFN
predicts in context from the training set, one forward pass per ensemble
member; there is no hyperparameter search, so the arm has no budget knob and
no validation protocol of its own.

The microbial feature tables are far wider than the model's pretraining budget
(500 features for v2), which the arm handles in one of two ways:

- ``--max-features`` unset: the full table goes in. TabPFN's own feature
  subsampling hands each ensemble member the metadata columns plus a balanced
  slice of the microbial features, and the ensemble is made large enough for
  every feature to be seen by at least one member.
- ``--max-features N``: only the microbial features with the highest mean
  relative abundance in the training split are kept, so the matrix has at
  most N columns and TabPFN runs within its pretraining limits with default
  settings.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, root_mean_squared_error
from tabpfn import TabPFNClassifier, TabPFNRegressor, __version__ as tabpfn_version
from tabpfn.constants import ModelVersion
from tabpfn.preprocessing.ensemble import DEFAULT_N_ESTIMATORS

from src.comparator_common import load_xy, write_configs, write_metrics
from src.eval_automl import get_metrics_n_roc_curve, get_metrics_n_scatterplot
from src.launch_models import REPO_ROOT
from src.tabpfn_models import checkpoint_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--usecase", required=True)
    p.add_argument("--task", required=True, choices=["regression", "classification"])
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
    p.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Cap on the design-matrix width: metadata columns plus the most "
        "abundant microbial features. Default: the full feature table.",
    )
    p.add_argument("--model-dir", required=True, help="Directory with checkpoints.")
    p.add_argument("--model-version", default="v2")
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--n-jobs", type=int, default=1, help="Torch CPU threads.")
    p.add_argument("--out-dir", default="comparators")
    p.add_argument("--method", default=None, help="Output file stem; default derived.")
    return p.parse_args()


def method_name(max_features: int | None) -> str:
    return "tabpfn" if max_features is None else f"tabpfn_top{max_features}"


def split_columns(X: pd.DataFrame, path_ft: str) -> tuple[list[str], list[str]]:
    """Return ``(metadata columns, microbial feature columns)`` of ``X``."""
    microbial = set(pd.read_csv(path_ft, sep="\t", index_col=0, nrows=0).columns)
    otu_cols = [c for c in X.columns if c in microbial]
    meta_cols = [c for c in X.columns if c not in microbial]
    return meta_cols, otu_cols


def most_abundant(X_train: pd.DataFrame, otu_cols: list[str], n_keep: int) -> list[str]:
    """The ``n_keep`` microbial features with the highest mean training abundance."""
    return X_train[otu_cols].mean(axis=0).nlargest(n_keep).index.tolist()


def build_model(
    task: str,
    version: str,
    checkpoint: Path,
    seed: int,
    n_features: int,
    n_constant: int,
) -> tuple[TabPFNClassifier | TabPFNRegressor, dict]:
    """TabPFN estimator for this table width, and the settings it resolved to.

    Within the pretraining budget the model keeps its defaults. Beyond it, the
    first ``n_constant`` columns (the metadata) go into every ensemble member,
    the microbial features are shared out round-robin, and the member count is
    raised until every feature is covered at least once.
    """
    cls = TabPFNClassifier if task == "classification" else TabPFNRegressor
    model = cls.create_default_for_version(
        ModelVersion(version),
        model_path=str(checkpoint),
        device="cpu",
        random_state=seed,
    )
    config = model.get_inference_config()
    budget = config.MAX_NUMBER_OF_FEATURES
    settings = {
        "pretraining_max_features": budget,
        "max_features_per_estimator": None,
        "feature_subsampling": None,
    }
    if n_features <= budget:
        return model, settings

    # The checkpoints' own per-member cap differs between tasks (none for the
    # v2 classifier), so every member is held to the pretraining budget.
    n_estimators = max(
        DEFAULT_N_ESTIMATORS,
        math.ceil((n_features - n_constant) / (budget - n_constant)),
    )
    model.set_params(
        ignore_pretraining_limits=True,
        n_estimators=n_estimators,
        inference_config={
            "PREPROCESS_TRANSFORMS": [
                dataclasses.replace(c, max_features_per_estimator=budget)
                for c in config.PREPROCESS_TRANSFORMS
            ],
            "FEATURE_SUBSAMPLING_METHOD": "constant_and_balanced",
            "FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT": n_constant,
        },
    )
    settings["max_features_per_estimator"] = budget
    settings["feature_subsampling"] = "constant_and_balanced"
    return model, settings


def regression_test_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 12,
) -> dict[str, float]:
    """Percentile-bootstrap CIs for the test RMSE, R2 and Pearson r.

    Same resampling as `src.bootstrap_metrics.bootstrap_metrics`, which cannot
    be imported here because it imports ritme at module level.
    """
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    samples: dict[str, list[float]] = {"rmse": [], "r2": [], "pearson_corr": []}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        samples["rmse"].append(root_mean_squared_error(yt, yp))
        samples["r2"].append(r2_score(yt, yp))
        if yt.std() > 0 and yp.std() > 0:
            samples["pearson_corr"].append(pearsonr(yt, yp).statistic)
    alpha = (1.0 - ci) / 2.0
    out = {}
    for name, values in samples.items():
        arr = np.asarray(values, dtype=float)
        out[f"{name}_test_ci_low"] = float(np.quantile(arr, alpha))
        out[f"{name}_test_ci_high"] = float(np.quantile(arr, 1 - alpha))
    return out


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.n_jobs)
    method = args.method or method_name(args.max_features)

    checkpoint = checkpoint_path(
        args.model_dir,
        args.model_version,
        "classifier" if args.task == "classification" else "regressor",
    )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No TabPFN checkpoint at {checkpoint}. Compute nodes are offline; "
            f"download it first on a node with internet access:\n"
            f"  python -m src.tabpfn_models {args.model_dir} "
            f"--version {args.model_version}"
        )

    X_train, y_train, X_test, y_test, _ = load_xy(
        args.path_to_features,
        args.path_to_md,
        args.data_splits_folder,
        args.target,
        args.task,
        enrich_with=args.enrich_with,
    )
    meta_cols, otu_cols = split_columns(X_train, args.path_to_features)
    if args.max_features is not None:
        n_keep = args.max_features - len(meta_cols)
        if n_keep <= 0:
            raise ValueError(
                f"--max-features {args.max_features} leaves no room for microbial "
                f"features next to {len(meta_cols)} metadata columns."
            )
        otu_cols = most_abundant(X_train, otu_cols, n_keep)
    columns = meta_cols + otu_cols
    X_train, X_test = X_train[columns], X_test[columns]
    print(f"Enriched with {args.enrich_with} -> {len(meta_cols)} metadata columns")
    print("X_train.shape", X_train.shape)
    print("X_test.shape", X_test.shape)

    model, settings = build_model(
        args.task,
        args.model_version,
        checkpoint,
        args.seed,
        n_features=len(columns),
        n_constant=len(meta_cols),
    )
    print(f"TabPFN {tabpfn_version}, model {args.model_version}: {settings}")

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    print(f"Fitted {model.n_estimators_} estimators in {fit_seconds:.1f}s")

    t0 = time.perf_counter()
    if args.task == "classification":
        metrics, fig = get_metrics_n_roc_curve(model, X_train, y_train, X_test, y_test)
    else:
        metrics, fig = get_metrics_n_scatterplot(
            model, X_train, y_train, X_test, y_test
        )
        cis = regression_test_cis(
            np.asarray(y_test, dtype=float),
            np.asarray(model.predict(X_test), dtype=float),
            seed=args.seed,
        )
        for name, value in cis.items():
            metrics[name] = value
    predict_seconds = time.perf_counter() - t0

    metrics["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    metrics["restricted_model"] = f"TabPFN-{args.model_version}"
    metrics["tabpfn_version"] = tabpfn_version
    metrics["n_estimators"] = model.n_estimators_
    metrics["n_features"] = len(columns)
    metrics["n_metadata_features"] = len(meta_cols)
    for name, value in settings.items():
        metrics[name] = value
    metrics["fit_seconds"] = round(fit_seconds, 1)
    metrics["predict_seconds"] = round(predict_seconds, 1)
    metrics["n_threads"] = args.n_jobs

    params = model.get_params()
    params["model_path"] = os.path.relpath(checkpoint, REPO_ROOT)
    configs = pd.DataFrame([{k: str(v) for k, v in sorted(params.items())}])

    metrics_path = write_metrics(args.out_dir, args.usecase, method, metrics)
    configs_path = write_configs(args.out_dir, args.usecase, method, configs)
    suffix = "roc" if args.task == "classification" else "true_vs_pred"
    fig_path = os.path.join(args.out_dir, f"{args.usecase}_{method}_best_{suffix}.png")
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Metrics written to {metrics_path}")
    print(f"Configs written to {configs_path}")
    print(f"Plot written to {fig_path}")
    print(metrics.T.to_string())


if __name__ == "__main__":
    main()
