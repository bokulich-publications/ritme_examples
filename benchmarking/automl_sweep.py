"""B4 auto-sklearn arm: search-only run at a fixed wall-clock budget, any use case.

The B4 generalisation of `automl_b2.py`. Runs inside the `autosklearn` conda
env (Python 3.8) on one SLURM job per (usecase, cores, seed) and mirrors the
comparator protocol: the same train_val split, the same metadata enrichment
ritme searches over, and the same resampling -- grouped 5-fold where the use
case defines a group column, otherwise shuffled (stratified) 5-fold seeded by
the run. The model space is restricted to the family closest to ritme's
winner. Writes exactly two small files per run -- a metrics JSON and a
per-evaluated-configuration CSV -- and keeps auto-sklearn's temporary
directory on node-local scratch, deleted when the job exits.

Every data-locating argument is explicit so this module imports nothing from
`src.launch_models`, which is not guaranteed to import under Python 3.8.
`benchmarking.launch_comparators` passes them all; running by hand needs all of them.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

# Import order is load-bearing: auto-sklearn's `ensembles` package imports
# `abstract_ensemble` partially unless an estimator module has been imported
# first, so the estimator modules must precede `SingleBest` (isort: skip).
import autosklearn.classification  # noqa: E402  isort: skip
import autosklearn.regression  # noqa: E402  isort: skip
from autosklearn.ensembles import SingleBest  # noqa: E402  isort: skip
from autosklearn.metrics import roc_auc, root_mean_squared_error  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from smac.tae import StatusType

from src.comparator_common import load_xy

N_FOLDS = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--usecase", required=True)
    p.add_argument("--task", required=True, choices=["regression", "classification"])
    p.add_argument("--data-splits-folder", required=True)
    p.add_argument("--path-to-features", required=True)
    p.add_argument("--path-to-md", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--group-by-column", default=None)
    p.add_argument("--enrich-with", action="append", default=[])
    p.add_argument("--budget-s", type=int, required=True)
    p.add_argument("--cores", type=int, required=True)
    p.add_argument("--n-jobs", type=int, required=True, help="parallel workers")
    p.add_argument(
        "--threads-per-job",
        type=int,
        required=True,
        help="BLAS/OpenMP threads per worker, recorded for the caption",
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--memory-limit-mb", type=int, required=True)
    p.add_argument("--restricted-model", required=True)
    return p.parse_args()


def make_cv(task: str, group_by_column, seed: int):
    """The splitter `src.comparator_tpot.make_cv` uses, so panel 2 compares
    like with like across arms."""
    if group_by_column:
        return GroupKFold(n_splits=N_FOLDS)
    if task == "classification":
        return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    X_train, y_train, _, _, groups = load_xy(
        args.path_to_features,
        args.path_to_md,
        args.data_splits_folder,
        args.target,
        args.task,
        group_by_column=args.group_by_column,
        enrich_with=args.enrich_with,
    )
    print(
        f"Loaded X{X_train.shape} enriched with {args.enrich_with}; "
        f"groups={'none' if groups is None else len(set(groups))} "
        f"cores={args.cores} n_jobs={args.n_jobs} "
        f"threads/job={args.threads_per_job} seed={args.seed} "
        f"budget={args.budget_s}s",
        flush=True,
    )

    if args.task == "regression":
        estimator_cls = autosklearn.regression.AutoSklearnRegressor
        include_key, metric, metric_name, mode = (
            "regressor",
            root_mean_squared_error,
            "rmse",
            "min",
        )
    else:
        estimator_cls = autosklearn.classification.AutoSklearnClassifier
        include_key, metric, metric_name, mode = (
            "classifier",
            roc_auc,
            "roc_auc",
            "max",
        )

    cv = make_cv(args.task, args.group_by_column, args.seed)
    resampling_args = {"groups": groups} if groups is not None else None

    # auto-sklearn's default is budget/10; floor it so short smoke runs can
    # still finish a configuration, and keep it below half the budget.
    per_run_time_limit = min(max(300, args.budget_s // 10), args.budget_s // 2)
    tmp_root = os.environ.get("TMPDIR", "/tmp")
    tmp_folder = os.path.join(
        tmp_root, f"askl_b4_{args.usecase}_c{args.cores}_s{args.seed}"
    )

    automl = estimator_cls(
        time_left_for_this_task=args.budget_s,
        per_run_time_limit=per_run_time_limit,
        n_jobs=args.n_jobs,
        seed=args.seed,
        memory_limit=args.memory_limit_mb,
        metric=metric,
        include={include_key: [args.restricted_model]},
        ensemble_class=SingleBest,
        resampling_strategy=cv,
        resampling_strategy_arguments=resampling_args,
        # `groups` is positional w.r.t. the rows of X, so any row subsampling
        # would silently misalign it; keep the data exactly as ritme sees it.
        dataset_compression=False,
        tmp_folder=tmp_folder,
        delete_tmp_folder_after_terminate=True,
    )
    print("Starting fit", flush=True)
    automl.fit(
        X_train,
        y_train,
        dataset_name=f"{args.usecase}_b4_c{args.cores}_s{args.seed}",
    )
    print("Fit returned", flush=True)

    # RunValue.cost is the validation loss as auto-sklearn minimises it: the
    # plain CV-mean RMSE for the RMSE scorer, and 1 - AUC for roc_auc
    # (optimum 1, greater_is_better). Convert back to the reported metric.
    def to_score(cost):
        return cost if metric_name == "rmse" else 1.0 - cost

    run_rows = []
    for run_key, run_value in automl.automl_.runhistory_.data.items():
        run_rows.append(
            {
                "config_id": run_key.config_id,
                "status": run_value.status.name,
                "cost": run_value.cost,
                f"val_{metric_name}": to_score(run_value.cost),
                "duration_s": run_value.time,
                "starttime": run_value.starttime,
                "endtime": run_value.endtime,
            }
        )
    runs_df = pd.DataFrame(
        run_rows,
        columns=[
            "config_id",
            "status",
            "cost",
            f"val_{metric_name}",
            "duration_s",
            "starttime",
            "endtime",
        ],
    )
    success = runs_df[runs_df["status"] == StatusType.SUCCESS.name]
    best_val = None
    if len(success):
        column = success[f"val_{metric_name}"]
        best_val = float(column.min() if mode == "min" else column.max())

    stem = f"automl_{args.usecase}_c{args.cores}_s{args.seed}"
    runs_path = os.path.join(args.out_dir, f"{stem}_runs.csv")
    runs_df.to_csv(runs_path, index=False)

    if args.group_by_column:
        resampling = f"GroupKFold({N_FOLDS}) on {args.group_by_column}"
    elif args.task == "classification":
        resampling = f"StratifiedKFold({N_FOLDS}, shuffle, seed={args.seed})"
    else:
        resampling = f"KFold({N_FOLDS}, shuffle, seed={args.seed})"

    metrics = {
        "method": "auto-sklearn",
        "usecase": args.usecase,
        "task": args.task,
        "cores": args.cores,
        "seed": args.seed,
        "budget_s": args.budget_s,
        "n_configs_evaluated": int(len(runs_df)),
        "n_configs_success": int(len(success)),
        "metric": metric_name,
        "mode": mode,
        "best_val": best_val,
        "restricted_model": args.restricted_model,
        "enrich_with": args.enrich_with,
        "n_jobs": args.n_jobs,
        "threads_per_job": args.threads_per_job,
        "memory_limit_mb": args.memory_limit_mb,
        "per_run_time_limit_s": per_run_time_limit,
        "resampling": resampling,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    metrics_path = os.path.join(args.out_dir, f"{stem}_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
        fh.write("\n")

    print(f"Wrote {metrics_path} and {runs_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
