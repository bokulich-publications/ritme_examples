"""B2 auto-sklearn arm: search-only run at a fixed wall-clock budget.

Runs inside the `autosklearn` conda env on one SLURM job per (cores, seed).
Mirrors ritme's U1 protocol: same train_val split and grouped 5-fold CV on
`host_id`, model space restricted to `gradient_boosting` (the xgb match).
Writes exactly two small files per run — a metrics JSON and a per-evaluated-
configuration CSV — and keeps auto-sklearn's temporary directory on the
node-local scratch, deleted when the job exits.

Submitted by `benchmarking.launch_b2`, which passes every setting that has
to match the rest of the sweep; running it by hand needs all of them:

    python -m benchmarking.automl_b2 --budget-s 7200 --cores 8 --n-jobs 2 \
        --threads-per-job 4 --memory-limit-mb 14336 --seed 0 \
        --out-dir benchmarking/runs/b2/automl
"""

from __future__ import annotations

import argparse
import json
import os

import autosklearn.regression
import pandas as pd
from autosklearn.ensembles import SingleBest
from autosklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from smac.tae import StatusType

from src.generic_automl import _read_split

DATA_SPLITS = "use_cases/u1_amplicon_age_prediction/data_splits_u1"
PATH_FT = "data/u1_subramanian14/otu_table_subr14_wq.tsv"
TARGET = "age_months"
GROUP_COLUMN = "host_id"
N_FOLDS = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
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
    p.add_argument("--restricted-model", default="gradient_boosting")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_jobs = args.n_jobs
    os.makedirs(args.out_dir, exist_ok=True)

    train_df = _read_split(DATA_SPLITS, "train_val")
    otu_df = pd.read_csv(PATH_FT, sep="\t", index_col=0)
    otu_df = otu_df.div(otu_df.sum(axis=1), axis=0)
    X_train = otu_df.loc[train_df.index]
    y_train = train_df[TARGET]
    groups = train_df[GROUP_COLUMN].to_numpy()
    print(
        f"Loaded X{X_train.shape}, {len(set(groups))} groups; "
        f"cores={args.cores} n_jobs={n_jobs} "
        f"threads/job={args.threads_per_job} "
        f"seed={args.seed} budget={args.budget_s}s",
        flush=True,
    )

    # auto-sklearn's default is budget/10; floor it so short smoke runs can
    # still finish a configuration, and keep it below half the budget.
    per_run_time_limit = min(max(300, args.budget_s // 10), args.budget_s // 2)
    tmp_root = os.environ.get("TMPDIR", "/tmp")
    tmp_folder = os.path.join(tmp_root, f"askl_b2_c{args.cores}_s{args.seed}")

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=args.budget_s,
        per_run_time_limit=per_run_time_limit,
        n_jobs=n_jobs,
        seed=args.seed,
        memory_limit=args.memory_limit_mb,
        metric=root_mean_squared_error,
        include={"regressor": [args.restricted_model]},
        ensemble_class=SingleBest,
        resampling_strategy=GroupKFold(n_splits=N_FOLDS),
        resampling_strategy_arguments={"groups": groups},
        # `groups` is positional w.r.t. the rows of X, so any row subsampling
        # would silently misalign it; keep the data exactly as ritme sees it.
        dataset_compression=False,
        tmp_folder=tmp_folder,
        delete_tmp_folder_after_terminate=True,
    )
    print("Starting fit", flush=True)
    automl.fit(X_train, y_train, dataset_name=f"u1_b2_c{args.cores}_s{args.seed}")
    print("Fit returned", flush=True)

    # RunValue.cost is the validation loss; for the RMSE scorer
    # (optimum 0, greater_is_better False) that is the plain CV-mean RMSE.
    run_rows = []
    for run_key, run_value in automl.automl_.runhistory_.data.items():
        run_rows.append(
            {
                "config_id": run_key.config_id,
                "status": run_value.status.name,
                "cost_rmse_val": run_value.cost,
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
            "cost_rmse_val",
            "duration_s",
            "starttime",
            "endtime",
        ],
    )
    success = runs_df[runs_df["status"] == StatusType.SUCCESS.name]

    stem = f"automl_c{args.cores}_s{args.seed}"
    runs_path = os.path.join(args.out_dir, f"{stem}_runs.csv")
    runs_df.to_csv(runs_path, index=False)

    metrics = {
        "method": "auto-sklearn",
        "cores": args.cores,
        "seed": args.seed,
        "budget_s": args.budget_s,
        "n_configs_evaluated": int(len(runs_df)),
        "n_configs_success": int(len(success)),
        "best_rmse_val": (
            float(success["cost_rmse_val"].min()) if len(success) else None
        ),
        "restricted_model": args.restricted_model,
        "n_jobs": n_jobs,
        "threads_per_job": args.threads_per_job,
        "memory_limit_mb": args.memory_limit_mb,
        "per_run_time_limit_s": per_run_time_limit,
        "resampling": f"GroupKFold({N_FOLDS}) on {GROUP_COLUMN}",
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
