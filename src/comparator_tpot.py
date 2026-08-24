"""TPOT comparator arm for the usecase regression and classification tasks.

Runs classic (genetic-programming) TPOT 0.12.2 against the same train/test
split, metadata enrichment and wall-clock budget as the ritme and auto-sklearn
arms. The estimator family is pinned to match the ritme winner while the
preprocessing operators stay searchable, so the run still exercises joint
feature-model optimization.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from tpot import TPOTClassifier, TPOTRegressor
from tpot.config.classifier import classifier_config_dict
from tpot.config.regressor import regressor_config_dict

from src.comparator_common import load_xy, write_configs, write_metrics
from src.eval_automl import get_metrics_n_roc_curve, get_metrics_n_scatterplot
from src.launch_models import USECASES

# Everything in TPOT's default configs that is not a final estimator.
OPERATOR_PREFIXES = (
    "sklearn.preprocessing.",
    "sklearn.decomposition.",
    "sklearn.kernel_approximation.",
    "sklearn.feature_selection.",
    "sklearn.cluster.FeatureAgglomeration",
    "tpot.builtins.",
)

# Operators dropped from the search because they cannot run at these feature
# counts. PolynomialFeatures(degree=2) expands p features to p*(p+1)/2: 63.7M
# columns for u3's 11,285 and 635M for u2's 35,651, i.e. hundreds of terabytes
# dense. Classic TPOT applies no per-evaluation memory cap, so a single such
# individual OOM-kills the whole job. Pass --allow-infeasible-operators to
# restore them.
INFEASIBLE_OPERATORS = ("sklearn.preprocessing.PolynomialFeatures",)

# Closest TPOT estimator to each usecase's ritme winner. Pick from TPOT's own
# catalogue rather than reusing the auto-sklearn arm's substitutions: that arm
# has no ElasticNet and had to stand `sgd` in for ritme's linreg, a handicap
# TPOT does not share.
ESTIMATOR_FOR_USECASE = {
    # ritme xgb -> the same library, not sklearn's GradientBoosting
    "u1": "xgboost.XGBRegressor",
    # ritme linreg is StandardScaler + ElasticNet (static_trainables.py:195-207);
    # ElasticNetCV is the direct analogue. NB it does not expose alpha - it picks
    # it by internal CV over a 100-point path, so its alpha search is not reduced.
    "u2": "sklearn.linear_model.ElasticNetCV",
    # ritme xgb_class -> the same library
    "u3": "xgboost.XGBClassifier",
}

N_FOLDS = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-time-s", type=int, required=True)
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
        "--restricted-model",
        default=None,
        help="TPOT estimator path. Defaults to ESTIMATOR_FOR_USECASE[usecase].",
    )
    p.add_argument(
        "--unrestricted",
        action="store_true",
        help="Search TPOT's full default config instead of one estimator family.",
    )
    p.add_argument(
        "--allow-infeasible-operators",
        action="store_true",
        help=f"Keep operators normally dropped for memory: {INFEASIBLE_OPERATORS}.",
    )
    p.add_argument(
        "--generations",
        type=int,
        default=None,
        help="GP generations. Default None lets --total-time-s bind instead.",
    )
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--max-eval-time-mins", type=int, default=30)
    p.add_argument("--out-dir", default="comparators")
    return p.parse_args()


def default_config(task: str) -> dict:
    return classifier_config_dict if task == "classification" else regressor_config_dict


def drop_infeasible(config: dict, allow_infeasible: bool) -> dict:
    if allow_infeasible:
        return config
    dropped = [k for k in config if k in INFEASIBLE_OPERATORS]
    if dropped:
        print(f"Dropping operators infeasible at this feature count: {dropped}")
    return {k: v for k, v in config.items() if k not in INFEASIBLE_OPERATORS}


def restricted_config(task: str, estimator_path: str) -> dict:
    """TPOT config keeping all preprocessing operators and one estimator."""
    full = default_config(task)
    if estimator_path not in full:
        raise KeyError(
            f"{estimator_path!r} is not in TPOT's default {task} config. "
            f"Available estimators: "
            f"{sorted(k for k in full if not k.startswith(OPERATOR_PREFIXES))}"
        )
    return {
        k: v
        for k, v in full.items()
        if k.startswith(OPERATOR_PREFIXES) or k == estimator_path
    }


def make_cv(usecase: str, task: str, seed: int):
    """CV splitter matching ritme's validation protocol for this usecase."""
    if USECASES[usecase]["group_by_column"]:
        return GroupKFold(n_splits=N_FOLDS)
    if task == "classification":
        return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)


def evaluated_individuals_frame(est) -> pd.DataFrame:
    rows = [
        {
            "pipeline": pipeline,
            "internal_cv_score": stats.get("internal_cv_score"),
            "generation": stats.get("generation"),
            "operator_count": stats.get("operator_count"),
        }
        for pipeline, stats in est.evaluated_individuals_.items()
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    spec = USECASES[args.usecase]

    X_train, y_train, X_test, y_test, groups = load_xy(
        args.path_to_features,
        args.path_to_md,
        args.data_splits_folder,
        args.target,
        args.task,
        group_by_column=spec["group_by_column"],
        enrich_with=args.enrich_with,
    )
    print(f"Enriched with {args.enrich_with}")
    print("X_train.shape", X_train.shape)
    print("X_test.shape", X_test.shape)

    if args.unrestricted:
        config = drop_infeasible(
            default_config(args.task), args.allow_infeasible_operators
        )
        model_label = "unrestricted"
    else:
        estimator_path = args.restricted_model or ESTIMATOR_FOR_USECASE[args.usecase]
        config = drop_infeasible(
            restricted_config(args.task, estimator_path),
            args.allow_infeasible_operators,
        )
        model_label = estimator_path
        print(f"Restricted to {estimator_path} plus {len(config) - 1} operators")

    Estimator = TPOTClassifier if args.task == "classification" else TPOTRegressor
    est = Estimator(
        config_dict=config,
        # generations=None makes TPOT schedule 1e6 generations so max_time_mins
        # is the binding stop condition. Left at its default of 100, TPOT ends
        # the search once those generations are evaluated -- well inside the
        # matched budget -- and the arms are no longer budget-comparable.
        generations=args.generations,
        cv=make_cv(args.usecase, args.task, args.seed),
        scoring=(
            "roc_auc"
            if args.task == "classification"
            else "neg_root_mean_squared_error"
        ),
        max_time_mins=max(1, args.total_time_s // 60),
        max_eval_time_mins=args.max_eval_time_mins,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        disable_update_check=True,
        verbosity=2,
    )
    est.fit(X_train, y_train, groups=groups)
    print(f"Evaluated {len(est.evaluated_individuals_)} pipelines")
    print("Best pipeline:", est.fitted_pipeline_)

    # The fitted sklearn pipeline, not the TPOT wrapper: it carries `classes_`
    # and a plain predict_proba, which the shared evaluators expect.
    model = est.fitted_pipeline_
    if args.task == "classification":
        metrics, fig = get_metrics_n_roc_curve(model, X_train, y_train, X_test, y_test)
        fig_suffix = "roc"
    else:
        metrics, fig = get_metrics_n_scatterplot(
            model, X_train, y_train, X_test, y_test
        )
        fig_suffix = "true_vs_pred"

    metrics["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    metrics["restricted_model"] = model_label
    metrics["n_configs_evaluated"] = len(est.evaluated_individuals_)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = write_metrics(args.out_dir, args.usecase, "tpot", metrics)
    configs_path = write_configs(
        args.out_dir, args.usecase, "tpot", evaluated_individuals_frame(est)
    )
    pipeline_path = os.path.join(args.out_dir, f"{args.usecase}_tpot_best_pipeline.py")
    est.export(pipeline_path)
    fig_path = os.path.join(args.out_dir, f"{args.usecase}_tpot_best_{fig_suffix}.png")
    fig.savefig(fig_path, bbox_inches="tight")

    for label, path in [
        ("Metrics", metrics_path),
        ("Configs", configs_path),
        ("Pipeline", pipeline_path),
        ("Plot", fig_path),
    ]:
        print(f"{label} written to {path}")


if __name__ == "__main__":
    main()
