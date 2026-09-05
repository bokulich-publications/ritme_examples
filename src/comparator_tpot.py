"""TPOT comparator arm for the usecase regression and classification tasks.

Runs classic (genetic-programming) TPOT 0.12.2 against the same train/test
split, metadata enrichment and wall-clock budget as the ritme and auto-sklearn
arms. The estimator family is pinned to match the ritme winner while the
preprocessing operators stay searchable, so the run still exercises joint
feature-model optimization.
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import re
from typing import Optional

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
    # ritme's u3 winner by validation AUROC is logreg (u3_logreg_tpe_no_fit,
    # 1-SE rule) since fit_result was dropped; LogisticRegression is the direct
    # analogue. The former XGBClassifier pin is archived under
    # comparators/archive_u3_xgb_pinned/.
    "u3": "sklearn.linear_model.LogisticRegression",
    # ritme's u4 winner by validation AUROC is xgb_class
    "u4": "xgboost.XGBClassifier",
}

N_FOLDS = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-time-s", type=int, required=True)
    p.add_argument("--usecase", required=True)
    p.add_argument("--task", required=True, choices=["regression", "classification"])
    p.add_argument(
        "--population-size",
        type=int,
        default=100,
        help="TPOT population per generation (default: TPOT's 100). A smoke "
        "run needs a population its budget can evaluate at least once.",
    )
    p.add_argument(
        "--xgb-tree-method",
        default=None,
        help="Override the restricted XGB estimator's tree_method (e.g. hist).",
    )
    p.add_argument(
        "--xgb-threads",
        type=int,
        default=8,
        help="Threads per XGB fit when --xgb-tree-method is set.",
    )
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
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "TPOT `periodic_checkpoint_folder`: writes each new pareto-front "
            "pipeline as exportable source, once per generation. Affects "
            "persistence only, not the search. Needed on u1, where TPOT's "
            "stopit timeout can corrupt XGBoost's heap and kill the process."
        ),
    )
    p.add_argument(
        "--recover-from",
        default=None,
        help=(
            "Skip the search: rebuild the best pipeline from this checkpoint "
            "directory, fit it on train_val and evaluate on test."
        ),
    )
    p.add_argument(
        "--recover-log",
        default=None,
        help=(
            "Job log of the crashed run. Its tqdm progress line gives an "
            "upper bound on pipelines evaluated (it counts GP re-proposals, "
            "which evaluated_individuals_ deduplicates)."
        ),
    )
    return p.parse_args()


def attempted_count_from_log(log_path: str) -> Optional[int]:
    """Pipelines *attempted*, read from TPOT's progress bar in a job log.

    An upper bound on `len(evaluated_individuals_)`, not the same quantity:
    the bar counts every individual the GP proposes, while the dict is keyed
    by pipeline string and deduplicates re-proposals. Measured inflation on
    clean runs here was +2.0% (u3: 11001 vs 10781) and +26.5% (u2: 5126 vs
    4052), so no fixed correction factor applies. Use only when a crash has
    destroyed the dict, and label the result as a bound.

    tqdm emits two forms and switches to the second once the counter passes
    the initial total, so both must be matched:

        Optimization Progress:  98%|####| 6303/6400 [...]
        Optimization Progress: 6401pipeline [...]

    Counters restart per run and job logs are opened in append mode, so only
    the trailing run's segment is considered.
    """
    fraction = re.compile(r"Optimization Progress:[^|]*\|[^|]*\|\s*(\d+)/\d+")
    bare = re.compile(r"Optimization Progress:\s*(\d+)pipeline")
    counts: list[int] = []
    try:
        with open(log_path, errors="replace") as fh:
            for line in fh:
                for pattern in (fraction, bare):
                    counts.extend(int(m.group(1)) for m in pattern.finditer(line))
    except OSError:
        return None
    if not counts:
        return None
    start_idx = 0
    for i in range(1, len(counts)):
        if counts[i] < counts[i - 1]:
            start_idx = i
    return max(counts[start_idx:])


# Names assigned by the data-loading preamble of a TPOT export, which cannot be
# re-executed here (it reads a CSV placeholder path).
_EXPORT_SKIP_TARGETS = frozenset(
    {
        "tpot_data",
        "features",
        "training_features",
        "testing_features",
        "training_target",
        "testing_target",
        "results",
    }
)


def _keep_export_node(node: ast.stmt) -> bool:
    """Whether a statement of a TPOT export is needed to rebuild the pipeline.

    Keeps imports, the ``exported_pipeline`` assignment and the random_state
    fix-up (which TPOT emits either as a bare ``set_param_recursive`` call or,
    for a single-estimator pipeline, as an ``if hasattr(...)`` guard). Drops the
    data-loading preamble and the fit/predict calls.
    """
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True
    if isinstance(node, ast.Assign):
        names = set()
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Tuple):
                names.update(e.id for e in target.elts if isinstance(e, ast.Name))
        return not (names & _EXPORT_SKIP_TARGETS)
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        func = node.value.func
        if isinstance(func, ast.Attribute) and func.attr in {"fit", "predict"}:
            return False
    return True


def _checkpoint_score(path: str) -> Optional[float]:
    """Internal CV score recorded in a TPOT export header, if present."""
    with open(path) as fh:
        for line in fh:
            if line.startswith("# Average CV score"):
                try:
                    return float(line.rsplit(":", 1)[1])
                except ValueError:
                    return None
    return None


def load_checkpoint_pipeline(checkpoint_dir: str):
    """Rebuild the best-scoring pipeline from a `periodic_checkpoint_folder`.

    TPOT writes exportable source per pareto-front pipeline, not a fitted
    model. The front trades accuracy against pipeline complexity, so the most
    recent file is frequently a longer, worse-scoring member -- selection is on
    the recorded CV score (higher is better for every TPOT scorer), falling
    back to file age only when no header carries one.
    """
    files = glob.glob(os.path.join(checkpoint_dir, "pipeline_gen_*.py"))
    if not files:
        raise FileNotFoundError(f"No pipeline_gen_*.py in {checkpoint_dir}")

    scored = [(f, _checkpoint_score(f)) for f in files]
    with_score = [(f, s) for f, s in scored if s is not None]
    if with_score:
        newest, score = max(with_score, key=lambda fs: fs[1])
    else:
        newest = max(files, key=os.path.getmtime)
        score = None
    source = open(newest).read()

    # The exported file also loads a CSV it cannot find and fits the pipeline;
    # drop those statements and keep the imports, the pipeline definition and
    # the random_state fix-up. Selecting on the AST rather than on text is what
    # makes this safe: TPOT's train_test_split call is split over two lines with
    # a backslash continuation, so line filtering leaves an orphaned indented
    # fragment behind and the exec fails with IndentationError.
    module = ast.Module(
        body=[n for n in ast.parse(source).body if _keep_export_node(n)],
        type_ignores=[],
    )
    namespace: dict = {}
    exec(compile(module, newest, "exec"), namespace)  # noqa: S102 - TPOT-generated
    print(f"Recovered {newest} (internal CV score {score})")
    return namespace["exported_pipeline"], score, len(files), newest


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


def _evaluate_and_write(
    args,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    model_label: str,
    configs: pd.DataFrame,
    n_evaluated: Optional[int],
    exporter=None,
    recovered: bool = False,
    n_checkpoints: Optional[int] = None,
    n_attempted_log: Optional[int] = None,
) -> None:
    """Score a fitted pipeline and write the arm's four output files."""
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
    metrics["n_configs_evaluated"] = n_evaluated
    metrics["recovered_from_checkpoint"] = recovered
    if n_checkpoints is not None:
        metrics["n_checkpoints"] = n_checkpoints
    if n_attempted_log is not None:
        metrics["n_configs_attempted_log_upper_bound"] = n_attempted_log

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = write_metrics(args.out_dir, args.usecase, "tpot", metrics)
    configs_path = write_configs(args.out_dir, args.usecase, "tpot", configs)
    written = [("Metrics", metrics_path), ("Configs", configs_path)]

    if exporter is not None:
        pipeline_path = os.path.join(
            args.out_dir, f"{args.usecase}_tpot_best_pipeline.py"
        )
        exporter(pipeline_path)
        written.append(("Pipeline", pipeline_path))

    fig_path = os.path.join(args.out_dir, f"{args.usecase}_tpot_best_{fig_suffix}.png")
    fig.savefig(fig_path, bbox_inches="tight")
    written.append(("Plot", fig_path))

    for label, path in written:
        print(f"{label} written to {path}")


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

    if args.recover_from:
        model, cv_score, n_checkpoints, newest = load_checkpoint_pipeline(
            args.recover_from
        )
        model.fit(X_train, y_train)
        _evaluate_and_write(
            args,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            model_label=args.restricted_model
            or ESTIMATOR_FOR_USECASE.get(args.usecase, ""),
            configs=pd.DataFrame([{"pipeline": newest, "internal_cv_score": cv_score}]),
            # Unknown after a crash: the dict is destroyed and the log only
            # bounds it from above.
            n_evaluated=None,
            n_attempted_log=(
                attempted_count_from_log(args.recover_log) if args.recover_log else None
            ),
            n_checkpoints=n_checkpoints,
            recovered=True,
        )
        return

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
        if args.xgb_tree_method and estimator_path in config:
            # TPOT's default XGB entry uses the exact method on one thread;
            # at ~3x10^5 features a single fit then outlives any per-eval
            # timeout, and stopit's async interrupt inside native XGBoost
            # code corrupts the heap. hist + threads makes one fit feasible.
            config[estimator_path] = {
                **config[estimator_path],
                "tree_method": [args.xgb_tree_method],
                "n_jobs": [args.xgb_threads],
            }
            print(
                f"XGB overrides: tree_method={args.xgb_tree_method}, "
                f"n_jobs={args.xgb_threads}"
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
        population_size=args.population_size,
        cv=make_cv(args.usecase, args.task, args.seed),
        # roc_auc is binary-only; the ovr variant matches ritme's
        # roc_auc_macro_ovr objective on multi-class targets.
        scoring=(
            ("roc_auc" if y_train.nunique() <= 2 else "roc_auc_ovr")
            if args.task == "classification"
            else "neg_root_mean_squared_error"
        ),
        max_time_mins=max(1, args.total_time_s // 60),
        max_eval_time_mins=args.max_eval_time_mins,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        disable_update_check=True,
        verbosity=2,
        periodic_checkpoint_folder=args.checkpoint_dir,
    )
    est.fit(X_train, y_train, groups=groups)
    print(f"Evaluated {len(est.evaluated_individuals_)} pipelines")
    print("Best pipeline:", est.fitted_pipeline_)

    # The fitted sklearn pipeline, not the TPOT wrapper: it carries `classes_`
    # and a plain predict_proba, which the shared evaluators expect.
    _evaluate_and_write(
        args,
        est.fitted_pipeline_,
        X_train,
        y_train,
        X_test,
        y_test,
        model_label=model_label,
        configs=evaluated_individuals_frame(est),
        n_evaluated=len(est.evaluated_individuals_),
        exporter=est.export,
    )


if __name__ == "__main__":
    main()
