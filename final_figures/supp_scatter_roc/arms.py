"""Row registry for the supplementary train/test figures.

One 3x2 figure per use case: three model arms as rows, the train and the
held-out test split as columns.

``original``  the published pipeline of the source study, refit from the
              recipe in its ``n4_original_setup.ipynb``
``ritme``     the ritme run selected on the validation metric by ritme's
              1-SE + simplicity rule, the choice
              ``use_cases/evaluate_all_trials.ipynb`` makes
``automl``    TPOT, the best or tied-best autoML comparator in all four use
              cases; unlike auto-sklearn it persists its winning pipeline,
              so the arm can be rebuilt without repeating the search

Import-safe from every comparator environment: nothing here imports ritme,
scikit-learn or TPOT at module level.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "use_cases" / "ritme_runs" / "local"
OUT_DIR = Path(__file__).resolve().parent
PRED_DIR = OUT_DIR / "predictions"

USECASES = ("u1", "u2", "u3", "u4")
ARMS = ("original", "ritme", "automl")
SPLITS = ("train", "test")

TASK = {
    "u1": "regression",
    "u2": "regression",
    "u3": "classification",
    "u4": "classification",
}

TITLES = {
    "u1": "Use case 1",
    "u2": "Use case 2",
    "u3": "Use case 3",
    "u4": "Use case 4",
}

ARM_LABELS = {
    "original": "Original",
    "ritme": "ritme",
    "automl": "AutoML (TPOT)",
}

# Experiment tags in scope per use case, matching `RUN_PATTERNS` in
# make_fig4_config_insights.py and `experiment_tags_to_compare` in
# evaluate_all_trials.ipynb.
RUN_PATTERNS = {
    "u1": re.compile(r"^u1_[a-z_]+_tpe$"),
    "u2": re.compile(r"^u2_[a-z_]+_tpe$"),
    "u3": re.compile(r"^u3_[a-z_]+_tpe_no_fit$"),
    "u4": re.compile(r"^u4_[a-z_]+_tpe$"),
}

VAL_METRIC = {
    "regression": ("metrics.rmse_val", "min"),
    "classification": ("metrics.roc_auc_macro_ovr_val", "max"),
}

# TPOT persists the winning pipeline as exportable source. u1's run died
# mid-search and never exported, so its arm is rebuilt from the pareto-front
# checkpoints the same way `src.comparator_tpot --recover-from` rebuilt it.
TPOT_EXPORT = {
    "u2": REPO_ROOT / "comparators" / "u2_tpot_best_pipeline.py",
    "u3": REPO_ROOT / "comparators" / "u3_tpot_best_pipeline.py",
    "u4": REPO_ROOT / "comparators" / "u4_tpot_best_pipeline.py",
}
TPOT_CHECKPOINTS = {"u1": REPO_ROOT / "comparators" / "u1_tpot_checkpoints"}

# Environment each arm has to run in, for the sbatch launcher and the README.
ARM_ENV = {
    "original": "ritme_usecases",
    "ritme": "ritme_usecases",
    "automl": "tpot_bench",
}


def prediction_path(usecase: str, arm: str) -> Path:
    return PRED_DIR / f"{usecase}_{arm}.csv"


def select_ritme_run(usecase: str) -> tuple[str, str, Path]:
    """The in-scope ritme run winning on validation: (tag, model type, dir).

    Applies ritme's own 1-SE + simplicity rule per experiment and then ranks
    the per-experiment winners on the mean validation metric, reproducing
    the selection in `evaluate_all_trials.ipynb`.
    """
    from types import SimpleNamespace

    from ritme.evaluate_models import _select_best_with_one_se

    metric_col, mode = VAL_METRIC[TASK[usecase]]
    pattern = RUN_PATTERNS[usecase]

    def to_result(idx, row):
        return SimpleNamespace(
            config={
                k.removeprefix("params."): v
                for k, v in row.items()
                if k.startswith("params.") and not pd.isna(v)
            },
            metrics={
                k.removeprefix("metrics."): v
                for k, v in row.items()
                if k.startswith("metrics.") and not pd.isna(v)
            },
            path=None,
            # MLflow logs carry no Ray checkpoint state; every logged trial
            # counts as deployable, as it does in the evaluation notebook.
            checkpoint=True,
            error=None,
            _df_idx=idx,
        )

    picked = []
    for run in sorted(RUNS_DIR.glob(f"{usecase}_*")):
        if not pattern.fullmatch(run.name) or not (run / "mlflow_logs.csv").exists():
            continue
        trials = pd.read_csv(run / "mlflow_logs.csv", low_memory=False)
        mean_col, se_col = f"{metric_col}_mean", f"{metric_col}_se"
        if trials[[mean_col, se_col]].dropna().empty:
            best = trials.sort_values(metric_col, ascending=(mode == "min")).head(1)
        else:
            chosen = _select_best_with_one_se(
                [to_result(i, r) for i, r in trials.iterrows()],
                metric=metric_col.removeprefix("metrics."),
                mode=mode,
                model_type=trials["params.model"].iloc[0],
            )
            best = trials.loc[[chosen._df_idx]]
        picked.append(best.assign(**{"tags.experiment_tag": run.name}))

    if not picked:
        raise SystemExit(f"no ritme runs matching {pattern.pattern} under {RUNS_DIR}")

    winner = (
        pd.concat(picked)
        .sort_values(f"{metric_col}_mean", ascending=(mode == "min"))
        .iloc[0]
    )
    tag = winner["tags.experiment_tag"]
    return tag, winner["params.model"], RUNS_DIR / tag
