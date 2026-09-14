"""Write the per-sample train/test predictions the supplementary figures need.

One CSV per (use case, arm) under ``predictions/``:

    sample_id, split, y_true, y_pred            regression (u1, u2)
    sample_id, split, y_true, p_<class>, ...    classification (u3, u4)

The ``ritme`` arm reloads the saved best model and only predicts. The
``original`` and ``automl`` arms refit -- neither the published baselines nor
TPOT persist a fitted estimator -- from the recipe and the exported pipeline
respectively, both seeded, so the refit reproduces the recorded metrics.

Usage (repo root):
    conda activate ritme_usecases
    python -m final_figures.supp_scatter_roc.extract_predictions u2 original
    python -m final_figures.supp_scatter_roc.extract_predictions u2 ritme
    conda activate tpot_bench
    python -m final_figures.supp_scatter_roc.extract_predictions u2 automl

`submit_predictions.sh` submits all twelve as SLURM jobs with the resources
each needs.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from final_figures.supp_scatter_roc.arms import (
        ARMS,
        PRED_DIR,
        REPO_ROOT,
        TASK,
        TPOT_CHECKPOINTS,
        TPOT_EXPORT,
        USECASES,
        prediction_path,
        select_ritme_run,
    )
except ImportError:  # running from inside final_figures/supp_scatter_roc/
    from arms import (  # type: ignore[no-redef]
        ARMS,
        PRED_DIR,
        REPO_ROOT,
        TASK,
        TPOT_CHECKPOINTS,
        TPOT_EXPORT,
        USECASES,
        prediction_path,
        select_ritme_run,
    )


def _write(usecase: str, arm: str, per_split: dict) -> Path:
    """Flatten ``{split: (y_true, y_pred[, classes])}`` into one tidy CSV."""
    frames = []
    for split, payload in per_split.items():
        y_true = payload[0]
        index = pd.Index(y_true.index, name="sample_id")
        out = pd.DataFrame({"split": split, "y_true": np.asarray(y_true)}, index=index)
        if len(payload) == 2:
            out["y_pred"] = np.asarray(payload[1])
        else:
            proba, classes = payload[1], payload[2]
            for i, cls in enumerate(classes):
                out[f"p_{cls}"] = proba[:, i]
        frames.append(out.reset_index())

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = prediction_path(usecase, arm)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def _report(usecase: str, per_split: dict) -> None:
    """Print the headline metric per split, to check a refit against the record."""
    from sklearn.metrics import mean_squared_error, roc_auc_score

    for split, payload in per_split.items():
        y_true = np.asarray(payload[0])
        if len(payload) == 2:
            score = np.sqrt(mean_squared_error(y_true, payload[1]))
            print(f"  {split}: RMSE {score:.4f}")
        else:
            proba, classes = payload[1], payload[2]
            if len(classes) == 2:
                score = roc_auc_score((y_true == classes[1]).astype(int), proba[:, 1])
            else:
                score = roc_auc_score(
                    y_true, proba, multi_class="ovr", average="macro", labels=classes
                )
            print(f"  {split}: ROC AUC {score:.4f}")


def extract_original(usecase: str) -> dict:
    try:
        from final_figures.supp_scatter_roc import originals
    except ImportError:
        import originals  # type: ignore[no-redef]

    return originals.predictions(usecase)


def extract_ritme(usecase: str) -> dict:
    from ritme.evaluate_models import load_best_model, load_experiment_config

    from src.launch_models import USECASES as SPECS

    tag, model_type, exp_dir = select_ritme_run(usecase)
    print(f"ritme winner: {tag} ({model_type})")
    tmodel = load_best_model(model_type, str(exp_dir))
    target = load_experiment_config(str(exp_dir))["target"]
    splits_dir = REPO_ROOT / SPECS[usecase]["data_splits"]

    per_split = {}
    # One split at a time, train first: `TunedModel.build_design_matrix` pins
    # the engineered column set on the train pass, and u4's raw test frame is
    # tens of gigabytes on its own. Prediction must see a split whole --
    # ritme's variance-based selectors recompute variances on the batch they
    # are given, so chunking would change the output.
    for split, name in (("train", "train_val"), ("test", "test")):
        data = pd.read_pickle(splits_dir / f"{name}.pkl")
        y_true = data[target]
        if TASK[usecase] == "regression":
            per_split[split] = (y_true, tmodel.predict(data, split))
        else:
            proba, classes = tmodel.predict_proba(data, split)
            per_split[split] = (y_true, proba, list(classes))
        del data
    return per_split


def _rebuild_export(path: Path):
    """Instantiate ``exported_pipeline`` from a TPOT export, unfitted.

    The export also loads a placeholder CSV and fits; `_keep_export_node`
    selects, on the AST, the statements that build the pipeline.
    """
    from src.comparator_tpot import _keep_export_node

    source = path.read_text()
    module = ast.Module(
        body=[n for n in ast.parse(source).body if _keep_export_node(n)],
        type_ignores=[],
    )
    namespace: dict = {}
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - TPOT-generated
    return namespace["exported_pipeline"]


def extract_automl(usecase: str) -> dict:
    from src.comparator_common import load_xy
    from src.comparator_tpot import load_checkpoint_pipeline
    from src.launch_automl import _read_enrich_with, _read_target
    from src.launch_models import USECASES as SPECS

    spec = SPECS[usecase]
    target = _read_target(usecase)
    X_train, y_train, X_test, y_test, _ = load_xy(
        str(REPO_ROOT / spec["path_ft"]),
        str(REPO_ROOT / spec["path_md"]),
        str(REPO_ROOT / spec["data_splits"]),
        target,
        spec["task"],
        group_by_column=spec["group_by_column"],
        enrich_with=_read_enrich_with(usecase),
    )
    print(f"X_train {X_train.shape} | X_test {X_test.shape}")

    if usecase in TPOT_EXPORT:
        model = _rebuild_export(TPOT_EXPORT[usecase])
    else:
        model, score, _, newest = load_checkpoint_pipeline(
            str(TPOT_CHECKPOINTS[usecase])
        )
        print(f"rebuilt {Path(newest).name} (internal CV score {score})")
    model.fit(X_train, y_train)

    if TASK[usecase] == "regression":
        return {
            "train": (y_train, model.predict(X_train)),
            "test": (y_test, model.predict(X_test)),
        }

    # `load_xy` encodes string targets as 0..k-1 in sorted order; map back so
    # every arm of a use case labels its classes the same way.
    md = pd.read_csv(REPO_ROOT / spec["path_md"], sep="\t", index_col=0)
    raw = pd.concat([md.loc[X_train.index, target], md.loc[X_test.index, target]])
    try:
        raw.astype(int)
        names = None
    except (ValueError, TypeError):
        names = sorted(raw.dropna().unique())
    classes = [names[i] for i in model.classes_] if names else list(model.classes_)

    def decode(y):
        return pd.Series([names[i] for i in y], index=y.index) if names else y

    return {
        "train": (decode(y_train), model.predict_proba(X_train), classes),
        "test": (decode(y_test), model.predict_proba(X_test), classes),
    }


EXTRACTORS = {
    "original": extract_original,
    "ritme": extract_ritme,
    "automl": extract_automl,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("usecase", choices=USECASES)
    p.add_argument("arm", choices=ARMS)
    args = p.parse_args()

    per_split = EXTRACTORS[args.arm](args.usecase)
    _report(args.usecase, per_split)
    print(f"written to {_write(args.usecase, args.arm, per_split)}")


if __name__ == "__main__":
    main()
