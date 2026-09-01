"""mAML comparator arm - the published pipeline, run unmodified.

mAML ships as a CLI (`code/sklearn_pipeline.py` in yangfenglong/mAML1.0) that
fits on the data it is given and dumps the winning pipeline with joblib. That
CLI is run as published rather than reimplemented, so this module only handles
the two ends the CLI does not: exporting the *ritme* split into the two CSVs it
expects, and turning the probabilities it yields into the same metric set every
other arm reports.

The CLI needs scikit-learn < 0.24 (it passes the `iid` argument to
`GridSearchCV`, removed in 0.24), so it runs in its own `maml_legacy`
environment while these two stages run in the host environment. Scoring lives
in `comparator_maml_score.py`, which the legacy environment can import.

Configuration follows the publication (Yang & Zou, *Database* 2020,
`10.1093/database/baaa050`), which documents a 20% prevalence filter, mRMR
selection to the top 50 features, SMOTE rebalancing and a joint search over
preprocessor x classifier with simultaneous hyperparameter tuning, scored on
accuracy - the same settings as all 18 benchmark runs in the repo's
`results/work.sh`.

Only the split is ours: the CLI sees `train_val` alone, so its supervised steps
(prevalence filter, mRMR, SMOTE) are fitted on training data and the `test`
split reaches the model exactly once, through the dumped estimator.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from src.comparator_common import load_xy, write_configs, write_metrics
from src.eval_automl import get_metrics_n_roc_curve

# Upstream's class names reach `re.sub`, so numeric labels raise. Names are
# zero-padded because `LabelEncoder` sorts lexicographically: unpadded
# "class10" would sort between "class1" and "class2" and silently permute the
# encoding relative to the one `load_xy` produced.
CLASS_FMT = "class{:02d}"


def _class_names(y: pd.Series) -> pd.Series:
    return y.astype(int).map(CLASS_FMT.format)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("export", "collect"))
    p.add_argument("--usecase", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--run-dir", required=True, help="where the CLI writes its artifacts"
    )
    p.add_argument("--data-splits-folder")
    p.add_argument("--path-to-features")
    p.add_argument("--path-to-md")
    p.add_argument("--target")
    p.add_argument("--enrich-with", action="append", default=[])
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--n-jobs", type=int, default=1)
    return p.parse_args()


def export(args: argparse.Namespace) -> None:
    """Write the split as the two CSVs the CLI reads."""
    if args.task != "classification":
        raise ValueError(
            f"mAML is classification-only; {args.usecase!r} is {args.task!r}"
        )

    X_train, y_train, X_test, y_test, _ = load_xy(
        args.path_to_features,
        args.path_to_md,
        args.data_splits_folder,
        args.target,
        args.task,
        enrich_with=args.enrich_with or None,
    )
    print(f"Enriched with {args.enrich_with}")
    print("X_train.shape", X_train.shape)
    print("X_test.shape", X_test.shape)
    print("classes", len(np.unique(y_train)))

    os.makedirs(args.run_dir, exist_ok=True)
    uc = args.usecase
    X_train.to_csv(os.path.join(args.run_dir, f"{uc}_train_X.csv"))
    X_test.to_csv(os.path.join(args.run_dir, f"{uc}_test_X.csv"))
    _class_names(y_train).to_frame(args.target).to_csv(
        os.path.join(args.run_dir, f"{uc}_train_Y.csv")
    )
    _class_names(y_test).to_frame(args.target).to_csv(
        os.path.join(args.run_dir, f"{uc}_test_Y.csv")
    )
    print("exported to", args.run_dir)


class _DumpedModel:
    """Replays the saved probabilities so the shared evaluator can score them.

    `get_metrics_n_roc_curve` only reaches the estimator through
    `predict_proba`, so a lookup keyed on row count stands in for the pipeline
    itself, which cannot be unpickled outside the legacy environment.
    """

    def __init__(self, by_rows: dict[int, np.ndarray], classes: list[int]) -> None:
        self._by_rows = by_rows
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X) -> np.ndarray:
        return self._by_rows[len(X)]


def collect(args: argparse.Namespace) -> None:
    """Turn the CLI's probabilities into the shared metric set."""
    splits = {}
    for split in ("train", "test"):
        path = os.path.join(args.run_dir, f"predictions_{split}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; the scoring stage did not run")
        splits[split] = pd.read_csv(path, index_col=0)

    train, test = splits["train"], splits["test"]
    if len(train) == len(test):
        raise SystemExit("row counts collide; the row-count lookup is unsafe")

    proba_cols = [c for c in train.columns if c.startswith("proba_")]
    classes = [int(c.replace("proba_", "")) for c in proba_cols]
    # LabelEncoder sorts, so sorted class names index to the encoded values.
    names = sorted(set(train["y_true"]) | set(test["y_true"]))
    code = {n: i for i, n in enumerate(names)}

    model = _DumpedModel(
        {
            len(train): train[proba_cols].to_numpy(),
            len(test): test[proba_cols].to_numpy(),
        },
        classes,
    )
    metrics, fig = get_metrics_n_roc_curve(
        model,
        pd.DataFrame(index=train.index),
        train["y_true"].map(code).to_numpy(),
        pd.DataFrame(index=test.index),
        test["y_true"].map(code).to_numpy(),
    )

    winner = ""
    winner_path = os.path.join(args.run_dir, "winner.txt")
    if os.path.exists(winner_path):
        with open(winner_path) as fh:
            winner = fh.read().strip()

    metrics["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    metrics["winner"] = winner
    metrics["n_features_searched"] = _feature_count(args.run_dir)
    write_metrics(args.out_dir, args.usecase, "maml", metrics)

    cv = glob.glob(os.path.join(args.run_dir, "*.all.cv_results.csv"))
    if cv:
        write_configs(args.out_dir, args.usecase, "maml", pd.read_csv(cv[0]))

    if fig is not None:
        fig.savefig(
            os.path.join(args.out_dir, f"{args.usecase}_maml_best_roc.png"),
            bbox_inches="tight",
        )
    print(metrics.to_string())


def _feature_count(run_dir: str) -> int | None:
    """Width of the space the search actually ran on (post-mRMR if it ran)."""
    for pattern in ("*.mrmr_sel_features.csv", "*.filter_*_prevalence.csv"):
        hits = glob.glob(os.path.join(run_dir, pattern))
        if hits:
            return pd.read_csv(hits[0], index_col=0, nrows=1).shape[1]
    return None


def main() -> None:
    args = _parse_args()
    if args.stage == "export":
        export(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
