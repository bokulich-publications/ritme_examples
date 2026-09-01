"""Score the model mAML's CLI dumped against the held-out test split.

Runs inside the `maml_legacy` environment, because the pickled estimator holds
scikit-learn 0.21 objects and upstream's own scaler classes. It emits only
predicted probabilities; `comparator_maml.py` computes the metrics in the host
environment through the same evaluator every other arm uses, so no metric
definition depends on the legacy stack.

Imports are limited to pandas, numpy and joblib so the module needs no package
install in the legacy environment - it is invoked by path.
"""

from __future__ import print_function

import argparse
import glob
import os
import sys

import pandas as pd
from joblib import load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--usecase", required=True)
    p.add_argument(
        "--maml-src", required=True, help="mAML checkout; the pickle needs it"
    )
    p.add_argument("--prevalence", default="0.2")
    p.add_argument(
        "--expect-mrmr",
        action="store_true",
        help="fail if mRMR wrote no output. Upstream wraps each pipeline step "
        "in @tryExcept, which prints a traceback and continues, so a failed "
        "selection step otherwise silently changes the configuration.",
    )
    args = p.parse_args()

    # The dumped pipeline holds upstream's own scaler classes.
    sys.path.insert(0, os.path.join(args.maml_src, "code"))

    models = glob.glob(os.path.join(args.run_dir, "*.model.z"))
    if len(models) != 1:
        raise SystemExit("expected exactly one dumped model, found %r" % models)
    model = load(models[0])

    # The CLI rewrites the matrix after each reduction step; the last one
    # written is the space the pipeline was fitted on, in the order it saw it.
    mrmr = glob.glob(os.path.join(args.run_dir, "*.mrmr_sel_features.csv"))
    if args.expect_mrmr and not mrmr:
        raise SystemExit(
            "mRMR was requested but wrote no output: the run fell back to the "
            "prevalence-filtered space and is NOT the configuration it is "
            "labelled as. Check the log for 'Error execute: mrmr_feature_select'."
        )
    filt = glob.glob(
        os.path.join(args.run_dir, "*.filter_%s_prevalence.csv" % args.prevalence)
    )
    source = mrmr or filt
    if len(source) != 1:
        raise SystemExit("expected exactly one feature-space file, found %r" % source)
    keep = pd.read_csv(source[0], index_col=0, nrows=1).columns.tolist()
    print(
        "feature space from", os.path.basename(source[0]), "->", len(keep), "features"
    )

    for split in ("train", "test"):
        X = pd.read_csv(
            os.path.join(args.run_dir, "%s_%s_X.csv" % (args.usecase, split)),
            index_col=0,
        )
        y = pd.read_csv(
            os.path.join(args.run_dir, "%s_%s_Y.csv" % (args.usecase, split)),
            index_col=0,
        )
        X = X.reindex(columns=keep)
        if X.isnull().any().any():
            raise SystemExit("split %s is missing columns the model needs" % split)
        proba = model.predict_proba(X)
        df = pd.DataFrame(
            proba, index=X.index, columns=["proba_%s" % c for c in model.classes_]
        )
        df["y_true"] = y.iloc[:, 0].reindex(X.index)
        df.to_csv(os.path.join(args.run_dir, "predictions_%s.csv" % split))
        print("%s: %s -> %s" % (split, X.shape, df.shape))

    with open(os.path.join(args.run_dir, "winner.txt"), "w") as fh:
        fh.write(os.path.basename(models[0]) + "\n")
    print("model:", os.path.basename(models[0]))


if __name__ == "__main__":
    main()
