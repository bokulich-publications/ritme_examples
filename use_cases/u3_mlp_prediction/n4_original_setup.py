import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from xgboost import XGBRegressor

from src.eval_originals import get_metrics_n_scatterplot
from src.process_u3 import preprocess_data_for_model

plt.rcParams.update({"font.family": "DejaVu Sans"})
plt.style.use("seaborn-v0_8-pastel")


def shannon_diversity(df, eps=1e-9):
    # 1) compute proportions per sample
    p = df.div(df.sum(axis=1), axis=0)
    # 2) Shannon index
    return -(p * np.log(p + eps)).sum(axis=1)


def main(cohort: str, target: str, use_shannon: bool = True):
    # paths
    if target == "count_log10":
        data_splits_folder = f"data_splits_u3_{cohort}_log"
    else:
        data_splits_folder = f"data_splits_u3_{cohort}"
    path_to_features = f"../../data/u3_mlp_nishijima24/{cohort}_otu_table.tsv"
    path_to_md = f"../../data/u3_mlp_nishijima24/md_{cohort}.tsv"

    # load train/test indices
    ritme_train_df = pd.read_pickle(f"{data_splits_folder}/train_val.pkl")
    ritme_test_df = pd.read_pickle(f"{data_splits_folder}/test.pkl")
    train_idx = ritme_train_df.index.tolist()
    test_idx = ritme_test_df.index.tolist()

    # read data
    otu_df = pd.read_csv(path_to_features, sep="\t", index_col=0)
    md_df = pd.read_csv(path_to_md, sep="\t", index_col=0)

    if use_shannon:
        # add shannon diversity to otu_df
        otu_df["shannon"] = shannon_diversity(otu_df)

    # split train-test
    X_train = otu_df.loc[train_idx]
    y_train = md_df.loc[train_idx, target]

    X_test = otu_df.loc[test_idx]
    y_test = md_df.loc[test_idx, target]

    # preprocess (filter, log, scale)
    X_train_scaled, feature_names, train_scaler = preprocess_data_for_model(X_train)
    # scaler is for log10 features:
    X_test = np.log10(X_test[feature_names] + 1e-4)
    X_test_scaled = train_scaler.transform(X_test)

    # hyper‐parameter grid + CV
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
    param_grid = {
        "n_estimators": [100, 500, 1000],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "gamma": [0.01],
        "colsample_bytree": [0.75],
        "subsample": [1],
        "min_child_weight": [1],
    }

    model = GridSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=0),
        param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X_train_scaled, y_train.values)

    # evaluate
    metrics, fig = get_metrics_n_scatterplot(
        model, X_train_scaled, y_train, X_test_scaled, y_test
    )

    # save metrics and scatter plot
    path_to_save_results = os.path.join(
        "n4_original_setup_results", f"{cohort}_{target}"
    )
    if not os.path.exists(path_to_save_results):
        os.makedirs(path_to_save_results)
    out_file = os.path.join(path_to_save_results, "metrics.csv")
    metrics.to_csv(out_file, index=False)
    print(f"Metrics written to {out_file}")
    print(f"Metrics: {metrics}")

    path_to_save = os.path.join(path_to_save_results, "best_true_vs_pred.png")
    fig.savefig(path_to_save, bbox_inches="tight")
    print(f"Scatter plots were saved in {path_to_save}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train & evaluate the original model by "
        "Nishijima et al. (2024) on the specified cohort."
    )
    parser.add_argument(
        "cohort",
        choices=["galaxy", "metacardis"],
        help="Which cohort to run: 'galaxy' or 'metacardis'",
    )
    parser.add_argument(
        "target",
        choices=["count", "count_log10"],
        help="Which column to predict: 'count_log10' or 'count'",
    )
    parser.add_argument(
        "use_shannon",
        type=bool,
        nargs="?",
        help="Whether to add Shannon diversity to the features (default: True)",
        default=True,
    )
    args = parser.parse_args()
    main(args.cohort, args.target, args.use_shannon)
