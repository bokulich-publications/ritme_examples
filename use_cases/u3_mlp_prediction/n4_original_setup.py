import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from xgboost import XGBRegressor

from src.process_u3 import preprocess_data_for_model


def shannon_diversity(df, eps=1e-9):
    # 1) compute proportions per sample
    p = df.div(df.sum(axis=1), axis=0)
    # 2) Shannon index
    return -(p * np.log(p + eps)).sum(axis=1)


def main(cohort: str, use_shannon: bool = False):
    # paths
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
    y_train = md_df.loc[train_idx, "count_log10"]

    X_test = otu_df.loc[test_idx]
    y_test = md_df.loc[test_idx, "count_log10"]

    # preprocess (filter, log, scale)
    X_train_scaled, feature_names, train_scaler = preprocess_data_for_model(X_train)
    X_test_scaled = train_scaler.transform(X_test[feature_names])

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
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)
    train_r2 = r2_score(y_train, train_preds)
    train_rmse = root_mean_squared_error(y_train, train_preds)
    train_pearson = pearsonr(y_train, train_preds)[0]

    test_r2 = r2_score(y_test, test_preds)
    test_rmse = root_mean_squared_error(y_test, test_preds)
    test_pearson = pearsonr(y_test, test_preds)[0]

    results = pd.DataFrame(
        {
            "R2 test": [test_r2],
            "RMSE test": [test_rmse],
            "Pearson test": [test_pearson],
            "R2 train": [train_r2],
            "RMSE train": [train_rmse],
            "Pearson train": [train_pearson],
        }
    ).round(3)

    out_file = f"n4_original_results_{cohort}.csv"
    results.to_csv(out_file, index=False)
    print(f"Results written to {out_file}")
    print(results)


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
        "use_shannon",
        type=bool,
        help="Whether to add Shannon diversity to the features",
    )
    args = parser.parse_args()
    main(args.cohort)
