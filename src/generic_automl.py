"""automl implementation for usecase regression tasks"""

import argparse
import os
from pprint import pprint

import autosklearn.regression
import pandas as pd
from autosklearn.ensembles import SingleBest
from autosklearn.metrics import root_mean_squared_error

from src.eval_automl import get_metrics_n_scatterplot


def parse_args():
    p = argparse.ArgumentParser(description="Run Auto-Sklearn regression")
    p.add_argument("--total-time-s", type=int, required=True)
    p.add_argument("--usecase", required=True)
    p.add_argument("--data-splits-folder", required=True)
    p.add_argument("--path-to-features", required=True)
    p.add_argument("--path-to-md", required=True)
    p.add_argument("--target", required=True, help="Target column in metadata")
    p.add_argument(
        "--restricted-model",
        dest="restricted_models",
        nargs="+",
        default=[],
        choices=[
            "ard_regression",
            "gradient_boosting",
            "mlp",
            "random_forest",
        ],
        help="Space separated list of regressors to include.",
    )
    p.add_argument(
        "--single-best",
        action="store_true",
        help=("If set, restrict Auto-Sklearn to single best model - no ensembles."),
    )
    return p.parse_args()


def main():
    args = parse_args()

    # load indices
    train_df = pd.read_pickle(f"{args.data_splits_folder}/train_val.pkl")
    test_df = pd.read_pickle(f"{args.data_splits_folder}/test.pkl")
    train_idx = train_df.index.tolist()
    test_idx = test_df.index.tolist()

    # load data
    otu_df = pd.read_csv(args.path_to_features, sep="\t", index_col=0)
    md_df = pd.read_csv(args.path_to_md, sep="\t", index_col=0)
    # Convert absolute abundances to relative abundances
    otu_df = otu_df.div(otu_df.sum(axis=1), axis=0)
    print("md_df.shape", md_df.shape)
    print("otu_df.shape", otu_df.shape)

    # subset
    X_train = otu_df.loc[train_idx]
    y_train = md_df.loc[train_idx, args.target]
    X_test = otu_df.loc[test_idx]
    y_test = md_df.loc[test_idx, args.target]

    # X, y = sklearn.datasets.load_diabetes(return_X_y=True)
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    #     X, y, random_state=1
    # )
    if args.single_best:
        print("No ensembles - only single best model.")
        common_kwargs = dict(
            time_left_for_this_task=args.total_time_s,
            n_jobs=-1,
            metric=root_mean_squared_error,
            ensemble_class=SingleBest,
            memory_limit=24000,
        )
    else:
        common_kwargs = dict(
            time_left_for_this_task=args.total_time_s,
            n_jobs=-1,
            metric=root_mean_squared_error,
            memory_limit=24000,
        )

    if len(args.restricted_models) > 0:
        print(f"Using only restricted models: {args.restricted_models}")
        automl = autosklearn.regression.AutoSklearnRegressor(
            include={"regressor": args.restricted_models},
            **common_kwargs,
        )
    else:
        automl = autosklearn.regression.AutoSklearnRegressor(
            include={
                "regressor": [
                    "ard_regression",
                    "gradient_boosting",
                    "mlp",
                    "random_forest",
                ]
            },
            **common_kwargs,
        )

    automl.fit(X_train, y_train)
    print("Print model leaderboard:")
    print(automl.leaderboard())

    print("Final ensemble:")
    pprint(automl.show_models(), indent=4)

    # evaluate
    metrics, fig = get_metrics_n_scatterplot(automl, X_train, y_train, X_test, y_test)

    out_dir = "automl"
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, f"{args.usecase}_metrics.csv")
    metrics.reset_index(names="model", inplace=True)
    metrics.to_csv(metrics_path, index=False)

    fig_path = os.path.join(out_dir, f"{args.usecase}_best_true_vs_pred.png")
    fig.savefig(fig_path, bbox_inches="tight")

    print(f"Metrics written to {metrics_path}")
    print(f"Plot written to {fig_path}")


if __name__ == "__main__":
    main()
