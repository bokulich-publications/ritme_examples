import matplotlib.pyplot as plt
import pandas as pd
from ritme.evaluate_tuned_models import _calculate_metrics, _plot_scatter_plots
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

plt.rcParams.update({"font.family": "DejaVu Sans"})
plt.style.use("seaborn-v0_8-pastel")


def load_baseline_split(splits_dir, ft_path, md_path, target):
    """Reproduce the train/test sample split used by ritme for a baseline run.

    Reads ritme's pickled splits in ``splits_dir`` to recover the sample ids,
    then slices the original (un-prefixed) feature table and metadata by
    those ids. Returns ``(X_train, y_train, X_test, y_test)``.

    Use the original publication's feature table here — not the one consumed
    by ritme — so the baseline reflects the original modelling pipeline.
    """
    train_idx = pd.read_pickle(f"{splits_dir}/train_val.pkl").index.tolist()
    test_idx = pd.read_pickle(f"{splits_dir}/test.pkl").index.tolist()
    ft = pd.read_csv(ft_path, sep="\t", index_col=0)
    md = pd.read_csv(md_path, sep="\t", index_col=0)
    predictor_cols = ft.columns
    data = md.join(ft, how="inner")
    return (
        data.loc[train_idx, predictor_cols],
        data.loc[train_idx, target],
        data.loc[test_idx, predictor_cols],
        data.loc[test_idx, target],
    )


def get_metrics_n_scatterplot(model, X_train, y_train, X_test, y_test):
    model_type = "original"
    dic_data = {"train": [X_train, y_train], "test": [X_test, y_test]}

    # get predictions
    preds_df = pd.DataFrame()
    for split, data in dic_data.items():
        split_pred = pd.DataFrame()
        split_pred["true"] = data[1]
        split_pred["pred"] = model.predict(data[0])
        split_pred["split"] = split
        preds_df = pd.concat([preds_df, split_pred], ignore_index=True)

    # get metrics
    metrics = _calculate_metrics(preds_df, model_type)

    # get scatterplot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=400)
    _plot_scatter_plots(
        all_preds=preds_df,
        metrics_split=metrics,
        axs=axs,
        row_idx=0,
        model_name=model_type,
        only_one_model=True,
    )

    return metrics, fig


def get_metrics_n_roc_curve(model, X_train, y_train, X_test, y_test):
    """Classification analog of :func:`get_metrics_n_scatterplot`.

    Computes AUROC, accuracy, F1, precision, recall on train and test sets,
    and draws side-by-side train/test ROC curves. Returns
    ``(metrics_df, fig)``.

    Requires ``model`` to expose ``predict_proba`` (a scikit-learn-style
    binary classifier with the positive class encoded as ``1``).
    """
    model_type = "original"
    dic_data = {"train": (X_train, y_train), "test": (X_test, y_test)}

    metrics = pd.DataFrame()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=400)

    for ax, (split, (X, y)) in zip(axs, dic_data.items()):
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics.loc[model_type, f"auroc_{split}"] = roc_auc_score(y, y_proba)
        metrics.loc[model_type, f"accuracy_{split}"] = accuracy_score(y, y_pred)
        metrics.loc[model_type, f"f1_{split}"] = f1_score(y, y_pred, zero_division=0)
        metrics.loc[model_type, f"precision_{split}"] = precision_score(
            y, y_pred, zero_division=0
        )
        metrics.loc[model_type, f"recall_{split}"] = recall_score(
            y, y_pred, zero_division=0
        )

        RocCurveDisplay.from_predictions(y, y_proba, ax=ax, name=model_type)
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
        ax.set_title(f"{split} (n={len(y)})")
        ax.set_aspect("equal")

    fig.tight_layout()
    return metrics, fig
