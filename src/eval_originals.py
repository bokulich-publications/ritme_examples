import matplotlib.pyplot as plt
import pandas as pd
from ritme.evaluate_tuned_models import _calculate_metrics, _plot_scatter_plots

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
