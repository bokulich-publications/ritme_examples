# script needed since sklearn is way older in autosklearn conda env than ritme
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.transforms import offset_copy
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams.update({"font.family": "DejaVu Sans"})
plt.style.use("seaborn-v0_8-pastel")


def _plot_scatter_plots(
    all_preds: pd.DataFrame,
    metrics_split: pd.DataFrame,
    axs: list,
    row_idx: int,
    model_name: str,
    only_one_model: bool = False,
):
    i = 0
    colors = {"train": "cornflowerblue", "test": "coral"}

    for split in ["train", "test"]:
        if only_one_model:
            axs_set = axs[i]
        else:
            axs_set = axs[row_idx, i]

        pred_split = all_preds[all_preds["split"] == split].copy()

        # scatter plot with linear regression line
        if pred_split.shape[0] > 100:
            dot_size = 20
            dot_alpha = 0.3
        else:
            dot_size = 50
            dot_alpha = 0.8
        reg = sns.regplot(
            x=pred_split["true"].astype(float),
            y=pred_split["pred"],
            ax=axs_set,
            color=colors[split],
            scatter_kws={"s": dot_size, "alpha": dot_alpha},
            line_kws={"color": "dimgrey"},
        )
        # add model name as a higher‐level y‐label on the first column
        if i == 0:
            axs_set.set_ylabel(rf"$\mathbf{{{model_name}}}$" + "\n\nPredicted")
        else:
            axs_set.set_ylabel("Predicted")

        axs_set.set_xlabel("True")
        # 1:1 ratio between true and predicted values
        x0, x1 = reg.axes.get_xlim()
        y0, y1 = reg.axes.get_ylim()
        lims = [min(x0, y0), max(x1, y1)]
        reg.axes.plot(lims, lims, ":k")

        # add rmse, r2 and pearson corr metrics to plot
        rmse = metrics_split[f"rmse_{split}"].values[0]
        r2 = metrics_split[f"r2_{split}"].values[0]
        r = metrics_split[f"pearson_corr_{split}"].values[0]

        trans = offset_copy(axs_set.transData, x=1, y=-1, units="dots")
        axs_set.text(
            lims[0],
            lims[1],
            f"RMSE: {rmse:.2f}\nR²: {r2:.2f}\nR: {r:.2f}",
            transform=trans,
            color=colors[split],
            ha="left",
            va="top",
        )
        if row_idx == 0:
            axs_set.set_title(f"{split.capitalize()} set")
        i += 1


def _calculate_metrics(all_preds: pd.DataFrame, model_type: str) -> pd.DataFrame:
    metrics = pd.DataFrame()
    for split in ["train", "test"]:
        pred_split = all_preds[all_preds["split"] == split].copy()

        metrics.loc[model_type, f"rmse_{split}"] = mean_squared_error(
            pred_split["true"], pred_split["pred"], squared=False
        )
        metrics.loc[model_type, f"r2_{split}"] = r2_score(
            pred_split["true"], pred_split["pred"]
        )
        pearson_results = pearsonr(pred_split["true"].astype(float), pred_split["pred"])
        metrics.loc[model_type, f"pearson_corr_{split}"] = pearson_results[0]
        # two-sided alternative
        metrics.loc[model_type, f"pearson_corr_{split}_pvalue"] = pearson_results[1]
    return metrics


def get_metrics_n_scatterplot(model, X_train, y_train, X_test, y_test):
    model_type = "automl"
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
