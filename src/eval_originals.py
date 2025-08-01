import matplotlib.pyplot as plt
import pandas as pd
from ritme.evaluate_tuned_models import _calculate_metrics, _plot_scatter_plots

plt.rcParams.update({"font.family": "DejaVu Sans"})
plt.style.use("seaborn-v0_8-pastel")


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
