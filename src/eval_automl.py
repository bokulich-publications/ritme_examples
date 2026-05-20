# script needed since sklearn is way older in autosklearn conda env than ritme
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.transforms import offset_copy
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

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

        metrics.loc[model_type, f"rmse_{split}"] = np.sqrt(
            mean_squared_error(pred_split["true"], pred_split["pred"])
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


def _bootstrap_classification_cis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 12,
) -> dict[str, tuple[float, float]]:
    """Percentile-bootstrap CIs for the binary-classification metric set.

    Mirrors `src.bootstrap_metrics.bootstrap_classification_metrics` (same
    defaults, same seed, same metrics, same threshold of 0.5 already baked
    into ``y_pred``). Re-implemented inline here so the autosklearn env --
    whose sklearn predates ``root_mean_squared_error`` -- can import this
    module; the upstream `bootstrap_metrics` module top-level imports that
    symbol.

    Returns ``{metric: (ci_low, ci_high)}`` for ``auroc``, ``accuracy``,
    ``f1``, ``precision``, ``recall``. AUROC resamples that draw a single
    class are skipped (AUROC undefined); if every resample is degenerate
    the AUROC CI is NaN.
    """
    n = y_true.shape[0]
    rng = np.random.default_rng(seed)
    auroc_b: list = []
    accuracy_b: list = []
    f1_b: list = []
    precision_b: list = []
    recall_b: list = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, yp_proba, yp = y_true[idx], y_proba[idx], y_pred[idx]
        if np.unique(yt).size > 1:
            auroc_b.append(roc_auc_score(yt, yp_proba))
        accuracy_b.append(accuracy_score(yt, yp))
        f1_b.append(f1_score(yt, yp, zero_division=0))
        precision_b.append(precision_score(yt, yp, zero_division=0))
        recall_b.append(recall_score(yt, yp, zero_division=0))

    alpha = (1.0 - ci) / 2.0
    out: dict[str, tuple[float, float]] = {}
    for name, samples in [
        ("auroc", auroc_b),
        ("accuracy", accuracy_b),
        ("f1", f1_b),
        ("precision", precision_b),
        ("recall", recall_b),
    ]:
        arr = np.asarray(samples, dtype=float)
        if arr.size == 0:
            out[name] = (float("nan"), float("nan"))
        else:
            out[name] = (
                float(np.quantile(arr, alpha)),
                float(np.quantile(arr, 1 - alpha)),
            )
    return out


def get_metrics_n_roc_curve(model, X_train, y_train, X_test, y_test):
    """Auto-sklearn classification analog of :func:`get_metrics_n_scatterplot`.

    Mirrors `src.eval_originals.get_metrics_n_roc_curve` so the autoML and
    original-baseline outputs share the same metric layout, but lives here
    to keep `src/eval_automl.py` importable in the autosklearn env (whose
    sklearn is far older than ritme's).

    Also emits the ritme-aligned classification metric set
    (``roc_auc_macro_ovr``, ``log_loss``, ``f1_macro``, ``balanced_accuracy``,
    ``mcc`` per split) and the test-set percentile bootstrap CIs that
    `merge_best_metrics.py` pivots out of `bootstrap_test_metrics.csv` for
    ritme runs, so an autoML row sits in the same column space as ritme +
    original-baseline rows when concatenated in `evaluate_all_trials.ipynb`.
    Assumes binary classification (positive class encoded as 1); U3 is the
    only classification usecase here.
    """
    model_type = "automl"
    dic_data = {"train": (X_train, y_train), "test": (X_test, y_test)}
    classes = list(model.classes_)

    metrics = pd.DataFrame()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=400)
    colors = {"train": "cornflowerblue", "test": "coral"}

    test_eval: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for ax, (split, (X, y)) in zip(axs, dic_data.items()):
        y_proba_full = model.predict_proba(X)
        y_proba = y_proba_full[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        y_arr = np.asarray(y)

        metrics.loc[model_type, f"auroc_{split}"] = roc_auc_score(y_arr, y_proba)
        metrics.loc[model_type, f"accuracy_{split}"] = accuracy_score(y_arr, y_pred)
        metrics.loc[model_type, f"f1_{split}"] = f1_score(
            y_arr, y_pred, zero_division=0
        )
        metrics.loc[model_type, f"precision_{split}"] = precision_score(
            y_arr, y_pred, zero_division=0
        )
        metrics.loc[model_type, f"recall_{split}"] = recall_score(
            y_arr, y_pred, zero_division=0
        )

        # ritme-aligned set, matches ritme._calculate_classification_metrics
        # for the binary branch (len(classes) == 2): macro_ovr AUC reduces
        # to `roc_auc_score(y, y_proba[:, 1])`.
        metrics.loc[model_type, f"roc_auc_macro_ovr_{split}"] = roc_auc_score(
            y_arr, y_proba
        )
        metrics.loc[model_type, f"log_loss_{split}"] = log_loss(
            y_arr, y_proba_full, labels=classes
        )
        metrics.loc[model_type, f"f1_macro_{split}"] = f1_score(
            y_arr, y_pred, average="macro"
        )
        metrics.loc[model_type, f"balanced_accuracy_{split}"] = balanced_accuracy_score(
            y_arr, y_pred
        )
        metrics.loc[model_type, f"mcc_{split}"] = matthews_corrcoef(y_arr, y_pred)

        if split == "test":
            test_eval = (y_arr, y_proba, y_pred)

        fpr, tpr, _ = roc_curve(y_arr, y_proba)
        auroc = metrics.loc[model_type, f"auroc_{split}"]
        ax.plot(
            fpr,
            tpr,
            color=colors[split],
            label=f"{model_type} (AUROC = {auroc:.2f})",
        )
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{split.capitalize()} (n={len(y_arr)})")
        ax.set_aspect("equal")
        ax.legend(loc="lower right")

    assert test_eval is not None
    y_true_t, y_proba_t, y_pred_t = test_eval
    cis = _bootstrap_classification_cis(y_true_t, y_proba_t, y_pred_t)
    for name, (lo, hi) in cis.items():
        metrics.loc[model_type, f"{name}_test_ci_low"] = lo
        metrics.loc[model_type, f"{name}_test_ci_high"] = hi

    fig.tight_layout()
    return metrics, fig
