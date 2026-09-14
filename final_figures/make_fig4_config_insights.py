"""Figure 4: configuration insights across the use cases.

Merges the per-use-case "boxplot_all_trials" views onto one A4 page. Each
column is a use case with its own metric axis; each row-block is a ritme
configuration choice, sharing one y-axis across the figure so the category
labels are written once (column 1 only).

Row order is fixed by hand (see ORDERS), grouping options by what they do:
the paired abundance/variance selectors, the compositional transforms
together.

Usage (repo root, ritme_usecases env):
    python -m final_figures.make_fig4_config_insights   # use cases 1-3
    python -m final_figures.make_w_u4                   # all four, as `_w_u4`
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "use_cases" / "ritme_runs" / "local"
OUT_DIR = REPO_ROOT / "final_figures"
STEM = "fig4_config_insights"

# Runs of record per use case. u3 uses the `_no_fit` experiments (no
# `fit_result` covariate); the `_no_enrich` / `_reduced` / `_dynamic`
# variants are excluded everywhere.
RUN_PATTERNS = {
    "u1": re.compile(r"^u1_[a-z_]+_tpe$"),
    "u2": re.compile(r"^u2_[a-z_]+_tpe$"),
    "u3": re.compile(r"^u3_[a-z_]+_tpe_no_fit$"),
    "u4": re.compile(r"^u4_[a-z_]+_tpe$"),
}
USECASES = ("u1", "u2", "u3", "u4")
TAG_COL = "tags.experiment_tag"
MODEL_TYPE_COL = "model_type"  # the raw `params.model`, kept beside its row label
# Figure variants: which use cases a render draws, keyed by the suffix its
# files carry. The unsuffixed figures cover use cases 1-3; `_w_u4` adds use
# case 4 (see make_w_u4.py). Every figure here renders either set through one
# code path -- main() takes `use_cases` and `suffix`, and defaults to the set
# registered for the suffix. Registering the pairs keeps one set from being
# written over the other's file names, and keeps the sets to layouts the
# figures are known to handle.
VARIANTS = {
    "": tuple(uc for uc in USECASES if uc != "u4"),
    "_w_u4": USECASES,
}


def check_variant(use_cases, suffix: str) -> None:
    """Reject any use-case/suffix pair that is not a registered variant."""
    if VARIANTS.get(suffix) != tuple(use_cases):
        raise SystemExit(
            f"{suffix!r} is not the registered suffix for {list(use_cases)}; "
            "see VARIANTS in make_fig4_config_insights"
        )


def variant_use_cases(use_cases, suffix: str) -> list[str]:
    """The use cases a render draws: `use_cases`, or those registered for `suffix`."""
    use_cases = list(VARIANTS.get(suffix, ()) if use_cases is None else use_cases)
    check_variant(use_cases, suffix)
    return use_cases


TITLES = {
    "u1": "Use case 1",
    "u2": "Use case 2",
    "u3": "Use case 3",
    "u4": "Use case 4",
}
RMSE_LABEL = "RMSE Validation\n(↓, log scale)"
AUC_LABEL = "ROC AUC\nValidation (↑)"
METRIC = {
    "u1": ("metrics.rmse_val", RMSE_LABEL, True),
    "u2": ("metrics.rmse_val", RMSE_LABEL, True),
    "u3": ("metrics.roc_auc_macro_ovr_val", AUC_LABEL, False),
    "u4": ("metrics.roc_auc_macro_ovr_val", AUC_LABEL, False),
}  # column -> (metric, axis label, log scale)

GROUPS = [
    ("params.data_aggregation", "Data aggregation"),
    ("params.data_selection", "Data selection"),
    ("params.data_transform", "Data transform"),
    ("params.data_enrich", "Data enrichment"),
    ("params.model", "Model type"),
]

# The option being switched off is logged as `nan`; it reads as `None` and
# leads every group.
NONE = "None"

# Row order per group, top to bottom.
ORDERS = {
    "params.data_aggregation": [
        NONE,
        "tax_class",
        "tax_order",
        "tax_family",
        "tax_genus",
    ],
    "params.data_selection": [
        NONE,
        "abundance_topi",
        "variance_topi",
        "abundance_ith",
        "variance_ith",
        "abundance_quantile",
        "variance_quantile",
        "abundance_threshold",
        "variance_threshold",
    ],
    "params.data_transform": [NONE, "clr", "ilr", "alr", "pa", "rank"],
    "params.data_enrich": [NONE, "shannon", "shannon_and_metadata", "metadata_only"],
    "params.model": [
        "xgb",
        "rf",
        "linreg | logreg",
        "nn_reg | nn_class",
        "nn_corn",
        "trac",
    ],
}

# Raw parameter value -> row, where the two differ. The regression and
# classification variants of a model family share a row, so a row means the
# same thing in every column; the label names both.
VALUE_TO_ROW = {
    "params.model": {
        "xgb": "xgb",
        "xgb_class": "xgb",
        "rf": "rf",
        "rf_class": "rf",
        "linreg": "linreg | logreg",
        "logreg": "linreg | logreg",
        "nn_reg": "nn_reg | nn_class",
        "nn_class": "nn_reg | nn_class",
        "nn_corn": "nn_corn",
        "trac": "trac",
    }
}

# Panels drawn on a cut x-axis. U1's `alr` transform spans ~3 to ~1e5 RMSE
# while every other transform tops out near 6, so on one continuous axis it
# flattens the whole use case. Cutting this panel also frees the rest of the
# U1 column, which then autoscales to the range the other options occupy.
BROKEN_PANELS = {("params.data_transform", "u1")}
BREAK_GAP = 1600.0  # factor skipped between the two segments; keeps 1e4 in view
BREAK_WIDTH_RATIOS = (2.25, 1.0)  # tail wide enough to separate its decades
TAIL_HI_MARGIN = 1.15  # headroom past the tail whisker
TAIL_LABEL_SIZE = 8.5  # tick labels on both segments of the cut axis

# Axes are otherwise never capped.
BOX_EDGE = "black"
MEDIAN = "black"
PALETTE_NAME = "Set3"
Y_LABEL_PAD = 138  # anchor for the left-aligned category names, in points
GROUP_LABEL_SIZE = 12.5  # matches the use-case column headers


def to_rows(values: pd.Series, col: str) -> pd.Series:
    """Map raw parameter values onto the figure's row labels."""
    mapping = VALUE_TO_ROW.get(col, {})
    rows = values.astype(str).map(
        lambda v: mapping.get(v, NONE if v == "nan" else v)  # noqa: B023
    )
    unknown = set(rows.unique()) - set(ORDERS[col])
    if unknown:
        raise SystemExit(f"{col}: no row defined for {sorted(unknown)}")
    return rows


def load_trials(use_cases=None) -> dict[str, pd.DataFrame]:
    """Concatenated trial logs per use case, values mapped onto rows.

    Only `use_cases` is read, so a variant neither needs nor touches the
    logs of the use cases it does not draw.
    """
    out = {}
    for uc in USECASES if use_cases is None else use_cases:
        pattern = RUN_PATTERNS[uc]
        frames = []
        for run in sorted(RUNS_DIR.glob(f"{uc}_*")):
            if not pattern.fullmatch(run.name):
                continue
            log = run / "mlflow_logs.csv"
            if log.exists():
                frames.append(pd.read_csv(log, low_memory=False))
        if not frames:
            raise SystemExit(f"no trial logs for {uc} under {RUNS_DIR}")
        # copy() consolidates the concatenated blocks, so adding a column is cheap
        df = pd.concat(frames, ignore_index=True).copy()
        df[MODEL_TYPE_COL] = df["params.model"]  # ritme's selector keys on it
        for col, _ in GROUPS:
            df[col] = to_rows(df[col], col)
        out[uc] = df
    return out


def selected_trial(df: pd.DataFrame, metric: str, mode: str) -> pd.Series:
    """The trial reported for a use case, chosen as `evaluate_all_trials.ipynb` does.

    Within each experiment ritme's one-standard-error rule names the deployed
    trial; across experiments, the deployed trial with the best mean
    validation score is the one reported. `mode` is "min" or "max".
    """
    from types import SimpleNamespace

    from ritme.evaluate_models import _select_best_with_one_se

    mean_col, se_col = f"{metric}_mean", f"{metric}_se"
    ascending = mode == "min"

    def to_result(idx, row):
        return SimpleNamespace(
            config={
                k.removeprefix("params."): v
                for k, v in row.items()
                if k.startswith("params.") and not pd.isna(v)
            },
            metrics={
                k.removeprefix("metrics."): v
                for k, v in row.items()
                if k.startswith("metrics.") and not pd.isna(v)
            },
            path=None,
            checkpoint=True,  # MLflow logs carry no Ray checkpoint state
            error=None,
            _df_idx=idx,
        )

    picks = []
    for _, group in df.groupby(TAG_COL):
        if group[[mean_col, se_col]].dropna().empty:
            picks.append(group.sort_values(metric, ascending=ascending).iloc[0])
            continue
        best = _select_best_with_one_se(
            [to_result(i, r) for i, r in group.iterrows()],
            metric=metric.removeprefix("metrics."),
            mode=mode,
            model_type=group[MODEL_TYPE_COL].iloc[0],
        )
        picks.append(group.loc[best._df_idx])
    picks = pd.DataFrame(picks)
    return picks.loc[
        picks[mean_col].idxmin() if ascending else picks[mean_col].idxmax()
    ]


# Figures 5 and 6 ring the reported trial wherever they draw it: one marker
# and one legend entry for both.
SELECTED_RING = {
    "s": 80,
    "facecolors": "none",
    "edgecolors": "black",
    "linewidths": 1.3,
    "zorder": 6,
}
SELECTED_LABEL = "configuration selected by ritme (one-standard-error rule)"


def ring_selected(ax: plt.Axes, x: float, y: float, **kwargs):
    """Ring the point at (x, y); drawn whole even where it crosses the frame."""
    return ax.scatter([x], [y], clip_on=False, **SELECTED_RING, **kwargs)


def selected_handle() -> plt.Line2D:
    return plt.Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        markersize=SELECTED_RING["s"] ** 0.5,
        markerfacecolor="none",
        markeredgecolor=SELECTED_RING["edgecolors"],
        markeredgewidth=SELECTED_RING["linewidths"],
    )


def whisker_ends(values: np.ndarray) -> tuple[float, float]:
    """Matplotlib's boxplot whisker ends: the data within 1.5 IQR."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return (
        float(values[values >= q1 - 1.5 * iqr].min()),
        float(values[values <= q3 + 1.5 * iqr].max()),
    )


def break_ranges(data: list[np.ndarray]) -> tuple[tuple, tuple]:
    """Main segment and far-tail segment for a cut axis.

    Tail categories are those whose whisker reaches far past the others; the
    two segments cover the bulk and the tail, separated by BREAK_GAP.
    """
    ends = [whisker_ends(d) for d in data if len(d)]
    his = sorted(e[1] for e in ends)
    bulk = [h for h in his if h <= his[len(his) // 2] * 20]
    main_hi = max(bulk) if bulk else his[-1]
    lo = min(e[0] for e in ends)
    return (lo * 0.9, main_hi * 1.35), (
        main_hi * BREAK_GAP,
        max(his) * TAIL_HI_MARGIN,
    )


def draw_break_marks(ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    """Slanted ticks marking the cut, on both facing spines."""
    kw = dict(
        marker=[(-1, -0.6), (1, 0.6)],
        markersize=6,
        linestyle="none",
        color="black",
        mec="black",
        mew=1,
        clip_on=False,
    )
    ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **kw)
    ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **kw)


def draw_boxes(
    ax: plt.Axes,
    data: list,
    drawn: list[int],
    positions,
    order: list[str],
    colors: dict,
) -> None:
    bp = ax.boxplot(
        [data[i] for i in drawn],
        positions=positions[drawn],
        vert=False,
        showfliers=False,
        widths=0.62,
        patch_artist=True,
        boxprops={"edgecolor": BOX_EDGE, "linewidth": 0.8},
        medianprops={"color": MEDIAN, "linewidth": 1.5},
        whiskerprops={"color": BOX_EDGE, "linewidth": 0.8},
        capprops={"color": BOX_EDGE, "linewidth": 0.8},
    )
    for patch, i in zip(bp["boxes"], drawn):
        patch.set_facecolor(colors[order[i]])


def main(use_cases: list[str] | None = None, suffix: str = "") -> None:
    """Render the figure; `use_cases` defaults to the variant `suffix` names."""
    use_cases = variant_use_cases(use_cases, suffix)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 10.5,
            "axes.titlesize": GROUP_LABEL_SIZE,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.grid": True,
            "grid.alpha": 1.0,
            "grid.color": "#E3E3E3",
            "grid.linewidth": 0.5,
            # EPS has no alpha channel: keep every element opaque.
            "ps.fonttype": 42,
            "pdf.fonttype": 42,
        }
    )

    trials = load_trials(use_cases)
    orders = {col: ORDERS[col] for col, _ in GROUPS}

    cmap = plt.get_cmap(PALETTE_NAME)
    colors: dict[str, tuple] = {}
    for order in orders.values():
        for i, cat in enumerate(order):
            colors.setdefault(cat, cmap(i % cmap.N))

    fig = plt.figure(figsize=(8.0, 11.2))
    gs = GridSpec(
        len(GROUPS),
        len(use_cases),
        figure=fig,
        height_ratios=[len(orders[col]) for col, _ in GROUPS],
        hspace=0.16,
        wspace=0.10,
        left=0.300,
        right=0.985,
        top=0.960,
        bottom=0.060,
    )

    col_axes: dict[int, plt.Axes] = {}
    for r, (col, group_label) in enumerate(GROUPS):
        order = orders[col]
        for c, uc in enumerate(use_cases):
            df = trials[uc]
            metric, xlabel, log_scale = METRIC[uc]
            data = [df.loc[df[col] == cat, metric].dropna().values for cat in order]
            positions = np.arange(len(order))
            drawn = [i for i, d in enumerate(data) if len(d)]
            broken = bool(drawn) and (col, uc) in BROKEN_PANELS

            if broken:
                # Two segments in this cell. The panel keeps its own scale, so
                # the rest of the column is free to autoscale to the bulk.
                inner = gs[r, c].subgridspec(
                    1, 2, width_ratios=BREAK_WIDTH_RATIOS, wspace=0.09
                )
                ax = fig.add_subplot(inner[0])
                ax_tail = fig.add_subplot(inner[1], sharey=ax)
                for sub, xlim in zip((ax, ax_tail), break_ranges(data)):
                    draw_boxes(sub, data, drawn, positions, order, colors)
                    sub.set_xscale("log")
                    sub.set_xlim(*xlim)
                    sub.set_ylim(len(order) - 0.5, -0.5)
                    sub.set_yticks(positions)
                    sub.grid(axis="x")
                    sub.set_axisbelow(True)
                    sub.xaxis.set_major_locator(mticker.LogLocator(numticks=3))
                    sub.xaxis.set_minor_locator(mticker.NullLocator())
                    sub.tick_params(axis="x", labelsize=TAIL_LABEL_SIZE)
                ax.spines["right"].set_visible(False)
                ax_tail.spines["left"].set_visible(False)
                ax_tail.tick_params(axis="y", left=False, labelleft=False)
                ax_tail.set_yticklabels([])
                draw_break_marks(ax, ax_tail)
            else:
                # One metric axis per column, shared by its unbroken blocks.
                ax = fig.add_subplot(gs[r, c], sharex=col_axes.get(c))
                col_axes.setdefault(c, ax)
                if drawn:
                    draw_boxes(ax, data, drawn, positions, order, colors)
                ax.set_ylim(len(order) - 0.5, -0.5)
                ax.set_yticks(positions)
                if log_scale:
                    ax.set_xscale("log")
                ax.grid(axis="x")
                ax.set_axisbelow(True)

            if c == 0:
                ax.set_yticklabels(order)
                # Left-align: anchor clear of the axis, text runs towards it.
                for label in ax.get_yticklabels():
                    label.set_horizontalalignment("left")
                ax.tick_params(axis="y", pad=Y_LABEL_PAD)
                ax.set_ylabel(group_label, labelpad=12, fontsize=GROUP_LABEL_SIZE)
            elif not broken:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(TITLES[uc], pad=5)
            if r == len(GROUPS) - 1:
                ax.set_xlabel(xlabel)
                if log_scale:
                    ax.xaxis.set_major_locator(mticker.LogLocator(numticks=4))
                    ax.xaxis.set_minor_locator(mticker.NullLocator())
                else:
                    ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
            elif not broken:
                ax.tick_params(axis="x", labelbottom=False)
            if not broken:
                ax.tick_params(axis="x", labelsize=10)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("eps", "pdf", "png"):
        fig.savefig(OUT_DIR / f"{STEM}{suffix}.{ext}", format=ext, dpi=600)
        print(f"Wrote {OUT_DIR / f'{STEM}{suffix}.{ext}'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
