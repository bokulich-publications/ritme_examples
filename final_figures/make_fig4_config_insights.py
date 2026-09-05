"""Figure 4: configuration insights across all four use cases.

Merges the per-use-case "boxplot_all_trials" views onto one A4 page. Each
column is a use case with its own metric axis; each row-block is a ritme
configuration choice, sharing one y-axis across the figure so the category
labels are written once (column 1 only).

Row order is fixed by hand (see ORDERS): options are grouped by what they do
-- the paired abundance/variance selectors, the compositional transforms
together -- rather than ranked by score, which cannot be pooled across RMSE
and ROC AUC anyway.

Regenerated from the trial logs rather than composed from the per-use-case
PDFs: a shared order and uniform styling cannot be recovered from the PDFs.

Usage (repo root, ritme_usecases env):
    python -m final_figures.make_fig4_config_insights
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
USECASES = ["u1", "u2", "u3", "u4"]
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


def load_trials() -> dict[str, pd.DataFrame]:
    """Concatenated trial logs per use case, values mapped onto rows."""
    out = {}
    for uc, pattern in RUN_PATTERNS.items():
        frames = []
        for run in sorted(RUNS_DIR.glob(f"{uc}_*")):
            if not pattern.fullmatch(run.name):
                continue
            log = run / "mlflow_logs.csv"
            if log.exists():
                frames.append(pd.read_csv(log, low_memory=False))
        if not frames:
            raise SystemExit(f"no trial logs for {uc} under {RUNS_DIR}")
        df = pd.concat(frames, ignore_index=True)
        for col, _ in GROUPS:
            df[col] = to_rows(df[col], col)
        out[uc] = df
    return out


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


def main() -> None:
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

    trials = load_trials()
    orders = {col: ORDERS[col] for col, _ in GROUPS}

    cmap = plt.get_cmap(PALETTE_NAME)
    colors: dict[str, tuple] = {}
    for order in orders.values():
        for i, cat in enumerate(order):
            colors.setdefault(cat, cmap(i % cmap.N))

    fig = plt.figure(figsize=(8.0, 11.2))
    gs = GridSpec(
        len(GROUPS),
        len(USECASES),
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
        for c, uc in enumerate(USECASES):
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
        fig.savefig(OUT_DIR / f"{STEM}.{ext}", format=ext, dpi=600)
        print(f"Wrote {OUT_DIR / f'{STEM}.{ext}'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
