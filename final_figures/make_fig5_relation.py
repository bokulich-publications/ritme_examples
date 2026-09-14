"""Figure 5: feature-set size and its relation to predictive performance.

Two panels over the same use cases -- 1-3, or all four as the `_w_u4`
variant -- on a portrait page that fits A4:

* **a** -- how many features every model family actually keeps, across all
  trials of the run of record. One row per family, shared with figure 4, so
  the labels are written once (column 1 only) and one feature-count axis is
  shared by the columns: the use cases are then directly comparable.
* **b** -- feature-set size against validation performance for the 1000 best
  trials, so the panel shows where in the complexity range good models live.
  Each use case autoscales, since its own metric range is what is read here.
  The trial reported for the use case -- ritme's one-standard-error pick of
  the winning search, as `use_cases/evaluate_all_trials.ipynb` selects it --
  is ringed in black.

Axes follow figure 4: log-scaled feature counts, the metric named with the
arrow that gives its improving direction, and its Set3 slot per model family.
One size for every axis label and one for every tick label, and each label is
written once per shared axis -- once on the left, once along the bottom.
Run selection, model rows and the palette are imported from
`make_fig4_config_insights`, so figures 4 and 5 cannot drift apart.

Usage (ritme_usecases env), from the repo root or from this directory:
    python -m final_figures.make_fig5_relation   # use cases 1-3
    python make_fig5_relation.py
    python -m final_figures.make_w_u4            # all four, as `_w_u4`
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.gridspec import GridSpec

try:
    from final_figures.make_fig4_config_insights import (
        GROUP_LABEL_SIZE,
        GROUPS,
        METRIC,
        ORDERS,
        OUT_DIR,
        PALETTE_NAME,
        SELECTED_LABEL,
        TAG_COL,
        TITLES,
        draw_boxes,
        load_trials,
        ring_selected,
        selected_handle,
        selected_trial,
        variant_use_cases,
    )
except ImportError:  # run as a plain script from inside final_figures/
    from make_fig4_config_insights import (
        GROUP_LABEL_SIZE,
        GROUPS,
        METRIC,
        ORDERS,
        OUT_DIR,
        PALETTE_NAME,
        SELECTED_LABEL,
        TAG_COL,
        TITLES,
        draw_boxes,
        load_trials,
        ring_selected,
        selected_handle,
        selected_trial,
        variant_use_cases,
    )

STEM = "fig5_relation"

MODEL_COL = "params.model"
FEATURE_COL = "metrics.nb_features"
FEATURE_LABEL = "Number of features (log scale)"
TOP_N = 1000  # trials kept for panel b, ranked by validation metric
# Metric axis of panel b. The arrow gives the improving direction, as in
# figure 4; the scale is linear, since panel b spans the best trials only.
RMSE_LABEL = "RMSE Validation (↓)"
AUC_LABEL = "ROC AUC Validation (↑)"
METRIC_LABEL = {
    "metrics.rmse_val": RMSE_LABEL,
    "metrics.roc_auc_macro_ovr_val": AUC_LABEL,
}
LOWER_IS_BETTER = {
    "metrics.rmse_val": True,
    "metrics.roc_auc_macro_ovr_val": False,
}

# Style mirrors make_fig4_config_insights, with one size for every axis
# label and one for every tick label, so panels a and b typeset alike.
LABEL_SIZE = GROUP_LABEL_SIZE
TICK_SIZE = 11
RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": LABEL_SIZE,
    "axes.titlesize": GROUP_LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
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

Y_LABEL_PAD = 95  # anchor for the left-aligned model names, in points
MARKER_SIZE = 14
MARKER_EDGE = "0.35"
LEGEND_EDGE = "#C8C8C8"
SCATTER_SEED = 0  # opaque markers: draw order is shuffled, not by model
PANEL_LETTER_X = 0.006
# Page height follows the content: the panels and the gaps between them are
# sized first, so nothing is padded out to fill an A4 sheet it need not fill.
FIG_SIZE = (8.0, 10.0)  # inches; the width of figure 4, so both typeset alike
X_LABEL_DROP_IN = 0.29  # below an axes, clear of its tick labels
PANEL_LETTER_LIFT_IN = 0.28  # above a panel's first axes
# Panel a is 6 x 0.2979 in tall, the cell height of figure 4, and runs the
# full text width; its left margin carries the model names. Panel b starts on
# the same spine; its gutter is wider than panel a's 0.142 in because every
# column keeps its metric ticks on the left, and those need 0.44 in.
PANEL_A = {"left": 0.222, "right": 0.985, "top": 0.9626, "bottom": 0.7839}
PANEL_B_BAND = {"left": 0.222, "right": 0.9194}
MARGIN_RIGHT_IN = 0.12
MARGIN_BOTTOM_IN = 1.15  # under panel b: its axis name, then the legend
SELECTED_LEGEND_IN = 0.28  # the legend's second row, which holds the ring entry
PANEL_GAP_IN = 0.72  # between panel a and panel b
LEGEND_WIDTH_IN = 6.95  # the six model entries on one row, measured
COLUMN_GAP = 0.10  # wspace of panel a at the size it was tuned
PANEL_A_GAP_IN = 0.142  # the gap that wspace works out to, held as cells grow
# Panel b keeps two use cases per row, so a row is one metric. Four use cases
# make a 2 x 2 block on panel b's own narrower band, with square cells. Three
# or fewer fit on one row, and that row is laid out on panel a's columns
# instead, so the two panels line up; the page then loses the height the
# second row would have taken. A single row needs the wider gutter of the two,
# because every column carries its own metric ticks and the column that opens
# a metric carries its name as well.
PANEL_B_COLUMNS = 2
PANEL_B_SINGLE_ROW_MAX = 3
# One gap per panel b layout, used both across and down, so its cells are
# evenly spaced in either direction. The single row needs the wider of the
# two: its middle column opens a metric and so carries the name as well.
PANEL_B_GAP_IN = 0.52
PANEL_B_ROW_GAP_IN = 0.72
# Both panels' cells against the bands they were originally tuned in. The
# page widens to carry the increase; the gaps do not scale with it, since
# what they hold -- tick labels, a metric name -- has not changed size.
PANEL_SCALE = 1.10
# Sub-decade tick steps, densest first. A panel narrow enough that the
# scientific labels of one step would collide falls back to the next.
SUB_DECADE_STEPS = ((1.0, 2.0, 5.0), (1.0, 3.0), (1.0,))
TICK_LABEL_GAP_PT = 2.0
# Metric ticks land on whole hundredths, so every label reads to 2 decimals.
DECIMAL_STEPS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
MAX_TICK_INTERVALS = 5


def page_layout(count: int) -> dict:
    """Page size, panel boxes, grid spacing and legend anchor for `count`.

    Panel b is a 2 x 2 block on its own band for the full set, and a single
    row for three use cases or fewer -- laid out then on panel a's columns,
    so the two panels align and panel a widens with it. Either way every cell
    of a panel gets one size and one gap, across and down, and the page is
    whatever the panels, the gaps and the legend add up to.
    """
    single = count <= PANEL_B_SINGLE_ROW_MAX
    band = PANEL_A if single else PANEL_B_BAND
    gap = PANEL_B_ROW_GAP_IN if single else PANEL_B_GAP_IN
    rows, columns = (
        (1, count) if single else (-(-count // PANEL_B_COLUMNS), PANEL_B_COLUMNS)
    )

    # Cells keep the size the band was tuned for, scaled; the span follows.
    tuned = (band["right"] - band["left"]) * FIG_SIZE[0]
    cell = (tuned - (columns - 1) * gap) / columns * PANEL_SCALE
    span = columns * cell + (columns - 1) * gap
    panel_b_height = rows * cell + (rows - 1) * gap
    wspace = gap / cell

    left = PANEL_A["left"] * FIG_SIZE[0]
    # A single row shares panel a's columns; otherwise panel a keeps its own
    # count and gutter, and its cells take the same scaling.
    if single:
        panel_a_span, panel_a_wspace = span, wspace
    else:
        tuned_a = (PANEL_A["right"] - PANEL_A["left"]) * FIG_SIZE[0]
        cell_a = (tuned_a - (count - 1) * PANEL_A_GAP_IN) / count * PANEL_SCALE
        panel_a_span = count * cell_a + (count - 1) * PANEL_A_GAP_IN
        panel_a_wspace = PANEL_A_GAP_IN / cell_a
    centre = left + span / 2
    # Wide enough for either panel, and for the legend wherever it is anchored.
    width = max(
        left + panel_a_span + MARGIN_RIGHT_IN,
        left + span + MARGIN_RIGHT_IN,
        centre + LEGEND_WIDTH_IN / 2 + MARGIN_RIGHT_IN,
    )

    margin_top = (1.0 - PANEL_A["top"]) * FIG_SIZE[1]
    panel_a_height = (PANEL_A["top"] - PANEL_A["bottom"]) * FIG_SIZE[1]
    height = (
        margin_top
        + panel_a_height
        + PANEL_GAP_IN
        + panel_b_height
        + MARGIN_BOTTOM_IN
        + SELECTED_LEGEND_IN
    )

    a_top = 1.0 - margin_top / height
    a_bottom = a_top - panel_a_height / height
    b_top = a_bottom - PANEL_GAP_IN / height
    panel_a_box = {
        "left": left / width,
        "right": (left + panel_a_span) / width,
        "top": a_top,
        "bottom": a_bottom,
    }
    return {
        "figsize": (width, height),
        "panel_a": panel_a_box,
        "panel_b": {
            "left": left / width,
            "right": (left + span) / width,
            "top": b_top,
            "bottom": b_top - panel_b_height / height,
        },
        # Panel a keeps its own tighter gutter unless it is sharing panel b's
        # columns: four columns cannot take panel b's without their decade
        # tick labels running together.
        "wspace_a": panel_a_wspace,
        "wspace_b": wspace,
        "hspace_b": wspace,
        "shape": (rows, columns),
        # The legend belongs to the plots it labels, so it is centred under
        # panel b rather than on the page; the page is widened above to keep
        # all six entries on the sheet at that anchor.
        "legend_x": centre / width,
    }


def row_colors() -> dict[str, tuple]:
    """Figure 4's category -> Set3 slot assignment, reproduced verbatim."""
    cmap = plt.get_cmap(PALETTE_NAME)
    colors: dict[str, tuple] = {}
    for col, _ in GROUPS:
        for i, cat in enumerate(ORDERS[col]):
            colors.setdefault(cat, cmap(i % cmap.N))
    return colors


def top_trials(df, metric: str):
    """The TOP_N best-scoring trials that report a feature count."""
    ranked = df.dropna(subset=[metric, FEATURE_COL]).sort_values(
        metric, ascending=LOWER_IS_BETTER[metric]
    )
    return ranked.head(TOP_N)


def log_x_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(numticks=4))
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def x_labels_collide(ax: plt.Axes) -> bool:
    """Whether the x tick labels currently in view touch one another."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lo, hi = ax.get_xlim()
    boxes = sorted(
        (
            label.get_window_extent(renderer)
            for loc, label in zip(ax.get_xticks(), ax.get_xticklabels())
            if lo <= loc <= hi and label.get_text()
        ),
        key=lambda box: box.x0,
    )
    room = TICK_LABEL_GAP_PT * fig.dpi / 72
    return any(b.x0 - a.x1 < room for a, b in zip(boxes, boxes[1:]))


def widen_log_ticks(ax: plt.Axes) -> None:
    """Label 2x and 5x steps where the view spans less than two decades.

    Use case 2's best trials sit inside one decade, which the decade-only
    locator of figure 4 labels with a single tick. Scientific labels are wide,
    though, so a narrow panel -- panel b on a single row -- steps back to a
    sparser set rather than run them together.
    """
    ax.autoscale_view()
    lo, hi = ax.get_xlim()
    if np.log10(hi / lo) >= 2:
        return
    for subs in SUB_DECADE_STEPS:
        ax.xaxis.set_major_locator(mticker.LogLocator(subs=subs))
        # The default formatter drops everything but exact powers once the
        # view spans more than one decade; these ticks are majors, so label
        # them all.
        ax.xaxis.set_major_formatter(
            mticker.LogFormatterSciNotation(minor_thresholds=(np.inf, np.inf))
        )
        if not x_labels_collide(ax):
            return


def two_decimal_ticks(ax: plt.Axes) -> None:
    """Metric ticks on the coarsest hundredth step that still fills the axis."""
    ax.autoscale_view()
    lo, hi = ax.get_ylim()
    step = next(
        (s for s in DECIMAL_STEPS if (hi - lo) / s <= MAX_TICK_INTERVALS),
        DECIMAL_STEPS[-1],
    )
    ax.yaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


def shared_x_label(
    fig: plt.Figure, left: plt.Axes, right: plt.Axes, drop: float
) -> None:
    """One x label under a row of axes that all measure the same quantity."""
    box = left.get_position()
    fig.text(
        (box.x0 + right.get_position().x1) / 2,
        box.y0 - drop,
        FEATURE_LABEL,
        ha="center",
        va="top",
        fontsize=LABEL_SIZE,
    )


def panel_letter(fig: plt.Figure, ax: plt.Axes, letter: str, lift: float) -> None:
    fig.text(
        PANEL_LETTER_X,
        ax.get_position().y1 + lift,
        letter,
        fontsize=14,
        fontweight="bold",
        va="top",
    )


def draw_panel_a(
    fig: plt.Figure,
    trials: dict,
    models: list,
    colors: dict,
    use_cases: list,
    geom: dict,
) -> plt.Axes:
    """Feature-set size per model family, all trials, one column per use case."""
    gs = GridSpec(
        1,
        len(use_cases),
        figure=fig,
        wspace=geom["wspace_a"],
        **geom["panel_a"],
    )
    axes: list[plt.Axes] = []
    for c, uc in enumerate(use_cases):
        df = trials[uc]
        ax = fig.add_subplot(gs[0, c], sharex=axes[0] if axes else None)
        data = [df.loc[df[MODEL_COL] == m, FEATURE_COL].dropna().values for m in models]
        positions = np.arange(len(models))
        drawn = [i for i, d in enumerate(data) if len(d)]
        if drawn:
            draw_boxes(ax, data, drawn, positions, models, colors)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.set_yticks(positions)
        log_x_axis(ax)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_title(TITLES[uc], pad=5)
        if c == 0:
            ax.set_yticklabels(models)
            # Left-align: anchor clear of the axis, text runs towards it.
            for label in ax.get_yticklabels():
                label.set_horizontalalignment("left")
            ax.tick_params(axis="y", pad=Y_LABEL_PAD)
            ax.set_ylabel("Model type", labelpad=12)
        else:
            ax.set_yticklabels([])
        axes.append(ax)
    # One axis for every column, so one label under the row.
    shared_x_label(fig, axes[0], axes[-1], geom["x_label_drop"])
    return axes[0]


def draw_panel_b(
    fig: plt.Figure,
    tops: dict,
    selected: dict,
    colors: dict,
    use_cases: list,
    geom: dict,
) -> plt.Axes:
    """Size against performance for the best trials, two use cases per row.

    The trial reported for each use case is ringed.
    """
    rows, columns = geom["shape"]
    gs = GridSpec(
        rows,
        columns,
        figure=fig,
        wspace=geom["wspace_b"],
        hspace=geom["hspace_b"],
        **geom["panel_b"],
    )
    rng = np.random.default_rng(SCATTER_SEED)
    axes: list[plt.Axes] = []
    for i, uc in enumerate(use_cases):
        row, col = divmod(i, columns)
        ax = fig.add_subplot(gs[row, col])
        top, metric = tops[uc], METRIC[uc][0]
        shuffled = rng.permutation(len(top))
        ax.scatter(
            top[FEATURE_COL].to_numpy()[shuffled],
            top[metric].to_numpy()[shuffled],
            c=[colors[m] for m in top[MODEL_COL].to_numpy()[shuffled]],
            s=MARKER_SIZE,
            linewidths=0.25,
            edgecolors=MARKER_EDGE,
            zorder=3,
        )
        ring_selected(ax, selected[uc][FEATURE_COL], selected[uc][metric])
        log_x_axis(ax)
        widen_log_ticks(ax)
        two_decimal_ticks(ax)
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(TITLES[uc], pad=5)
        # Consecutive use cases share a metric: the column that opens one
        # names it, whether that is the start of a row or the middle of one.
        if i == 0 or METRIC[use_cases[i - 1]][0] != metric:
            ax.set_ylabel(METRIC_LABEL[metric])
        axes.append(ax)
    # The trial axis is named under the bottom row, however full it is.
    bottom = axes[(rows - 1) * columns :]
    shared_x_label(fig, bottom[0], bottom[-1], geom["x_label_drop"])
    return axes[0]


def main(use_cases: list[str] | None = None, suffix: str = "") -> None:
    """Render the figure; `use_cases` defaults to the variant `suffix` names."""
    use_cases = variant_use_cases(use_cases, suffix)
    mpl.rcParams.update(RC_PARAMS)

    trials = load_trials(use_cases)
    tops = {uc: top_trials(trials[uc], METRIC[uc][0]) for uc in use_cases}
    selected = {}
    for uc in use_cases:
        metric = METRIC[uc][0]
        mode = "min" if LOWER_IS_BETTER[metric] else "max"
        pick = selected_trial(trials[uc], metric, mode)
        drawn = (
            tops[uc].index.get_loc(pick.name) + 1
            if pick.name in tops[uc].index
            else None
        )
        print(
            f"{uc}: {pick[TAG_COL]} ({pick[MODEL_COL]}) "
            f"{metric.removeprefix('metrics.')} {pick[metric]:.4f}, "
            f"{pick[FEATURE_COL]:.0f} features, "
            + (
                f"rank {drawn} of the {len(tops[uc])} drawn"
                if drawn
                else "outside the trials panel b draws"
            )
        )
        selected[uc] = pick
    colors = row_colors()
    models = ORDERS[MODEL_COL]

    geom = page_layout(len(use_cases))
    geom["x_label_drop"] = X_LABEL_DROP_IN / geom["figsize"][1]
    lift = PANEL_LETTER_LIFT_IN / geom["figsize"][1]

    fig = plt.figure(figsize=geom["figsize"])
    ax_a = draw_panel_a(fig, trials, models, colors, use_cases, geom)
    ax_b = draw_panel_b(fig, tops, selected, colors, use_cases, geom)
    panel_letter(fig, ax_a, "a", lift)
    panel_letter(fig, ax_b, "b", lift)

    # The legend gets an empty second row under the model types and the ring
    # entry is centred in it: it belongs with the other entries but is not a
    # model type, so it has the row to itself.
    blank = plt.Line2D([], [], linestyle="none")
    handles, labels = [], []
    for m in models:  # columns fill top to bottom: a model type over a blank
        handles += [
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=6,
                markerfacecolor=colors[m],
                markeredgecolor=MARKER_EDGE,
                markeredgewidth=0.4,
            ),
            blank,
        ]
        labels += [m, " "]
    legend = fig.legend(
        handles,
        labels,
        title="Model type",
        loc="lower center",
        bbox_to_anchor=(geom["legend_x"], 0.0),
        ncol=len(models),
        handletextpad=0.4,
        columnspacing=1.2,
        fontsize=TICK_SIZE,
        title_fontsize=TICK_SIZE,
        frameon=True,
        edgecolor=LEGEND_EDGE,
        facecolor="white",
        framealpha=1.0,  # EPS has no alpha channel
    )
    legend.get_frame().set_linewidth(0.8)
    fig.canvas.draw()
    inner = legend.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted()
    )
    fig.legend(
        [selected_handle()],
        [SELECTED_LABEL],
        loc="lower center",
        bbox_to_anchor=((inner.x0 + inner.x1) / 2, inner.y0),
        borderpad=0,
        handletextpad=0.4,
        fontsize=TICK_SIZE,
        frameon=False,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("eps", "pdf", "png"):
        fig.savefig(OUT_DIR / f"{STEM}{suffix}.{ext}", format=ext, dpi=600)
        print(f"Wrote {OUT_DIR / f'{STEM}{suffix}.{ext}'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
