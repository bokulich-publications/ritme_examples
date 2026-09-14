"""Figure 6: how each search improves over its trials, per use case.

An A4 portrait page merging the per-use-case ``trend_over_time_*_tpe`` views
onto one grid: a column per use case (1-3, or all four as the `_w_u4`
variant), a row per model family. Every cell shows the raw validation metric
of each trial against its position in that model's search, with a rolling mean
and a +/- 1 SD band on top, so the figure answers one question -- did the
sampler still gain ground by the end of the budget? The trial reported for
each use case -- ritme's one-standard-error pick of the winning search, as
`use_cases/evaluate_all_trials.ipynb` selects it -- is ringed in black in the
cell of its model family.

Rows, run selection, use-case titles, the metric per use case (and hence
which axes are log-scaled) and the Set3 colour per model family are all
imported from figures 4 and 5, so the three figures cannot drift apart.
Every panel keeps its own y range, since use case 1's `linreg` excursions
(up to 5e9 RMSE) do not share an axis with `xgb` (2.3-4.0). The columns are
laid out as two blocks of two, one per metric, so the metric is named once
down the left of the pair that shares it.

Two choices the panel size forces:

* the smoothing window adapts (``window = clip(n // 4, 5, 100)``), so that a
  search of only tens of trials is not smoothed into a flat global mean.
  Each panel states its own ``w``.
* transparency is replaced by opaque tints (EPS has no alpha channel), the
  same choice figure 5 makes for its scatter.

Usage (ritme_usecases env), from the repo root or from this directory:
    python -m final_figures.make_fig6_time_course   # use cases 1-3
    python make_fig6_time_course.py
    python -m final_figures.make_w_u4               # all four, as `_w_u4`
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from matplotlib.gridspec import GridSpec

try:
    from final_figures.make_fig4_config_insights import (
        AUC_LABEL,
        METRIC,
        ORDERS,
        OUT_DIR,
        RMSE_LABEL,
        SELECTED_LABEL,
        TITLES,
        load_trials,
        ring_selected,
        selected_handle,
        selected_trial,
        variant_use_cases,
    )
    from final_figures.make_fig5_relation import (
        LEGEND_EDGE,
        LOWER_IS_BETTER,
        RC_PARAMS,
        row_colors,
    )
except ImportError:  # run as a plain script from inside final_figures/
    from make_fig4_config_insights import (
        AUC_LABEL,
        METRIC,
        ORDERS,
        OUT_DIR,
        RMSE_LABEL,
        SELECTED_LABEL,
        TITLES,
        load_trials,
        ring_selected,
        selected_handle,
        selected_trial,
        variant_use_cases,
    )
    from make_fig5_relation import LEGEND_EDGE, LOWER_IS_BETTER, RC_PARAMS, row_colors

STEM = "fig6_time_course"

MODEL_COL = "params.model"
NAME_SEPARATOR = "|"  # between a family's regression and classification names
TAG_COL = "tags.experiment_tag"
TIME_COL = "start_time"
X_LABEL = "Trial within search"
# The metric axis is named in figure 4's words, set on one line.
METRIC_LABEL = {
    "metrics.rmse_val": RMSE_LABEL.replace("\n", " "),
    "metrics.roc_auc_macro_ovr_val": AUC_LABEL.replace("\n", " "),
}
BLOCK_SIZE = 2  # use cases per metric: 1 and 2 on RMSE, 3 and 4 on ROC AUC

# Rolling window: the notebook's 100, but never more than a quarter of the
# search, so short searches still show a trend rather than their mean.
WINDOW = 100
MIN_WINDOW = 5
WINDOW_FRACTION = 4

# Every point size on the page is figure 5's, a tenth larger; the page grows
# with them so the layout keeps its proportions. Only the height grows -- the
# width is already close to A4's 8.27 in.
FONT_SCALE = 1.10
POINT_SIZES = (
    "font.size",
    "axes.labelsize",
    "axes.titlesize",
    "xtick.labelsize",
    "ytick.labelsize",
)
RC_SCALED = {k: v * FONT_SCALE if k in POINT_SIZES else v for k, v in RC_PARAMS.items()}
# The page is A4's width, so it can be placed without scaling, and short
# enough that a long caption still fits under it on the same sheet.
SELECTED_LEGEND_IN = (
    0.34  # the framed legend under the trial axis name, naming the ring
)
PAGE = (8.27, 8.6 + SELECTED_LEGEND_IN)
# 24 panels on one page: ticks sit between figure 4's cramped-panel size and
# figure 5's, every other size is inherited from figures 4 and 5.
TICK_SIZE = 9.5 * FONT_SCALE
NOTE_SIZE = 7.0 * FONT_SCALE
# The column titles are the reference: every other label on the page -- the
# model names, the metric names, the trial axis -- is set from their size.
LABEL_SIZE = RC_SCALED["axes.titlesize"]
MAX_X_TICKS = 4
MAX_Y_TICKS = 4
WIDE_DECADES = 2.0  # from here up, whole decades are enough
NARROW_DECADES = 0.6  # below this, log ticks are too sparse: space linearly
# Tick steps that read to one decimal, as figure 5's metric ticks read to two.
DECIMAL_STEPS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
MAX_TICK_INTERVALS = 5
MIN_TICKS = 2

RAW_COLOR = "0.62"  # opaque: readable both on white and on the SD band
# Marker area falls with the trial count: without alpha, a fixed size turns
# the 10k-trial searches into a solid block that buries the SD band, while
# the 10-trial ones would be nearly invisible.
RAW_SIZE_SCALE = 45.0
RAW_SIZE_LIMITS = (0.4, 3.0)
BAND_TINT = 0.32  # weight of the model colour against white
LINE_VALUE = 0.62  # HSV value of the trend line, so pale Set3 slots read
LINE_SATURATION = 1.35
LINE_WIDTH = 1.4
# A ring in this top-right region of a panel moves the window note to the
# top left, out of its way.
NOTE_CLEAR = (0.70, 0.78)

# Page geometry. Every margin holds type, so it is given in inches and
# turned into a figure fraction against PAGE -- reshaping the page then
# leaves the type its room and only the panels resize. Rows are spaced
# uniformly and every column divide is the same width but one: the divide
# between use case 2 and use case 3 is doubled, marking the regression /
# classification split that the two metrics also announce. Reading leftwards
# from a panel: its tick labels, then -- for the first column of a metric --
# that metric's name; the model names are set once at the outer margin.
MARGIN_TOP_IN = 0.35  # the column titles
MARGIN_BOTTOM_IN = (
    0.68 + SELECTED_LEGEND_IN
)  # tick labels, trial axis name, legend line
MARGIN_LEFT_IN = 1.50  # model names, metric name, tick labels
MARGIN_RIGHT_IN = 0.12
COLUMN_GAP_IN = 0.49  # every divide but the one below
COLUMN_SPLIT = 2  # columns before the widened divide
COLUMN_GAP_FACTOR = 2.0  # that divide, as a multiple of every other one
HSPACE = 0.24
Y_LABEL_CHANNEL_IN = 0.18  # a metric name to the tick labels beside it
ROW_LABEL_X_IN = 0.30  # model names, rotated, at the outer margin
X_LABEL_DROP_IN = 0.50  # below the bottom axes, clear of their tick labels

GRID_TOP = 1.0 - MARGIN_TOP_IN / PAGE[1]
GRID_BOTTOM = MARGIN_BOTTOM_IN / PAGE[1]
AXES_LEFT = MARGIN_LEFT_IN / PAGE[0]
AXES_RIGHT = 1.0 - MARGIN_RIGHT_IN / PAGE[0]
COLUMN_GAP = COLUMN_GAP_IN / PAGE[0]
Y_LABEL_CHANNEL = Y_LABEL_CHANNEL_IN / PAGE[0]
ROW_LABEL_X = ROW_LABEL_X_IN / PAGE[0]
X_LABEL_DROP = X_LABEL_DROP_IN / PAGE[1]


def tint(color, weight: float) -> tuple:
    """`color` at `weight` over white -- what alpha would look like, opaque."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb * weight + (1.0 - weight))


def deepen(color) -> tuple:
    """A darker, more saturated twin of a Set3 slot, for line work."""
    h, s, v = mcolors.rgb_to_hsv(mcolors.to_rgb(color))
    return tuple(mcolors.hsv_to_rgb((h, min(1.0, s * LINE_SATURATION), v * LINE_VALUE)))


def one_decimal_ticks(ax: plt.Axes) -> None:
    """Ticks on the coarsest step that still fills the axis, read to 0.1.

    The y twin of figure 5's `two_decimal_ticks`, one digit shorter because
    these panels are a third of the width.
    """
    ax.autoscale_view()
    lo, hi = ax.get_ylim()
    step = next(
        (s for s in DECIMAL_STEPS if (hi - lo) / s <= MAX_TICK_INTERVALS),
        DECIMAL_STEPS[-1],
    )
    if np.floor(hi / step) - np.ceil(lo / step) + 1 < MIN_TICKS:
        # A panel narrower than one step -- use case 4's `xgb` spans 0.01 ROC
        # AUC -- would carry no label at all; widen it to enclosing steps.
        widened = (np.floor(lo / step) * step, np.ceil(hi / step) * step)
        if widened[0] > 0 or ax.get_yscale() != "log":
            ax.set_ylim(*widened)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))


def log_y_ticks(ax: plt.Axes) -> None:
    """Readable ticks on a log axis, whatever the panel happens to span.

    The searches here range from a quarter of a decade (`xgb` on use case 1)
    to eight (`linreg`), and no single locator serves both: decade ticks
    leave the narrow panels blank, while `LogLocator` falls back to linear
    ticks below a decade. Pick the locator from the span; only the panels
    that really reach the scientific range are labelled as powers.
    """
    ax.autoscale_view()
    lo, hi = ax.get_ylim()
    decades = np.log10(hi / lo)
    if decades >= WIDE_DECADES:
        ax.yaxis.set_major_locator(mticker.LogLocator(numticks=MAX_Y_TICKS))
        ax.yaxis.set_major_formatter(
            mticker.LogFormatterSciNotation(minor_thresholds=(np.inf, np.inf))
        )
    elif decades >= NARROW_DECADES:
        ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    else:
        one_decimal_ticks(ax)
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def searches(df: pd.DataFrame, metric: str) -> list[pd.DataFrame]:
    """One time-ordered frame per search, indexed from trial 1.

    Trials whose metric is missing keep their slot, so the x axis counts
    trials attempted rather than trials that scored.
    """
    out = []
    for tag in pd.unique(df[TAG_COL].dropna()):
        d = df.loc[df[TAG_COL] == tag, [metric, TIME_COL]].copy()
        d[TIME_COL] = pd.to_datetime(d[TIME_COL], format="ISO8601", errors="coerce")
        d = d.sort_values(TIME_COL)
        d["row"] = d.index  # the trial's row in the use case's log
        d = d.reset_index(drop=True)
        d["trial"] = np.arange(1, len(d) + 1)
        window = int(np.clip(len(d) // WINDOW_FRACTION, MIN_WINDOW, WINDOW))
        roll = d[metric].rolling(window=window, center=True, min_periods=1)
        d["mean"] = roll.mean()
        d["sd"] = roll.std(ddof=0)
        d.attrs["window"] = window
        d.attrs["tag"] = tag
        out.append(d)
    return out


def compact(value: float, _pos=None) -> str:
    """Tick labels that fit a one inch panel: 8000 -> 8k."""
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def draw_panel(
    ax: plt.Axes,
    parts: list[pd.DataFrame],
    metric: str,
    log: bool,
    color,
    selected: tuple[float, float] | None = None,
):
    """One search family in one use case; `selected` rings the reported trial."""
    band = tint(color, BAND_TINT)
    line = deepen(color)
    floor = (
        min(float(d.loc[d[metric] > 0, metric].min()) for d in parts) * 0.7
        if log
        else None
    )
    for d in parts:
        lo = d["mean"] - d["sd"]
        hi = d["mean"] + d["sd"]
        if log:
            # A symmetric SD band reaches below zero on wide-tailed searches;
            # hold it at the data floor rather than dropping the whole patch.
            lo = lo.clip(lower=floor)
            hi = hi.clip(lower=floor)
        size = float(np.clip(RAW_SIZE_SCALE / np.sqrt(len(d)), *RAW_SIZE_LIMITS))
        ax.fill_between(d["trial"], lo, hi, color=band, linewidth=0, zorder=2)
        ax.scatter(d["trial"], d[metric], color=RAW_COLOR, s=size, zorder=3)
        ax.plot(d["trial"], d["mean"], color=line, linewidth=LINE_WIDTH, zorder=4)

    if log:
        ax.set_yscale("log")
        log_y_ticks(ax)
    else:
        one_decimal_ticks(ax)
    ax.set_xlim(0, max(int(d["trial"].iloc[-1]) for d in parts))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(MAX_X_TICKS, integer=True))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(compact))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True)
    ax.set_axisbelow(True)

    note_x, note_ha = 0.97, "right"
    if selected is not None:
        ring_selected(ax, *selected)
        ax.get_ylim()  # settle the autoscaled view before reading positions
        fx, fy = ax.transAxes.inverted().transform(ax.transData.transform(selected))
        if fx > NOTE_CLEAR[0] and fy > NOTE_CLEAR[1]:
            note_x, note_ha = 0.03, "left"
    windows = sorted({d.attrs["window"] for d in parts})
    ax.text(
        note_x,
        0.94,
        f"w={'/'.join(str(w) for w in windows)}",
        transform=ax.transAxes,
        ha=note_ha,
        va="top",
        fontsize=NOTE_SIZE,
        color="0.35",
        zorder=5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )


def page_grids(fig: plt.Figure, rows: int, columns: int) -> list[GridSpec]:
    """Two side-by-side grids holding all 24 panels, split at `COLUMN_SPLIT`.

    A single GridSpec spaces every column alike; the regression cases and
    the classification cases want a wider divide between them, so each pair
    gets its own grid and the divide between the two is `COLUMN_GAP_FACTOR`
    times the others. Both grids share the vertical geometry and `hspace`,
    so panels stay identical and rows stay uniformly spaced and aligned.
    """
    # One normal divide inside each grid, plus the widened one between them.
    gap_units = columns - 2 + COLUMN_GAP_FACTOR
    width = (AXES_RIGHT - AXES_LEFT - gap_units * COLUMN_GAP) / columns

    grids, left = [], AXES_LEFT
    for count in (COLUMN_SPLIT, columns - COLUMN_SPLIT):
        right = left + count * width + (count - 1) * COLUMN_GAP
        grids.append(
            GridSpec(
                rows,
                count,
                figure=fig,
                wspace=COLUMN_GAP / width,
                hspace=HSPACE,
                left=left,
                right=right,
                top=GRID_TOP,
                bottom=GRID_BOTTOM,
            )
        )
        left = right + COLUMN_GAP_FACTOR * COLUMN_GAP
    return grids


def visible_y_labels(ax: plt.Axes) -> list:
    """Tick labels actually inside the view -- locators emit others too."""
    if not ax.axison:
        return []
    lo, hi = ax.get_ylim()
    return [
        t
        for loc, t in zip(ax.get_yticks(), ax.get_yticklabels())
        if lo <= loc <= hi and t.get_text()
    ]


def align_column_labels(fig: plt.Figure, axes: list, renderer) -> None:
    """Left-align a column's y tick labels, padded out to its widest.

    Tick labels hang right-aligned off the axis, so a row whose numbers are
    wider -- `10^8` against `2.5` -- reaches further left and narrows the
    channel between the metric name and the numbers. Pad the whole column
    out to its widest label instead, so every row keeps one channel.
    """
    widths = [
        t.get_window_extent(renderer).width for ax in axes for t in visible_y_labels(ax)
    ]
    pad = mpl.rcParams["ytick.major.pad"] + max(widths) * 72 / fig.dpi
    for ax in axes:
        ax.tick_params(axis="y", pad=pad)
        for label in ax.get_yticklabels():
            label.set_horizontalalignment("left")


def label_left_edge(axes: list, renderer, width_px: float) -> float:
    """Where a column's tick labels start, in figure fractions."""
    return (
        min(
            t.get_window_extent(renderer).x0
            for ax in axes
            for t in visible_y_labels(ax)
        )
        / width_px
    )


def fit_row_label(label, renderer, pitch_px: float) -> None:
    """Break a model name over two lines when it outgrows its row.

    Set upright, `nn_reg | nn_class` is longer than the row it names once the
    page is short enough to leave a caption its space. Splitting at the
    separator halves that reach without touching the name itself.
    """
    if label.get_window_extent(renderer).height <= pitch_px:
        return
    label.set_text(
        label.get_text().replace(f" {NAME_SEPARATOR} ", f" {NAME_SEPARATOR}\n")
    )
    label.set_multialignment("center")


def name_metric(fig: plt.Figure, renderer, text: str, right_of: float) -> None:
    """Set a metric name upright of the column at `right_of`, one channel off."""
    label = fig.text(
        0.0,
        (GRID_TOP + GRID_BOTTOM) / 2,
        text,
        ha="center",
        va="center",
        rotation=90,
        fontsize=LABEL_SIZE,
    )
    half = label.get_window_extent(renderer).width / 2 / (fig.dpi * fig.get_figwidth())
    label.set_x(right_of - Y_LABEL_CHANNEL - half)


def main(use_cases: list[str] | None = None, suffix: str = "") -> None:
    """Render the figure; `use_cases` defaults to the variant `suffix` names."""
    use_cases = variant_use_cases(use_cases, suffix)
    mpl.rcParams.update(RC_SCALED)

    trials = load_trials(use_cases)
    models = ORDERS[MODEL_COL]
    colors = row_colors()
    selected = {
        uc: selected_trial(
            trials[uc],
            METRIC[uc][0],
            "min" if LOWER_IS_BETTER[METRIC[uc][0]] else "max",
        )
        for uc in use_cases
    }

    fig = plt.figure(figsize=PAGE)
    grids = page_grids(fig, len(models), len(use_cases))

    col_axes: dict[int, list[plt.Axes]] = {}
    bottom_axes: dict[int, plt.Axes] = {}
    row_axes: dict[int, plt.Axes] = {}
    for r, model in enumerate(models):
        for c, uc in enumerate(use_cases):
            metric, _, log = METRIC[uc]
            df = trials[uc]
            parts = [
                d
                for d in searches(df[df[MODEL_COL] == model], metric)
                if d[metric].notna().any()
            ]
            block, column = divmod(c, COLUMN_SPLIT)
            ax = fig.add_subplot(grids[block][r, column])
            if parts:
                ring = None
                pick = selected[uc]
                if pick[MODEL_COL] == model:
                    part = next(d for d in parts if d.attrs["tag"] == pick[TAG_COL])
                    x = int(part.loc[part["row"] == pick.name, "trial"].iloc[0])
                    ring = (x, pick[metric])
                    print(
                        f"{uc}: ring on {pick[TAG_COL]} trial {x} of {len(part)}, "
                        f"{metric.removeprefix('metrics.')} {pick[metric]:.4f}"
                    )
                draw_panel(ax, parts, metric, log, colors[model], ring)
            else:
                # The regression-only families have no classification runs;
                # their cells stay blank.
                ax.set_axis_off()
            if r == 0:
                ax.set_title(TITLES[uc], pad=5)
            if r == len(models) - 1:
                bottom_axes[c] = ax
            col_axes.setdefault(c, []).append(ax)
            row_axes.setdefault(r, ax)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Model names run down the outer margin, clear of the metric names, so
    # each metric can sit beside the first column that measures it. Each is
    # set in its own trend colour, so a row and its curves read as one.
    pitch = abs(row_axes[1].get_position().y0 - row_axes[0].get_position().y0)
    pitch_px = pitch * PAGE[1] * fig.dpi
    for r, model in enumerate(models):
        box = row_axes[r].get_position()
        label = fig.text(
            ROW_LABEL_X,
            (box.y0 + box.y1) / 2,
            model,
            ha="center",
            va="center",
            rotation=90,
            fontsize=LABEL_SIZE,
            color=deepen(colors[model]),
        )
        fit_row_label(label, renderer, pitch_px)
    # Consecutive use cases share a metric: name it once, beside the first
    # column that measures it, one channel clear of that column's numbers.
    leading = list(range(0, len(use_cases), BLOCK_SIZE))
    for first in leading:
        align_column_labels(fig, col_axes[first], renderer)
    fig.canvas.draw()  # the new pads have to land before the names are placed
    width_px = fig.canvas.get_width_height()[0]
    for first in leading:
        name_metric(
            fig,
            renderer,
            METRIC_LABEL[METRIC[use_cases[first]][0]],
            label_left_edge(col_axes[first], renderer, width_px),
        )

    # One trial axis for all columns, so one label under the page.
    x_centre = (
        bottom_axes[0].get_position().x0
        + bottom_axes[len(use_cases) - 1].get_position().x1
    ) / 2
    fig.text(
        x_centre,
        bottom_axes[0].get_position().y0 - X_LABEL_DROP,
        X_LABEL,
        ha="center",
        va="center",
        fontsize=LABEL_SIZE,
    )
    legend = fig.legend(
        [selected_handle()],
        [SELECTED_LABEL],
        loc="lower center",
        bbox_to_anchor=(x_centre, 0.0),
        fontsize=TICK_SIZE,
        handletextpad=0.4,
        frameon=True,  # figure 5's legend frame
        edgecolor=LEGEND_EDGE,
        facecolor="white",
        framealpha=1.0,  # EPS has no alpha channel
    )
    legend.get_frame().set_linewidth(0.8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("eps", "pdf", "png"):
        fig.savefig(OUT_DIR / f"{STEM}{suffix}.{ext}", format=ext, dpi=600)
        print(f"Wrote {OUT_DIR / f'{STEM}{suffix}.{ext}'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
