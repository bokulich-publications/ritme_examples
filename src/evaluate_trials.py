import hashlib
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import seaborn as sns
from scipy.stats import kruskal

###############################
# Global plotting parameters  #
###############################
# Single source of truth for figure size and font scaling; all plotting
# functions will use these unless explicitly overridden via arguments.
GLOBAL_FIGSIZE = (7, 3)  # (width, height) in inches
GLOBAL_DPI = 400
GLOBAL_FONT_SCALE = 1.7

pio.templates.default = "seaborn"
plt.rcParams.update({"font.family": "DejaVu Sans"})
plt.style.use("tableau-colorblind10")


def _set_seaborn_context(font_scale: float | None = None):
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=font_scale or GLOBAL_FONT_SCALE)


_set_seaborn_context()


def create_color_map(df, column, cmap_name="Set3"):
    """
    Create a color map based on unique values in a specified column.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to base the color map on.
        cmap_name (str): Matplotlib colormap name.

    Returns:
        list: List of colors corresponding to each row.
        dict: Mapping of unique values to colors.
    """
    unique_vals = df[column].unique()
    cmap = plt.cm.get_cmap(cmap_name, len(unique_vals))
    color_map = {val: cmap(i) for i, val in enumerate(unique_vals)}
    colors = df[column].map(color_map).tolist()
    return colors, color_map


# Global registry to ensure consistent colors across figures per group column
GLOBAL_COLOR_REGISTRY: dict[str, dict] = {}

# Preset, hard-coded category orders per column to ensure identical colors
# across independent runs. Colors are assigned from the Set3 palette in the
# given order (index 0,1,2,...). NaN entries from the user's lists are
# intentionally omitted because rows with NaN group values are dropped before
# plotting.
PRESET_COLOR_ORDERS: dict[str, list[str]] = {
    "params.data_aggregation": [
        "tax_order",
        "tax_genus",
        "tax_class",
        "tax_family",
    ],
    "params.data_selection": [
        "abundance_topi",
        "variance_quantile",
        "abundance_quantile",
        "abundance_threshold",
        "variance_topi",
        "variance_ith",
        "abundance_ith",
        "variance_threshold",
    ],
    "params.data_transform": [
        "ilr",
        "alr",
        "rank",
        "clr",
        "pa",
    ],
    "params.data_enrich": [
        "shannon_and_metadata",
        "metadata_only",
        "shannon",
    ],
    "params.model": [
        "xgb",
        "rf",
        "nn_class",
        "nn_corn",
        "nn_reg",
        "linreg",
        "trac",
    ],
}


def get_consistent_color_map(
    df: pd.DataFrame,
    column: str,
    palette_name: str = "Set3",
    n_colors: int = 12,
):
    """
    Return a stable mapping from category -> color for the given column that
    persists across calls, ensuring the same category always gets the same
    color even in different figures.

    Strategy:
    - Use a fixed-size palette (e.g., tab20).
    - Assign each category an initial index from a stable hash of the category.
    - Resolve collisions by linear probing within the palette size.
    - Store/extend mapping in a module-level registry keyed by column.
    """
    palette = sns.color_palette(palette_name, n_colors)

    # Categories present in this dataframe (ignore NaN)
    uniques = [u for u in pd.unique(df[column]) if pd.notna(u)]

    # Start from preset mapping if defined for this column; this guarantees the
    # same category->color assignment across independent runs.
    if column in PRESET_COLOR_ORDERS:
        preset_order = PRESET_COLOR_ORDERS[column]
        existing: dict = {
            cat: palette[i % n_colors] for i, cat in enumerate(preset_order)
        }
    else:
        existing = GLOBAL_COLOR_REGISTRY.get(column, {}).copy()

    # Track used palette indices from existing mapping (if any)
    used_indices = set()
    if any(color not in palette for color in existing.values()):
        # Enforce requested palette strictly if previous colors came from a
        # different palette
        existing = {k: v for k, v in existing.items() if v in palette}
    for cat, color in existing.items():
        try:
            idx = palette.index(color)
        except ValueError:
            idx = None
        if idx is not None:
            used_indices.add(idx)

    # Fallback assignment for any unseen categories in the dataframe.
    # Use a stable hash to pick an initial index, then linear probe to avoid collisions.
    def stable_idx(val) -> int:
        h = hashlib.md5(str(val).encode("utf-8")).hexdigest()
        return int(h, 16) % n_colors

    for cat in uniques:
        if cat in existing:
            continue
        base = stable_idx(cat)
        idx = base
        assigned = False
        for _ in range(n_colors):
            if idx not in used_indices:
                existing[cat] = palette[idx]
                used_indices.add(idx)
                assigned = True
                break
            idx = (idx + 1) % n_colors
        if not assigned:
            existing[cat] = palette[base]

    GLOBAL_COLOR_REGISTRY[column] = existing
    colors = df[column].map(existing)
    return colors, existing


def plot_complexity_vs_metric(
    trials,
    metric_col,
    metric_name,
    group_col,
    group_name,
    n,
    figsize=None,
    font_scale=None,
    dpi=None,
    title: str = "",
    x_log_scale: bool = False,
):
    """Plot model complexity vs. a metric.

    Parameters allow optional override of global figure settings.

    Parameters
    ----------
    trials : pd.DataFrame
        DataFrame containing at least "metrics.nb_features", the metric column,
        and the grouping column.
    x_log_scale : bool, optional (default: False)
        If True, plot the x-axis (number of features) on a logarithmic scale.
        Requires all x values to be strictly positive; otherwise keeps linear
        scale and prints a short message.
    """
    _set_seaborn_context(font_scale)
    _, color_map = create_color_map(trials, "params.model")

    fig, ax = plt.subplots(figsize=figsize or GLOBAL_FIGSIZE, dpi=dpi or GLOBAL_DPI)
    sns.scatterplot(
        x="metrics.nb_features",
        y=metric_col,
        hue=group_col,
        palette=color_map,
        data=trials,
        s=50,
        ax=ax,
        # edgecolor="black",
        # linewidth=0.01,
    )
    # Optionally apply log scale on X (requires positive values)
    x_label = "Number of Features"
    if x_log_scale:
        x_vals = trials["metrics.nb_features"].dropna().values
        if (x_vals <= 0).any():
            print(
                "[plot_complexity_vs_metric] x_log_scale=True requested but some x "
                "values are <= 0; keeping linear scale."
            )
        else:
            ax.set_xscale("log")
            x_label += " (log scale)"

    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(metric_name, labelpad=10)
    ax.set_title(f"{title}", pad=15, fontsize=20)
    # Place legend inside bottom-right
    ax.legend(
        title=group_name,
        loc="lower right",
        borderaxespad=0.0,
    )

    # Reserve a margin on the right so the outside legend is not clipped
    plt.tight_layout(rect=(0, 0, 0.85, 1))
    plt.show()
    return fig, ax


def plot_trend_over_time(
    df,
    y_col,
    time_col="start_time",
    window=20,
    title_prefix="",
    figsize=None,
    raw_color="gray",
    raw_alpha=0.4,
    trend_color="C0",
    font_scale=None,
    dpi=None,
    first_n: int | None = None,
    y_log_scale: bool = False,
):
    """
    Plot raw points and a rolling-mean trend of y_col across trials ordered by
    time_col. The x-axis is the trial index (1..N) after sorting by time.

      df:         DataFrame with your data
      y_col:      name of the metric column (e.g. "metrics.rmse_val")
      time_col:   name of the datetime column used for ordering (default "start_time")
      window:     rolling window size
    first_n:    if provided, only the first N trials (after time sort) are plotted
    y_log_scale: if True, plot the y-axis on a logarithmic scale (requires
             all y values to be strictly positive)
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], format="ISO8601")
    df = df.sort_values(time_col)
    if first_n is not None and first_n > 0:
        df = df.head(first_n)
    # Index 1..N on x-axis
    df["trial_index"] = range(1, len(df) + 1)
    df["smoothed"] = df[y_col].rolling(window=window, center=True, min_periods=1).mean()

    _set_seaborn_context(font_scale)
    plt.figure(figsize=figsize or GLOBAL_FIGSIZE, dpi=dpi or GLOBAL_DPI)
    plt.scatter(
        df["trial_index"], df[y_col], color=raw_color, alpha=raw_alpha, label="Raw"
    )
    plt.plot(
        df["trial_index"],
        df["smoothed"],
        color=trend_color,
        linewidth=2,
        label=f"Rolling mean (w={window})",
    )
    plt.xlabel("Trial number")

    title_suffix = "across trials"
    if first_n is not None and first_n > 0:
        title_suffix += f" (first {len(df)} trials)"

    if y_col == "metrics.rmse_val":
        y_label = "RMSE Validation"
    else:
        y_label = y_col

    plt.ylabel(y_label)
    plt.title(f"{title_prefix} - {y_label} {title_suffix}")
    plt.xticks(rotation=0)
    plt.legend()
    # Optionally apply log scale on Y (requires positive values)
    if y_log_scale:
        if (df[y_col] <= 0).any() or (df["smoothed"] <= 0).any():
            print(
                "[plot_trend_over_time] y_log_scale=True requested but some y values "
                "are <= 0; keeping linear scale."
            )
        else:
            plt.yscale("log")
            plt.ylabel(f"{y_label} (log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_trend_over_time_multi_models(
    df: pd.DataFrame,
    y_col: str,
    group_col: str = "params.model",
    time_col: str = "start_time",
    models: list | None = None,
    window: int = 20,
    title_prefix: str = "Model: ",
    figsize: tuple | None = None,
    raw_color: str = "gray",
    raw_alpha: float = 0.4,
    trend_color: str = "C0",
    std_alpha: float = 0.15,
    font_scale: float | None = None,
    dpi: int | None = None,
    first_n: int | None = None,
    y_log_scale: bool = False,
):
    """
    Plot the trend over time for multiple models stacked vertically (n_models × 1)
    using a shared x-axis.

    Each row corresponds to one model from `models` (or unique values from
    `group_col` when `models` is None). Within each subplot we render raw points
    and a rolling-mean trend of `y_col` across trials ordered by `time_col`.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe containing at least `group_col`, `y_col`, and `time_col`.
    y_col : str
        Name of the metric column (e.g. "metrics.rmse_val").
    group_col : str, default "params.model"
        Column containing model identifiers.
    time_col : str, default "start_time"
        Column used for ordering trials over time. Will be parsed via pandas
        to_datetime.
    models : list[str] | None, default None
        Specific list of models to include. Defaults to all unique values in
        `group_col`.
    window : int, default 20
        Rolling mean window size.
    title_prefix : str, default "Model: "
        Prefix for each subplot title.
    figsize : tuple | None
        Overall figure size. Defaults to scaling width by number of models.
    raw_color : str, default "gray"
        Color for raw scatter points.
    raw_alpha : float, default 0.4
        Alpha for raw scatter points.
    trend_color : str, default "C0"
        Color for rolling mean line.
    font_scale : float | None
        Optional override of global seaborn font scale.
    dpi : int | None
        DPI for the figure.
    first_n : int | None
        If provided, only the first N trials per model (after time sort) are plotted.
    y_log_scale : bool, default False
        If True, plot the y-axis on a logarithmic scale in each subplot when
        all relevant values are strictly positive.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list[matplotlib.axes.Axes]
    """
    required_cols = {group_col, y_col, time_col}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for plot_trend_over_time_multi_models: {missing}"
        )

    _set_seaborn_context(font_scale)

    # Determine models to plot
    if models is None:
        models = list(pd.unique(df[group_col]))
    models = sorted([m for m in models if pd.notna(m)])
    # Prepare per-model series and track max length for shared x-axis
    series = []
    max_len = 0
    for model in models:
        d = df[df[group_col] == model].copy()
        d[time_col] = pd.to_datetime(d[time_col], format="ISO8601")
        d = d.sort_values(time_col)
        if first_n is not None and first_n > 0:
            d = d.head(first_n)
        d = d[[y_col]].copy()

        d["trial_index"] = range(1, len(d) + 1)
        roll = d[y_col].rolling(window=window, center=True, min_periods=1)
        d["smoothed"] = roll.mean()
        d["smoothed_std"] = roll.std(ddof=0)
        max_len = max(max_len, len(d))
        series.append((model, d))

    # Figure size defaults: scale height by number of models (ensure a minimum
    # per-row height of ~4 inches)
    if figsize is None:
        figsize = (GLOBAL_FIGSIZE[0], GLOBAL_FIGSIZE[1] * len(models))

    fig, axes = plt.subplots(
        nrows=len(models),
        ncols=1,
        sharex=True,
        figsize=figsize,
        dpi=dpi or GLOBAL_DPI,
    )
    if len(models) == 1:
        axes = [axes]

    # Human-friendly y-label mapping like plot_trend_over_time
    if y_col == "metrics.rmse_val":
        y_label = "RMSE Validation"
    else:
        y_label = y_col

    for ax, (model, d) in zip(axes, series):
        ax.scatter(
            d["trial_index"],
            d[y_col],
            color=raw_color,
            alpha=raw_alpha,
            s=12,
            label="Raw",
        )
        ax.plot(
            d["trial_index"],
            d["smoothed"],
            color=trend_color,
            linewidth=2,
            label=f"Rolling mean (w={window})",
        )
        # Shaded band for ±1 std around the rolling mean
        if "smoothed_std" in d.columns:
            lower = d["smoothed"] - d["smoothed_std"]
            upper = d["smoothed"] + d["smoothed_std"]
            ax.fill_between(
                d["trial_index"],
                lower,
                upper,
                color=trend_color,
                alpha=std_alpha,
                linewidth=0,
                label=("±1 std" if ax is axes[0] else None),
            )
        ax.set_title(f"{title_prefix}{model}")
        ax.grid(True)
        # Y label on the third subplot only to reduce clutter
        if ax is axes[-1]:
            y_label_printed = y_label
        else:
            y_label_printed = ""
        ax.set_ylabel(y_label_printed)
        # Optional log scale per subplot if values are positive
        if y_log_scale:
            has_nonpos = (d[y_col] <= 0).any() or (d["smoothed"] <= 0).any()
            if "smoothed_std" in d.columns:
                has_nonpos = has_nonpos or (d["smoothed_std"] <= 0).any()
            if has_nonpos:
                # keep linear for this subplot
                pass
            else:
                ax.set_yscale("log")
                if y_label_printed != "":
                    ax.set_ylabel(y_label_printed + " (log scale)")

    # Shared x-label and limits
    for ax in axes:
        if max_len > 0:
            ax.set_xlim(1, max_len)
        # X label on last subplot only to reduce clutter
        if ax is axes[-1]:
            ax.set_xlabel("Trial number")
        else:
            ax.set_xlabel("")
    # Common legend: use the last axis to collect handles
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.tight_layout()
    plt.show()
    return fig, axes


def boxplot_metric(
    trials,
    metric_col,
    metric_name,
    group_col,
    group_name,
    figsize=None,
    font_scale=None,
    dpi=None,
):
    """Plot distribution of metric within each group using a boxplot,
    sorted by increasing median, hiding outliers.

    Parameters allow overriding global figure size, font scale, and dpi.
    """
    _set_seaborn_context(font_scale)
    _, color_map = create_color_map(trials, group_col)
    medians = trials.groupby(group_col)[metric_col].median().sort_values()
    order = medians.index.tolist()

    fig, ax = plt.subplots(figsize=figsize or GLOBAL_FIGSIZE, dpi=dpi or GLOBAL_DPI)
    sns.boxplot(
        x=metric_col,
        y=group_col,
        data=trials,
        orient="h",
        order=order,
        palette=color_map,
        width=0.6,
        showfliers=False,
        linewidth=1.5,
        ax=ax,
    )

    ax.set_xlabel(metric_name, labelpad=10)
    ax.set_ylabel(group_name, labelpad=10)
    ax.set_title(f"Distribution of {metric_name} by {group_name}", pad=15)
    plt.tight_layout()
    plt.show()


def multi_boxplot_metric(
    trials: pd.DataFrame,
    metric_col: str,
    metric_name: str,
    group_specs: list,
    order_by_median: bool = True,
    figsize=None,
    font_scale=None,
    dpi=None,
    box_width: float = 0.6,
    showfliers: bool = False,
    show_separators: bool = True,
    min_gap: float = 0.006,
    sep_color: str = "lightgray",
    sep_lw: float = 0.8,
    sep_alpha: float = 0.85,
    title: str = "",
    alpha: float = 0.05,
    x_log_scale: bool = False,
):
    """Create a stacked (row-wise) set of horizontal boxplots sharing the same x-axis.

    Each element of group_specs is a (group_col, group_name) pair. For each
    group column we draw a horizontal boxplot of metric_col across its categories.

    Parameters
    ----------
    trials : pd.DataFrame
        Source dataframe containing metric and grouping columns.
    metric_col : str
        Column with numeric metric values.
    metric_name : str
        Pretty name for the x-axis / title context.
    group_specs : list[tuple[str, str]]
        List of (group_col, group_label) describing which categorical columns
        to visualize and how to label each subplot's y-axis.
    order_by_median : bool, default True
        If True, order categories within each group by increasing median of metric.
    figsize : tuple | None
        Overall figure size. Defaults to (GLOBAL_FIGSIZE[0],
        GLOBAL_FIGSIZE[1] * n_groups).
    font_scale : float | None
        Optional override of global font scale.
    dpi : int | None
        DPI override.
    box_width : float
        Width parameter passed to seaborn.boxplot.
    showfliers : bool
        Whether to show outliers.
    x_log_scale : bool, default False
        If True, set the shared x-axis to logarithmic scale (requires all
        metric values to be strictly positive).

    Notes
    -----
    For each group column, a global significance test is performed across its
    categories using the Kruskal–Wallis test. After plotting, the function
    prints the p-value per group to stdout. When a p-value underflows to 0.0,
    it is replaced by the smallest positive representable float (math.ulp(0.0)).
    The same formatted p-value label ("p=" or "p<") is also annotated in the
    top-right corner of each corresponding subplot.

    Returns
    -------
    fig, axes : matplotlib Figure and list of Axes
    """
    _set_seaborn_context(font_scale)

    n = len(group_specs)
    # Scale figure height by number of subplots if figsize not provided
    if figsize is None:
        figsize = (GLOBAL_FIGSIZE[0], GLOBAL_FIGSIZE[1] * n)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        sharex=True,
        figsize=figsize,
        dpi=dpi or GLOBAL_DPI,
    )
    if n == 1:
        axes = [axes]
    # Determine a consistent color map per unique category per group.
    palette_cache = {}
    p_values = []  # collect (group_label, p_value or None)

    MISSING_CATEGORY_LABEL = "nan"
    for ax, (gcol, glabel) in zip(axes, group_specs):
        # Keep rows with metric present; map NaN groups to a visible category label
        subset = trials[[gcol, metric_col]].copy()
        subset = subset[subset[metric_col].notna()]
        subset[gcol] = subset[gcol].astype(object)
        subset.loc[subset[gcol].isna(), gcol] = MISSING_CATEGORY_LABEL
        if subset.empty:
            ax.text(0.5, 0.5, f"No data for {glabel}", ha="center", va="center")
            ax.set_axis_off()
            continue
        # Color map: reuse if already computed for this column
        if gcol not in palette_cache:
            _, palette_cache[gcol] = get_consistent_color_map(subset, gcol)
        color_map = palette_cache[gcol]

        order = None
        if order_by_median:
            order = (
                subset.groupby(gcol)[metric_col].median().sort_values().index.tolist()
            )
        sns.boxplot(
            x=metric_col,
            y=gcol,
            data=subset,
            orient="h",
            order=order,
            palette=color_map,
            width=box_width,
            showfliers=showfliers,
            linewidth=1.2,
            ax=ax,
        )
        ax.set_ylabel(glabel)
        ax.set_xlabel("")  # we add one shared label later
        ax.grid(True, axis="x", alpha=0.3)
        # Remove spines for cleaner look
        sns.despine(ax=ax, left=False, bottom=True)

        # Significance test across categories for this group column
        # Kruskal–Wallis; requires at least 2 non-empty groups
        try:
            groups = [vals[metric_col].values for _, vals in subset.groupby(gcol)]
            groups = [g for g in groups if len(g) > 0]
            p_value = None
            if len(groups) >= 2:
                stat, p_value = kruskal(*groups)
                if p_value == 0.0:
                    # Replace 0.0 (underflow) with the smallest positive float
                    p_value = math.ulp(0.0)
            # Annotate p-value on the subplot (top-right), if available
            if p_value is not None:
                if p_value == math.ulp(0.0):
                    label = f"p<{p_value:.3g}"
                else:
                    label = f"p={p_value:.3g}"
                ax.text(
                    0.98,
                    0.98,
                    label,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                )
            p_values.append((glabel, p_value))
        except Exception:
            # If the test fails (e.g., insufficient data), skip annotation silently
            p_values.append((glabel, None))

    # Optionally apply logarithmic scale to the shared x-axis
    if x_log_scale:
        metric_vals = trials[metric_col].dropna().values
        if (metric_vals <= 0).any():
            print(
                "[multi_boxplot_metric] x_log_scale=True requested but some metric "
                "values are <= 0; keeping linear scale."
            )
        else:
            for ax in axes:
                ax.set_xscale("log")
            metric_name = metric_name + " (log scale)"
    # Shared X label on last axis
    axes[-1].set_xlabel(metric_name, labelpad=12)

    # Layout first to finalize positions
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.canvas.draw()

    # Title after layout
    fig.suptitle(
        f"{title}",
        y=0.995,
    )

    if show_separators and n > 1:
        # Gather bboxes (top to bottom)
        bboxes = [(ax, ax.get_position()) for ax in axes]
        # Sort by vertical center descending to be robust
        bboxes.sort(key=lambda t: t[1].y1, reverse=True)
        for (ax_up, bbox_up), (ax_low, bbox_low) in zip(bboxes[:-1], bboxes[1:]):
            gap = bbox_up.y0 - bbox_low.y1
            if gap <= min_gap:
                continue  # skip if axes nearly touching
            y_sep = bbox_low.y1 + gap / 2.0
            x_left = max(bbox_up.x0, bbox_low.x0)
            x_right = min(bbox_up.x1, bbox_low.x1)
            fig.add_artist(
                plt.Line2D(
                    [x_left, x_right],
                    [y_sep, y_sep],
                    color=sep_color,
                    lw=sep_lw,
                    alpha=sep_alpha,
                    solid_capstyle="butt",
                )
            )
    plt.show()

    # Print p-values per group after plotting
    print("Group significance p-values (Kruskal–Wallis):")
    for glabel, p in p_values:
        if p is None:
            print(f"  {glabel}: n/a")
        else:
            # 3 significant digits, scientific notation if very small
            if p == math.ulp(0.0):
                print(f"  {glabel}: p<{p:.3g}")
            else:
                print(f"  {glabel}: p={p:.3g}")
    return fig, axes


def plot_recent_trials_barplot(
    df: pd.DataFrame,
    y_col: str = "metrics.rmse_val",
    group_col: str = "params.data_selection",
    time_col: str = "start_time",
    n_last: int = 200,
    title: str | None = None,
    figsize: tuple | None = None,
    font_scale: float | None = None,
    dpi: int | None = None,
    y_log_scale: bool = False,
    window: int = 20,
):
    """
    Plot a bar chart of the last N conducted trials (ordered by time),
    with the y-axis showing the metric (e.g., validation RMSE) and bar
    colors indicating the category in `group_col` (e.g., "params.data_selection").

    Inputs
    - df: DataFrame containing at least [y_col, group_col, time_col].
    - y_col: Metric column to plot on the y-axis (default: "metrics.rmse_val").
    - group_col: Categorical column used to color the bars.
    - time_col: Column used to order trials over time (default: "start_time").
    - n_last: Number of most recent trials to display (default: 200).
    - title: Optional title for the figure.
    - figsize, font_scale, dpi: Optional overrides of global plotting settings.
    - y_log_scale: If True, use a logarithmic y-axis (requires positive values).

    Returns
    - fig, ax: The Matplotlib figure and axes objects.

    Notes
    - Bars are ordered from older (left) to newer (right) among the last N trials.
    - NaN values in the group column are plotted under a visible category label "nan".
    """
    # Validate required columns
    required = {y_col, group_col, time_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for plot_recent_trials_barplot: {missing}"
        )

    # Work on a copy; parse time and sort
    d = df[[y_col, group_col, time_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col], format="ISO8601", errors="coerce")
    d = d.dropna(subset=[time_col])
    d = d.sort_values(time_col)

    # Keep last N trials, drop rows without metric
    if n_last is not None and n_last > 0:
        d = d.tail(n_last)
    d = d.dropna(subset=[y_col])

    if d.empty:
        raise ValueError("No data available to plot after filtering and dropping NaNs.")

    # Map NaN groups to a visible label for legend consistency
    MISSING_CATEGORY_LABEL = "nan"
    d[group_col] = d[group_col].astype(object)
    d.loc[d[group_col].isna(), group_col] = MISSING_CATEGORY_LABEL

    # Create sequential x positions (1..k) in chronological order
    d["trial_index"] = range(1, len(d) + 1)

    # We keep positions as 1..k; x tick labels are hidden for clarity.

    # Use consistent color mapping across figures for categories in group_col
    _, color_map = get_consistent_color_map(d, group_col)
    bar_colors = d[group_col].map(color_map)

    # Set plotting context
    _set_seaborn_context(font_scale)

    # Two-column layout: left plot, right legend (to match param_binned style)
    fig = plt.figure(
        figsize=figsize or (max(GLOBAL_FIGSIZE[0], 10), GLOBAL_FIGSIZE[1]),
        dpi=dpi or GLOBAL_DPI,
    )
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 0.35], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    right_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.2)
    legend_ax = fig.add_subplot(right_gs[0, 0])
    legend_ax.axis("off")
    # Placeholder colorbar axis for parity with other plot
    cbar_ax = fig.add_subplot(right_gs[1, 0])
    cbar_ax.set_visible(False)

    # Bars and trend line
    positions = list(d["trial_index"].values)
    ax.bar(positions, d[y_col], color=bar_colors, width=0.9, edgecolor="none")
    d["smoothed"] = d[y_col].rolling(window=window, center=True, min_periods=1).mean()
    (trend_line,) = ax.plot(
        positions,
        d["smoothed"],
        color="black",
        linewidth=1,
        label=f"Rolling mean (w={window})",
    )

    # Axis labels and optional title
    y_label = "RMSE Validation" if y_col == "metrics.rmse_val" else y_col
    ax.set_ylabel(y_label)
    ax.set_xlabel("Last N trials by time →")
    if title:
        ax.set_title(title)

    # Optional log scale on y
    if y_log_scale:
        if not (d[y_col] <= 0).any():
            ax.set_yscale("log")

    # Legend in right panel: trend + categories (ordered by color map keys)
    present_cats = set(pd.unique(d[group_col]))
    ordered_present = [c for c in color_map.keys() if c in present_cats]
    cat_handles = [
        plt.matplotlib.patches.Patch(color=color_map[c], label=str(c))
        for c in ordered_present
    ]
    handles = [trend_line] + cat_handles
    if handles:
        legend_ax.legend(
            handles=handles,
            title=(
                "Trend & "
                f"{group_col.replace('params.', '').replace('_', ' ')} categories"
            ),
            loc="center left",
            borderaxespad=0.0,
            frameon=True,
            fontsize=9,
            title_fontsize=9,
            labelspacing=0.4,
            handlelength=1.2,
        )

    # X ticks and limits; hide labels for cleanliness
    k = len(d)
    if k > 40:
        step = max(1, k // 20)
        ax.set_xticks(list(range(1, k + 1, step)))
    else:
        ax.set_xticks(list(range(1, k + 1)))
    ax.set_xticklabels([])
    ax.set_xlim(0.5, k + 0.5)

    sns.despine(ax=ax)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_recent_param_binned_over_time(
    df: pd.DataFrame,
    param_col: str,
    group_col: str = "params.data_selection",
    time_col: str = "start_time",
    n_last: int = 200,
    n_bins: int = 6,
    binning: str = "quantile",  # "quantile" or "uniform"
    title: str | None = None,
    figsize: tuple | None = None,
    font_scale: float | None = None,
    dpi: int | None = None,
    y_col: str = "metrics.rmse_val",
    y_log_scale: bool = False,
    window: int = 20,
    palette_name: str = "viridis",
    show_colorbar: bool = False,
    colorbar_label: str | None = None,
):
    """
    Plot the last N trials over time (x-axis) as a bar chart of a metric
    (default: RMSE validation) on the y-axis. Bars are colored by the value
    range (bin) of the provided continuous parameter `param_col`.

    Examples: visualize how performance evolved while coloring by
    binned ranges of `params.n_estimators`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least [param_col, y_col, time_col].
    param_col : str
        Name of the continuous parameter column to bin (e.g., "params.n_estimators").
    group_col : str, default "params.data_selection"
        Unused for coloring in this plot; retained for backward compatibility.
    time_col : str, default "start_time"
        Column used to order trials over time; parsed via pandas to_datetime.
    y_col : str, default "metrics.rmse_val"
        Metric column for bar heights (y-axis).
    n_last : int, default 200
        Number of most recent trials to display.
    n_bins : int, default 6
        Number of bins to use for the parameter.
    binning : {"quantile", "uniform"}, default "quantile"
        Binning strategy: quantile-based (roughly equal counts) or uniform-width.
    title : str | None
        Optional plot title.
    figsize, font_scale, dpi : optional
        Plot styling overrides. Falls back to module-level defaults.
    y_log_scale : bool, default False
        If True, use a logarithmic y-axis (requires strictly positive values).
    window : int, default 20
        Window size for rolling-mean line over the bars.
    palette_name : str, default "viridis"
        Name of the sequential palette to map low->high bins to colors.
        Examples: "viridis", "plasma", "magma", "cividis".
    show_colorbar : bool, default False
        If True, display a colorbar in the right panel to emphasize the
        low→high mapping for the parameter values.
    colorbar_label : str | None, default None
        Optional label for the colorbar; falls back to `param_col`.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    required = {param_col, y_col, time_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns for "
            f"plot_recent_param_binned_over_time: {missing}"
        )

    # Work on a copy; parse time and sort chronologically
    d = df[[param_col, y_col, time_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col], format="ISO8601", errors="coerce")
    # d = d.dropna(subset=[time_col, param_col, y_col])
    d = d.sort_values(time_col)
    if n_last is not None and n_last > 0:
        d = d.tail(n_last)

    # Determine the bins
    try:
        if binning == "quantile":
            binned = pd.qcut(d[param_col], q=n_bins, duplicates="drop")
        else:
            binned = pd.cut(d[param_col], bins=n_bins)
    except Exception:
        # Fallback to uniform bins if quantile fails (e.g., many duplicates)
        binned = pd.cut(d[param_col], bins=n_bins)

    # Create ordered categorical labels for bins and map to colors
    binned = binned.astype("category")
    categories = list(binned.cat.categories)
    # Human-friendly bin labels for legend
    bin_labels = [
        (
            f"[{c.left:.3g}, {c.right:.3g}"
            f"{']' if c.closed in ('right', 'both') else ')'}"
            if hasattr(c, "left")
            else str(c)
        )
        for c in categories
    ]
    # Store labels as string in a new column used for color mapping
    d["param_bin_label"] = [bin_labels[categories.index(cat)] for cat in binned]

    # Build a sequential, low-to-high palette for the bins (ordered by value)
    # Use a perceptually uniform, colorblind-friendly colormap
    seq_palette = sns.color_palette(palette_name, n_colors=len(categories))
    # Map each ordered bin label to a corresponding color (low->high)
    label_to_color = {lbl: seq_palette[i] for i, lbl in enumerate(bin_labels)}

    # Sequential x positions (older -> newer)
    d["x_pos"] = range(1, len(d) + 1)

    _set_seaborn_context(font_scale)
    # Create a two-column layout: left for plot, right for legend with fixed width
    fig = plt.figure(
        figsize=figsize or (max(GLOBAL_FIGSIZE[0], 10), GLOBAL_FIGSIZE[1]),
        dpi=dpi or GLOBAL_DPI,
    )
    # Fix the relative space: keep left plot area constant irrespective of
    # legend/colorbar footprint
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 0.35], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    right_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.2)
    legend_ax = fig.add_subplot(right_gs[0, 0])
    legend_ax.axis("off")
    # Colorbar axis (created only if requested)
    cbar_ax = fig.add_subplot(right_gs[1, 0])
    cbar_ax.set_visible(False)

    # Bars: height is y_col (metric), color indicates parameter bin
    positions = list(d["x_pos"].values)
    # Map bar colors by the ordered bin labels using the sequential palette
    bar_colors = d["param_bin_label"].map(label_to_color)
    ax.bar(positions, d[y_col], color=bar_colors, width=0.9, edgecolor="none")

    # Rolling mean line over bar heights
    d["smoothed"] = d[y_col].rolling(window=window, center=True, min_periods=1).mean()
    (trend_line,) = ax.plot(
        positions,
        d["smoothed"],
        color="black",
        linewidth=1,
        label=f"Rolling mean (w={window})",
    )

    # Axis labels and title
    if y_col == "metrics.rmse_val":
        y_label = "RMSE Validation"
    else:
        y_label = y_col
    ax.set_ylabel(y_label)
    ax.set_xlabel("Last N trials by time →")
    if title:
        ax.set_title(title)

    # Optional log scale with safety check
    if y_log_scale:
        if (d[y_col] <= 0).any():
            pass
        else:
            ax.set_yscale("log")

    # Legend in the fixed right panel: parameter bin patches ordered low->high
    # plus the rolling-mean trend line
    present = set(pd.unique(d["param_bin_label"]))
    ordered_present_labels = [lbl for lbl in bin_labels if lbl in present]
    bin_handles = [
        plt.matplotlib.patches.Patch(color=label_to_color[lbl], label=str(lbl))
        for lbl in ordered_present_labels
    ]
    handles = [trend_line] + bin_handles
    if handles:
        legend_ax.legend(
            handles=handles,
            title=f"Trend & {param_col.replace('params.', '').replace('_', ' ')} bins",
            loc="center left",
            borderaxespad=0.0,
            frameon=True,
            fontsize=9,
            title_fontsize=9,
            labelspacing=0.4,
            handlelength=1.2,
        )

    # Optional colorbar below the legend to emphasize low→high mapping
    if show_colorbar:
        # Determine value range from available (non-NaN) parameter values
        param_vals = pd.to_numeric(d[param_col], errors="coerce").dropna()
        if len(param_vals) > 0:
            vmin = float(param_vals.min())
            vmax = float(param_vals.max())
            cmap = mpl.cm.get_cmap(palette_name)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar_ax.set_visible(True)
            cbar = fig.colorbar(
                sm,
                cax=cbar_ax,
                orientation="horizontal",
            )
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(colorbar_label or param_col, fontsize=9)

    # X ticks and bounds; hide labels for cleanliness
    k = len(d)
    if k > 40:
        step = max(1, k // 20)
        ax.set_xticks(list(range(1, k + 1, step)))
    else:
        ax.set_xticks(list(range(1, k + 1)))
    ax.set_xticklabels([])
    ax.set_xlim(0.5, k + 0.5)

    sns.despine(ax=ax)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()
    return fig, ax
