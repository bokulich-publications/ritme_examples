import hashlib
import math

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


def _static_scatter(
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
):
    """Internal helper for static scatter plot.

    Parameters mirror public wrapper; figsize/font_scale/dpi can override
    global defaults when provided.
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

    ax.set_xlabel("Number of Features", labelpad=10)
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
):
    """Plot model complexity vs. a metric.

    Parameters allow optional override of global figure settings.
    """
    fig, ax = _static_scatter(
        trials,
        metric_col,
        metric_name,
        group_col,
        group_name,
        n,
        figsize=figsize,
        font_scale=font_scale,
        dpi=dpi,
        title=title,
    )
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
