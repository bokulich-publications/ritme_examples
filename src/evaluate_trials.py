import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import seaborn as sns

###############################
# Global plotting parameters  #
###############################
# Single source of truth for figure size and font scaling; all plotting
# functions will use these unless explicitly overridden via arguments.
GLOBAL_FIGSIZE = (9, 4)  # (width, height) in inches
GLOBAL_DPI = 400
GLOBAL_FONT_SCALE = 1.6

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
        edgecolor="black",
        linewidth=0.05,
    )

    ax.set_xlabel("Number of Features", labelpad=10)
    ax.set_ylabel(metric_name, labelpad=10)
    ax.set_title(f"Performance vs. Model Complexity: Top {n} trials", pad=15)
    # Place legend outside the plot at the bottom-right
    ax.legend(
        title=group_name,
        loc="lower left",
        bbox_to_anchor=(1.02, 0),
        borderaxespad=0.0,
        # frameon=False,
    )

    # sns.despine(ax=ax)
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
):
    """
    Plot raw points and a rolling-mean trend of y_col over time_col.
      df:         DataFrame with your data
      y_col:      name of the metric column (e.g. "metrics.rmse_val")
      time_col:   name of the datetime column (default "start_time")
      window:     rolling window size
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], format="ISO8601")
    df = df.sort_values(time_col)
    df["smoothed"] = df[y_col].rolling(window=window, center=True, min_periods=1).mean()

    _set_seaborn_context(font_scale)
    plt.figure(figsize=figsize or GLOBAL_FIGSIZE, dpi=dpi or GLOBAL_DPI)
    plt.scatter(df[time_col], df[y_col], color=raw_color, alpha=raw_alpha, label="Raw")
    plt.plot(
        df[time_col],
        df["smoothed"],
        color=trend_color,
        linewidth=2,
        label=f"Rolling mean (w={window})",
    )

    plt.xlabel(time_col)
    plt.ylabel(y_col)
    plt.title(f"{title_prefix} - {y_col} trend over {time_col}")
    plt.xticks(rotation=45)
    plt.legend()
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

    Returns
    -------
    fig, axes : matplotlib Figure and list of Axes
    """
    if not group_specs:
        raise ValueError(
            "group_specs must contain at least one (group_col, group_name) pair"
        )

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
    # (Simpler: independent palette for each group column.)
    palette_cache = {}

    for ax, (gcol, glabel) in zip(axes, group_specs):
        if gcol not in trials.columns:
            ax.text(0.5, 0.5, f"Missing column: {gcol}", ha="center", va="center")
            ax.set_axis_off()
            continue
        subset = trials[[gcol, metric_col]].dropna()
        if subset.empty:
            ax.text(0.5, 0.5, f"No data for {glabel}", ha="center", va="center")
            ax.set_axis_off()
            continue

        # Color map: reuse if already computed for this column
        if gcol not in palette_cache:
            _, palette_cache[gcol] = create_color_map(subset, gcol)
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
    return fig, axes
