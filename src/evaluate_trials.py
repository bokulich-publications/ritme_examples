import hashlib
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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


# ---------- helpers (no docstrings) ----------
def _require_columns(df: pd.DataFrame, required: set[str], fname: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {fname}: {missing}")


def _pretty_y_label(y_col: str) -> str:
    return "RMSE Validation" if y_col == "metrics.rmse_val" else y_col


def _parse_time_and_sort(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], format="ISO8601", errors="coerce")
    return d.sort_values(time_col)


def _rolling_mean(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window=window, center=True, min_periods=1).mean()


def _per_search_rolling_mean(
    d: pd.DataFrame, y_col: str, tag_col: str | None, window: int
) -> pd.Series:
    """Rolling mean restricted to within each search (``tag_col`` group).

    When trials from independent searches are concatenated, a global rolling
    mean smears their distinct trajectories together. Computing the rolling
    mean per group keeps each search's trend line honest, and inserting NaN
    sentinels at group boundaries makes ``ax.plot`` automatically break the
    line so it doesn't draw a misleading bridge across searches.
    """
    if tag_col is None or tag_col not in d.columns:
        return _rolling_mean(d[y_col], window)
    smoothed = d.groupby(tag_col, sort=False, group_keys=False)[y_col].transform(
        lambda s: s.rolling(window=window, center=True, min_periods=1).mean()
    )
    # Insert NaN between consecutive trials whose tag changed, so matplotlib
    # breaks the line at each search boundary.
    if d[tag_col].nunique() > 1:
        tag_arr = d[tag_col].to_numpy()
        boundary_mask = np.zeros(len(d), dtype=bool)
        boundary_mask[1:] = tag_arr[1:] != tag_arr[:-1]
        smoothed = smoothed.where(~boundary_mask, np.nan)
    return smoothed


def _annotate_search_boundaries(
    ax: plt.Axes,
    d: pd.DataFrame,
    tag_col: str | None,
    x_col: str,
    *,
    line_color: str = "0.55",
    label_color: str = "0.35",
    label_fontsize: float = 7.5,
) -> None:
    """Draw thin dashed dividers at search boundaries and label each segment.

    Each segment of consecutive trials sharing the same ``tag_col`` value
    gets one small annotation centered horizontally above its bars, near
    the top of the axes. Boundaries get a light dashed vertical line so the
    visual break in the trend is unambiguous.
    """
    if tag_col is None or tag_col not in d.columns or d[tag_col].nunique() <= 1:
        return
    tag_arr = d[tag_col].to_numpy()
    x_arr = d[x_col].to_numpy()
    # Find contiguous runs of identical tags.
    starts = [0]
    for i in range(1, len(tag_arr)):
        if tag_arr[i] != tag_arr[i - 1]:
            starts.append(i)
    starts.append(len(tag_arr))
    for run_start, run_end in zip(starts[:-1], starts[1:]):
        # Vertical divider at the start of every segment except the first.
        if run_start > 0:
            x_div = (x_arr[run_start - 1] + x_arr[run_start]) / 2.0
            ax.axvline(
                x_div,
                linestyle=(0, (3, 3)),
                linewidth=0.8,
                color=line_color,
                alpha=0.7,
                zorder=2.5,
            )
        # Centered tag label near the top of the axes for this segment.
        x_mid = (x_arr[run_start] + x_arr[run_end - 1]) / 2.0
        ax.annotate(
            str(tag_arr[run_start]),
            xy=(x_mid, 0.96),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=label_fontsize,
            color=label_color,
            alpha=0.85,
        )


def _make_right_panel_figure(figsize, dpi):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 0.35], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    right_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.2)
    legend_ax = fig.add_subplot(right_gs[0, 0])
    legend_ax.axis("off")
    cbar_ax = fig.add_subplot(right_gs[1, 0])
    cbar_ax.set_visible(False)
    return fig, ax, legend_ax, cbar_ax


def _configure_x_ticks(ax: plt.Axes, k: int):
    if k > 40:
        step = max(1, k // 20)
        ax.set_xticks(list(range(1, k + 1, step)))
    else:
        ax.set_xticks(list(range(1, k + 1)))
    ax.set_xticklabels([])
    ax.set_xlim(0.5, k + 0.5)


def _legend_trend_patches(
    legend_ax: plt.Axes,
    trend_handle,
    ordered_labels: list,
    label_to_color: dict,
    title: str,
):
    bin_handles = [
        plt.matplotlib.patches.Patch(color=label_to_color[lbl], label=str(lbl))
        for lbl in ordered_labels
    ]
    handles = [trend_handle] + bin_handles
    if handles:
        legend_ax.legend(
            handles=handles,
            title=title,
            loc="center left",
            borderaxespad=0.0,
            frameon=True,
            fontsize=9,
            title_fontsize=9,
            labelspacing=0.4,
            handlelength=1.2,
        )


def _ensure_group_nan_label(d: pd.DataFrame, group_col: str, label: str = "nan"):
    d[group_col] = d[group_col].astype(object)
    d.loc[d[group_col].isna(), group_col] = label


def _legend_title_from_group(group_col: str, suffix: str) -> str:
    return f"Trend & {group_col.replace('params.', '').replace('_', ' ')} {suffix}"


def create_color_map(df, column, cmap_name="Set3"):
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
    palette = sns.color_palette(palette_name, n_colors)

    # Categories present in this dataframe (ignore NaN)
    uniques = [u for u in pd.unique(df[column]) if pd.notna(u)]

    # Always reserve a special lightgray color for missing category labels
    # (either actual NaN values or the explicit string label "nan").
    LIGHTGRAY = mpl.colors.to_rgb("lightgray")
    NAN_STR_KEY = "nan"

    # Start from preset mapping if defined for this column; this guarantees the
    # same category->color assignment across independent runs.
    if column in PRESET_COLOR_ORDERS:
        preset_order = PRESET_COLOR_ORDERS[column]
        existing: dict = {
            cat: palette[i % n_colors] for i, cat in enumerate(preset_order)
        }
    else:
        existing = GLOBAL_COLOR_REGISTRY.get(column, {}).copy()

    # Ensure the explicit "nan" string category is always mapped to lightgray.
    # We'll also use this color for true NaN values at return time.
    existing[NAN_STR_KEY] = LIGHTGRAY

    # Track used palette indices from existing mapping (if any)
    used_indices = set()
    if any(
        (color not in palette) for color in existing.values() if color is not LIGHTGRAY
    ):
        # Enforce requested palette strictly if previous colors came from a
        # different palette
        # Preserve the lightgray mapping for the special "nan" label.
        existing = {
            k: v
            for k, v in existing.items()
            if (v in palette) or (str(k).lower() == NAN_STR_KEY)
        }
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
        # Skip if already assigned (including the explicit "nan" string label)
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
    # Assign lightgray to actual NaN entries so they are consistently shown
    # the same as the explicit "nan" string label.
    if colors.isna().any():
        colors = colors.where(~df[column].isna(), LIGHTGRAY)
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
    """Plot model complexity vs. a metric, with optional log scale on X.

    Parameters allow optional override of global figure settings.
    """
    _set_seaborn_context(font_scale)
    d = trials[["metrics.nb_features", metric_col, group_col]].copy()
    _, color_map = get_consistent_color_map(d, group_col)

    fig, ax = plt.subplots(figsize=figsize or GLOBAL_FIGSIZE, dpi=dpi or GLOBAL_DPI)
    sns.scatterplot(
        x="metrics.nb_features",
        y=metric_col,
        hue=group_col,
        palette=color_map,
        data=d,
        s=50,
        ax=ax,
        edgecolor="grey",
        linewidth=0.02,
    )
    # Optionally apply log scale on X (requires positive values)
    x_label = "Number of Features"
    if x_log_scale:
        x_vals = d["metrics.nb_features"].dropna().values
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
    # Reorder legend to follow the consistent color map order but only include
    # categories present in the filtered data
    present = set(pd.unique(d[group_col]))
    ordered_labels = [c for c in color_map.keys() if c in present]
    handles, labels = ax.get_legend_handles_labels()
    # Build mapping from label to handle
    hmap = {lbl: h for h, lbl in zip(handles, labels) if lbl in ordered_labels}
    ordered_handles = [hmap[lbl] for lbl in ordered_labels if lbl in hmap]
    if ordered_handles:
        ax.legend(
            ordered_handles,
            ordered_labels,
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
    """Plot raw points and a rolling-mean trend of a metric over time.

    The x-axis shows the trial index after time sorting. Optionally shows Y on
    a log scale.
    """
    df = _parse_time_and_sort(df[[y_col, time_col]], time_col)
    if first_n is not None and first_n > 0:
        df = df.head(first_n)
    # Index 1..N on x-axis
    df["trial_index"] = range(1, len(df) + 1)
    df["smoothed"] = _rolling_mean(df[y_col], window)

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
    tag_col: str = "tags.experiment_tag",
    models: list | None = None,
    window: int = 20,
    title_prefix: str = "Model: ",
    figsize: tuple | None = None,
    raw_alpha: float = 0.35,
    trend_palette: str = "colorblind",
    std_alpha: float = 0.15,
    font_scale: float | None = None,
    dpi: int | None = None,
    first_n: int | None = None,
    y_log_scale: bool = False,
):
    """Plot rolling-mean trends over time for multiple models in stacked subplots.

    Each model panel overlays one rolling-mean curve per ``tag_col`` value
    (defaulting to ``tags.experiment_tag``), so independent search runs of
    the same model are drawn as separate lines aligned at trial 0 of their
    own search. Without this split, concatenating two TPE searches via
    global wall-clock time would produce a misleading
    "convergence-then-regression" artifact at the experiment boundary —
    each fresh search restarts the sampler's exploration phase. If
    ``tag_col`` is missing from ``df`` (or contains a single value) the
    panel collapses to one curve, matching the prior behaviour.
    """
    required_cols = {group_col, y_col, time_col}
    _require_columns(df, required_cols, "plot_trend_over_time_multi_models")
    has_tag = tag_col in df.columns

    _set_seaborn_context(font_scale)

    if models is None:
        models = list(pd.unique(df[group_col]))
    models = [m for m in models if pd.notna(m)]

    # Build per-(model, tag) time-sorted series with trial_index local to
    # each search. `per_model[model]` is a list of (tag, d) pairs.
    cols_to_keep = [y_col, time_col] + ([tag_col] if has_tag else [])
    per_model: dict = {}
    for model in models:
        sub = df[df[group_col] == model][cols_to_keep]
        tags = list(pd.unique(sub[tag_col].dropna())) if has_tag else ["all"]
        groups = []
        for tag in tags:
            tdf = sub[sub[tag_col] == tag] if has_tag else sub
            d = _parse_time_and_sort(tdf[[y_col, time_col]], time_col)
            if first_n is not None and first_n > 0:
                d = d.head(first_n)
            d = d[[y_col]].copy()
            d["trial_index"] = range(1, len(d) + 1)
            roll = d[y_col].rolling(window=window, center=True, min_periods=1)
            d["smoothed"] = roll.mean()
            d["smoothed_std"] = roll.std(ddof=0)
            groups.append((tag, d))
        per_model[model] = groups

    if figsize is None:
        figsize = (GLOBAL_FIGSIZE[0], GLOBAL_FIGSIZE[1] * len(models))

    fig, axes = plt.subplots(
        nrows=len(models),
        ncols=1,
        sharex=False,
        figsize=figsize,
        dpi=dpi or GLOBAL_DPI,
    )
    if len(models) == 1:
        axes = [axes]

    y_label = "RMSE Validation" if y_col == "metrics.rmse_val" else y_col

    for ax, model in zip(axes, models):
        groups = per_model[model]
        palette = sns.color_palette(trend_palette, n_colors=max(len(groups), 3))
        any_nonpos = False
        for color, (tag, d) in zip(palette, groups):
            ax.scatter(
                d["trial_index"],
                d[y_col],
                color=color,
                alpha=raw_alpha,
                s=10,
            )
            ax.plot(
                d["trial_index"],
                d["smoothed"],
                color=color,
                linewidth=2,
                label=f"{tag} (n={len(d)})",
            )
            lower = d["smoothed"] - d["smoothed_std"]
            upper = d["smoothed"] + d["smoothed_std"]
            ax.fill_between(
                d["trial_index"],
                lower,
                upper,
                color=color,
                alpha=std_alpha,
                linewidth=0,
            )
            if (
                (d[y_col] <= 0).any()
                or (d["smoothed"] <= 0).any()
                or (d["smoothed_std"] <= 0).any()
            ):
                any_nonpos = True

        ax.set_title(f"{title_prefix}{model}")
        ax.grid(True)
        ax.set_ylabel(y_label if ax is axes[-1] else "")
        if y_log_scale and not any_nonpos:
            ax.set_yscale("log")
            if ax is axes[-1]:
                ax.set_ylabel(y_label + " (log scale)")
        if groups:
            ax.legend(loc="best", fontsize="x-small")

    for ax in axes:
        ax.set_xlabel("Trial within search" if ax is axes[-1] else "")

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
    """Plot metric distribution per group using a horizontal boxplot."""
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
    """Create stacked horizontal boxplots by groups; annotate Kruskal–Wallis
    p-values."""
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
                    p_value = math.ulp(0.0)
            if p_value is not None:
                if p_value == math.ulp(0.0):
                    label = f"p<{p_value:.3g}"
                else:
                    label = f"p={p_value:.3g}"
                ax.text(0.98, 0.98, label, transform=ax.transAxes, ha="right", va="top")
            p_values.append((glabel, p_value))
        except Exception:
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
            if p == math.ulp(0.0):
                print(f"  {glabel}: p<{p:.3g}")
            else:
                print(f"  {glabel}: p={p:.3g}")
    return fig, axes


def plot_recent_param_cat_over_time(
    df: pd.DataFrame,
    y_col: str = "metrics.rmse_val",
    group_col: str = "params.data_selection",
    time_col: str = "start_time",
    tag_col: str | None = "tags.experiment_tag",
    n_last: int = 200,
    title: str | None = None,
    figsize: tuple | None = None,
    font_scale: float | None = None,
    dpi: int | None = None,
    y_log_scale: bool = False,
    window: int = 20,
):
    """Bar chart of the last N trials colored by a group column with a trend line.

    When ``tag_col`` is present and the last N trials span more than one
    search, the rolling-mean trend is computed per-search and breaks at
    search boundaries, with a thin dashed divider and small label per
    segment. Without that split, two independent searches are smeared
    together and the trend line invents a "convergence then regression"
    artifact at the boundary.
    """
    cols = [y_col, group_col, time_col]
    if tag_col and tag_col in df.columns:
        cols.append(tag_col)
    _require_columns(
        df, {y_col, group_col, time_col}, "plot_recent_param_cat_over_time"
    )

    # Work on a copy; parse time and sort
    d = _parse_time_and_sort(df[cols], time_col)
    d = d.dropna(subset=[time_col])

    # Keep last N trials, drop rows without metric
    if n_last is not None and n_last > 0:
        d = d.tail(n_last)
    d = d.dropna(subset=[y_col])

    if d.empty:
        raise ValueError("No data available to plot after filtering and dropping NaNs.")

    # Map NaN groups to a visible label for legend consistency
    _ensure_group_nan_label(d, group_col, "nan")

    # Create sequential x positions (1..k) in chronological order
    d["trial_index"] = range(1, len(d) + 1)

    # We keep positions as 1..k; x tick labels are hidden for clarity.

    # Use consistent color mapping across figures for categories in group_col
    _, color_map = get_consistent_color_map(d, group_col)
    bar_colors = d[group_col].map(color_map)

    # Set plotting context
    _set_seaborn_context(font_scale)

    fig, ax, legend_ax, _ = _make_right_panel_figure(
        figsize=figsize or (max(GLOBAL_FIGSIZE[0], 10), GLOBAL_FIGSIZE[1]),
        dpi=dpi or GLOBAL_DPI,
    )

    # Bars and per-search trend line (NaN gap at search boundaries)
    positions = list(d["trial_index"].values)
    ax.bar(positions, d[y_col], color=bar_colors, width=0.9, edgecolor="none")
    d["smoothed"] = _per_search_rolling_mean(d, y_col, tag_col, window)
    (trend_line,) = ax.plot(
        positions,
        d["smoothed"],
        color="black",
        linewidth=1,
        label=f"Rolling mean (w={window})",
    )

    # Axis labels and optional title
    y_label = _pretty_y_label(y_col)
    ax.set_ylabel(y_label)
    ax.set_xlabel(f"Last {n_last} trials by time →")
    if title:
        ax.set_title(title)

    # Optional log scale on y
    if y_log_scale and not (d[y_col] <= 0).any():
        ax.set_yscale("log")

    # Search-boundary dividers + per-segment tag labels
    _annotate_search_boundaries(ax, d, tag_col, "trial_index")

    # Legend in right panel: trend + categories (ordered by color map keys)
    present_cats = set(pd.unique(d[group_col]))
    ordered_present = [c for c in color_map.keys() if c in present_cats]
    label_to_color = {lbl: color_map[lbl] for lbl in ordered_present}
    _legend_trend_patches(
        legend_ax,
        trend_line,
        ordered_present,
        label_to_color,
        _legend_title_from_group(group_col, "categories"),
    )

    # X ticks and limits; hide labels for cleanliness
    _configure_x_ticks(ax, len(d))

    sns.despine(ax=ax)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_recent_param_cont_over_time(
    df: pd.DataFrame,
    param_col: str,
    group_col: str = "params.data_selection",
    time_col: str = "start_time",
    tag_col: str | None = "tags.experiment_tag",
    n_last: int = 200,
    n_bins: int = 6,
    binning: str = "quantile",  # "quantile", "uniform", or "log-uniform"
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
    """Bar chart of last N trials colored by binned ranges of a continuous parameter.

    When ``tag_col`` is present and the last N trials span more than one
    search, the rolling-mean trend is computed per-search and breaks at
    search boundaries, with a thin dashed divider and small label per
    segment — same fix as ``plot_recent_param_cat_over_time``.
    """
    required = {param_col, y_col, time_col}
    _require_columns(df, required, "plot_recent_param_cont_over_time")

    # Work on a copy; parse time and sort chronologically
    cols = [param_col, y_col, time_col]
    if tag_col and tag_col in df.columns:
        cols.append(tag_col)
    d = _parse_time_and_sort(df[cols], time_col)
    if n_last is not None and n_last > 0:
        d = d.tail(n_last)

    # Determine the bins
    p = pd.to_numeric(d[param_col], errors="coerce")
    if binning == "log-uniform":
        # For parameters spanning orders of magnitude: equally spaced bins in
        # log10 space. Note: log bins require positive values; non-positives
        # fall into a catch-all leftmost bin.
        pos = p[p > 0]
        if pos.empty:
            # No positive values; fallback to a single uniform bin to avoid
            # log errors
            binned = pd.cut(p, bins=n_bins)
        else:
            lo_exp = float(np.floor(np.log10(pos.min())))
            hi_exp = float(np.ceil(np.log10(pos.max())))
            if not np.isfinite(lo_exp) or not np.isfinite(hi_exp) or hi_exp <= lo_exp:
                # Degenerate range; fallback to uniform bins
                binned = pd.cut(p, bins=n_bins)
            else:
                edges = np.logspace(lo_exp, hi_exp, num=n_bins + 1, base=10.0)
                # Prepend -inf to capture zeros/negatives in a dedicated bin
                all_edges = np.concatenate(([-np.inf], edges))
                binned = pd.cut(p, bins=all_edges, include_lowest=True)
    elif binning == "quantile":
        binned = pd.qcut(p, q=n_bins, duplicates="drop")
    else:
        binned = pd.cut(p, bins=n_bins)

    # Create ordered categorical labels for bins and map to colors
    binned = binned.astype("category")
    categories = list(binned.cat.categories)
    # Human-friendly bin labels for legend (scientific notation for log bins;
    # handle +/-inf)

    def _fmt_bound(v: float, sci: bool = False) -> str:
        if isinstance(v, (float, int)) and not math.isfinite(v):
            return "-∞" if v < 0 else "∞"
        try:
            return f"{float(v):.1e}" if sci else f"{float(v):.3g}"
        except Exception:
            return str(v)

    use_sci = binning == "log-uniform"
    bin_labels = [
        (
            f"[{_fmt_bound(c.left, use_sci)}, {_fmt_bound(c.right, use_sci)}"
            f"{']' if c.closed in ('right', 'both') else ')'}"
            if hasattr(c, "left")
            else str(c)
        )
        for c in categories
    ]
    # Store labels as string in a new column used for color mapping; add a
    # dedicated label for missing values so we can display them as a white bin
    MISSING_BIN_LABEL = "NaN"
    has_missing = binned.isna().any()
    mapped_labels = []
    for cat in binned:
        if pd.isna(cat):
            mapped_labels.append(MISSING_BIN_LABEL)
        else:
            mapped_labels.append(bin_labels[categories.index(cat)])
    d["param_bin_label"] = mapped_labels

    # Build a sequential, low-to-high palette for the bins (ordered by value)
    # Use a perceptually uniform, colorblind-friendly colormap
    seq_palette = sns.color_palette(palette_name, n_colors=len(categories))
    # Map each ordered bin label to a corresponding color (low->high)
    label_to_color = {lbl: seq_palette[i] for i, lbl in enumerate(bin_labels)}
    # Add lightgray color for the NaN/missing bin if present
    if has_missing:
        label_to_color[MISSING_BIN_LABEL] = (0.85, 0.85, 0.85)

    # Sequential x positions (older -> newer)
    d["x_pos"] = range(1, len(d) + 1)

    _set_seaborn_context(font_scale)
    fig, ax, legend_ax, cbar_ax = _make_right_panel_figure(
        figsize=figsize or (max(GLOBAL_FIGSIZE[0], 10), GLOBAL_FIGSIZE[1]),
        dpi=dpi or GLOBAL_DPI,
    )

    # Bars: height is y_col (metric), color indicates parameter bin
    positions = list(d["x_pos"].values)
    # Map bar colors by the ordered bin labels using the sequential palette
    bar_colors = d["param_bin_label"].map(label_to_color)
    ax.bar(positions, d[y_col], color=bar_colors, width=0.9, edgecolor="none")

    # Rolling mean line over bar heights (per-search, with NaN gaps at
    # boundaries so independent searches don't share a smoothed trend).
    d["smoothed"] = _per_search_rolling_mean(d, y_col, tag_col, window)
    (trend_line,) = ax.plot(
        positions,
        d["smoothed"],
        color="black",
        linewidth=1,
        label=f"Rolling mean (w={window})",
    )

    # Axis labels and title
    y_label = _pretty_y_label(y_col)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Last N trials by time →")
    if title:
        ax.set_title(title)

    # Optional log scale with safety check
    if y_log_scale and not (d[y_col] <= 0).any():
        ax.set_yscale("log")

    # Search-boundary dividers + per-segment tag labels
    _annotate_search_boundaries(ax, d, tag_col, "x_pos")

    # Legend in the fixed right panel: parameter bin patches ordered low->high
    # plus the rolling-mean trend line. Include empty bins as well so users
    # can see the full set of intervals even if some receive zero counts.
    legend_labels = list(bin_labels)
    if has_missing:
        legend_labels.append(MISSING_BIN_LABEL)
    bin_handles = [
        plt.matplotlib.patches.Patch(color=label_to_color[lbl], label=str(lbl))
        for lbl in legend_labels
    ]
    handles = [trend_line] + bin_handles
    if handles:
        legend_ax.legend(
            handles=handles,
            title=_legend_title_from_group(param_col, "bins"),
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
            cmap = mpl.cm.get_cmap(palette_name)
            if binning == "log-uniform":
                pos = param_vals[param_vals > 0]
                if len(pos) > 0:
                    vmin = float(pos.min())
                    vmax = float(pos.max())
                    if vmin > 0 and vmax > vmin:
                        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
                    else:
                        norm = mpl.colors.Normalize(
                            vmin=float(param_vals.min()), vmax=float(param_vals.max())
                        )
                else:
                    norm = mpl.colors.Normalize(
                        vmin=float(param_vals.min()), vmax=float(param_vals.max())
                    )
            else:
                norm = mpl.colors.Normalize(
                    vmin=float(param_vals.min()), vmax=float(param_vals.max())
                )
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
    _configure_x_ticks(ax, len(d))

    sns.despine(ax=ax)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()
    return fig, ax
