"""Compute-efficiency figure: three stacked panels over allocated CPU cores.

All three share the compute axis and the same arms, so they read as one
column:

1. best validation score (RMSE for U1/U2, lower is better; ROC-AUC for U3),
2. configurations explored within the budget,
3. CPU utilisation, ``total_cpu_s / (ncpus * elapsed_s)`` from sacct.

Panel 3 takes its utilisation from the same summary CSV as the other two, so
every arm of the sweep appears.

TPOT points recovered from a crashed run's log have no exact configuration
count, only the log's upper bound: they are drawn with open markers and kept
out of the median line and band.

Usage: python -m benchmarking.plot_compute_efficiency          # u1
       python -m benchmarking.plot_compute_efficiency --usecase u3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from benchmarking.common import B4_METHODS, B4_USECASES, DATA_DIR, FINAL_DIR
from benchmarking.plotting import (
    ARM_COLORS,
    ARM_LABELS,
    ARM_LINESTYLES,
    ARM_MARKERS,
    apply_style,
    draw_band,
    ordered_methods,
    save_figure,
)

METRIC_LABELS = {
    "rmse": "Best validation RMSE (↓)",
    "roc_auc": "Best validation ROC-AUC (↑)",
}
UTILISATION_LABEL = "CPU utilisation (%)"
CONFIGS_LABEL = "# configurations explored"
SACCT_COLUMNS = ["total_cpu_s", "ncpus", "elapsed_s"]


def with_utilisation(summary: pd.DataFrame) -> pd.DataFrame:
    """Add the sacct CPU utilisation, NaN where the job accounting is missing."""
    usable = summary[SACCT_COLUMNS].notna().all(axis=1) & (summary["elapsed_s"] > 0)
    utilisation = (
        summary["total_cpu_s"] / (summary["ncpus"] * summary["elapsed_s"]) * 100
    )
    return summary.assign(utilisation=utilisation.where(usable))


def _draw_arm(
    ax: plt.Axes, group: pd.DataFrame, column: str, method: str, label: str
) -> None:
    stats = group.groupby("cores")[column].agg(["median", "min", "max"]).reset_index()
    color = ARM_COLORS[method]
    ax.plot(
        stats["cores"],
        stats["median"],
        marker=ARM_MARKERS[method],
        color=color,
        linestyle=ARM_LINESTYLES[method],
        label=label,
    )
    draw_band(ax, stats["cores"], stats["min"], stats["max"], color)
    # A crashed run has no exact count, only the log's upper bound.
    if column == "n_configs" and "n_configs_upper_bound" in group:
        recovered = group[group["n_configs_upper_bound"].notna()]
    else:
        recovered = group.iloc[0:0]
    if not recovered.empty:
        ax.scatter(
            recovered["cores"],
            recovered["n_configs_upper_bound"],
            facecolors="white",
            edgecolors=color,
            marker=ARM_MARKERS[method],
            zorder=4,
            s=36,
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--usecase",
        choices=B4_USECASES,
        default="u1",
        help="use case to draw (default: u1)",
    )
    p.add_argument(
        "--out-name",
        help="figure basename (default: compute_efficiency)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: results/final; smoke: results/data)",
    )
    args = p.parse_args()
    benchmark = "b4_smoke" if args.smoke else "b4"

    default_dir = DATA_DIR if args.smoke else FINAL_DIR
    out_dir = Path(args.out_dir) if args.out_dir else default_dir
    default_name = (
        f"{benchmark}_compute_efficiency" if args.smoke else "compute_efficiency"
    )
    out_name = args.out_name or default_name
    summary = pd.read_csv(DATA_DIR / f"{benchmark}_summary.csv")

    swept = summary[summary["usecase"] == args.usecase]
    if swept.empty:
        raise SystemExit(f"No {benchmark} rows for use case {args.usecase}.")
    missing = sorted(set(B4_METHODS) - set(swept["method"]))
    if missing:
        print(f"[warn] no rows yet for {missing}; drawing the arms that exist")
    swept = with_utilisation(swept)

    counts = swept.groupby(["method", "cores"]).size()
    for key, n in counts.items():
        if n < counts.max():
            print(f"[warn] {key} has only {n} seed(s)")

    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(5.4, 8.2), sharex=True)
    metric = swept["metric"].iloc[0]
    panels = [
        ("best_val", METRIC_LABELS[metric]),
        ("n_configs", CONFIGS_LABEL),
        ("utilisation", UTILISATION_LABEL),
    ]

    for method in ordered_methods(swept["method"]):
        group = swept[swept["method"] == method]
        label = ARM_LABELS.get(method, method)
        if group["utilisation"].isna().all():
            print(f"[warn] {method} has no job accounting; no utilisation panel arm")
        for ax, (column, _) in zip(axes, panels):
            _draw_arm(ax, group, column, method, label)

    for ax, (_, ylabel) in zip(axes, panels):
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    axes[2].set_ylim(0, 105)

    cores = sorted(swept["cores"].dropna().unique())
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(cores)
    axes[2].set_xticklabels([int(c) for c in cores])
    axes[2].xaxis.set_minor_locator(plt.NullLocator())
    axes[2].set_xlabel("Allocated CPU cores")
    # The score panel falls left to right, so its top-right corner is free.
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")

    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=1.4)
    save_figure(fig, out_dir, out_name)


if __name__ == "__main__":
    main()
