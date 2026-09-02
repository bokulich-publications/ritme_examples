"""B4 figure: compute scaling on all three use cases, 2 x 3 panels.

Columns are use cases; row 1 is configurations explored within the budget,
row 2 the best validation score (RMSE for U1/U2, lower is better; ROC-AUC for
U3, higher is better), both against allocated cores on a log-2 axis. One line
per method: median across seeds with a min-max band. TPOT points recovered
from a crashed run's log are drawn with open markers. mAML (U3 only) is a
fixed grid walked in its published order until the budget is spent, so its
count is a prefix of that grid rather than a search -- state this in the
caption.

Usage: python -m benchmarking.plot_b2_final          # u1, results/final/b2_final
       python -m benchmarking.plot_b2_final --usecases u1 u2 u3 --out-name b4_all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from benchmarking.common import (
    B4_METHODS,
    B4_USECASES,
    DATA_DIR,
    FINAL_DIR,
)
from benchmarking.plotting import (
    ARM_COLORS,
    ARM_LABELS,
    ARM_LINESTYLES,
    ARM_MARKERS,
    apply_style,
    ordered_methods,
    save_figure,
)

USECASE_TITLES = {
    "u1": "U1 - infant age (regression)",
    "u2": "U2 - ocean temperature (regression)",
    "u3": "U3 - CRC screening (classification)",
}
METRIC_LABELS = {
    "rmse": "Best validation RMSE (\u2193)",
    "roc_auc": "Best validation ROC-AUC (\u2191)",
}


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
    ax.fill_between(
        stats["cores"], stats["min"], stats["max"], color=color, alpha=0.2, linewidth=0
    )
    # A crashed run has no exact count, only the log's upper bound: drawn as
    # an open marker and kept out of the median line and band.
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
        "--usecases",
        nargs="+",
        choices=B4_USECASES,
        default=["u1"],
        help="use cases to draw (default: u1)",
    )
    p.add_argument(
        "--out-name",
        help="figure basename (default: b2_final; smoke: b4_smoke_compute_scaling)",
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
    default_name = f"{benchmark}_compute_scaling" if args.smoke else "b2_final"
    out_name = args.out_name or default_name
    summary = pd.read_csv(DATA_DIR / f"{benchmark}_summary.csv")
    swept = summary
    if args.usecases:
        swept = swept[swept["usecase"].isin(args.usecases)]
        if swept.empty:
            raise SystemExit(f"No {benchmark} rows for use cases {args.usecases}.")
        missing = sorted(set(B4_METHODS) - set(swept["method"]))
        if missing:
            print(f"[warn] no rows yet for {missing}; drawing the arms that exist")

    counts = swept.groupby(["usecase", "method", "cores"]).size()
    for key, n in counts.items():
        if n < counts.max():
            print(f"[warn] {key} has only {n} seed(s)")

    apply_style()
    usecases = [u for u in B4_USECASES if u in set(swept["usecase"])]
    fig, axes = plt.subplots(
        2,
        len(usecases),
        # A single-use-case figure needs more width than one column's share.
        figsize=(max(5.4, 4.4 * len(usecases)), 6.8),
        sharex="col",
        squeeze=False,
    )
    cores = sorted(swept["cores"].dropna().unique())

    for col, usecase in enumerate(usecases):
        sub = swept[swept["usecase"] == usecase]
        metric = sub["metric"].iloc[0]
        for method in ordered_methods(sub["method"]):
            group = sub[sub["method"] == method]
            label = ARM_LABELS.get(method, method)
            _draw_arm(axes[0, col], group, "n_configs", method, label)
            _draw_arm(axes[1, col], group, "best_val", method, label)
        axes[0, col].set_title(USECASE_TITLES.get(usecase, usecase), fontsize=10)
        axes[1, col].set_xscale("log", base=2)
        axes[1, col].set_xticks(cores)
        axes[1, col].set_xticklabels([int(c) for c in cores])
        axes[1, col].xaxis.set_minor_locator(plt.NullLocator())
        axes[1, col].set_xlabel("Allocated CPU cores")
        axes[1, col].set_ylabel(METRIC_LABELS[metric])
        axes[0, col].legend(frameon=False, fontsize=7.5, loc="upper left")
        for ax in axes[:, col]:
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    axes[0, 0].set_ylabel("# configurations explored")
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    save_figure(fig, out_dir, out_name)


if __name__ == "__main__":
    main()
