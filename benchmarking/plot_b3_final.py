"""B3 figures (B3_final): CPU utilisation over allocated compute.

Two figures, same compute axis as the B2_final figure:

- ``b3_final`` -- the ritme arm alone, from the B2 sweep's utilisation
  decomposition (results/data/b2_utilisation.csv, written by
  ``python -m benchmarking.analyze_utilization``).
- ``b3_final_comparators`` -- ritme vs the autoML comparator arms of the
  u1 sweep (results/data/b4_summary.csv, written by
  ``python -m benchmarking.collect_comparators``), with utilisation computed from
  the sacct fields as total_cpu_s / (ncpus * elapsed_s). mAML has no u1
  sweep (u3-only reference), so it cannot appear here.

Each arm is the median across seeds with a min-max band, per core level.
Both figures are written to results/final/.

Usage: python -m benchmarking.plot_b3_final
"""

import matplotlib.pyplot as plt
import pandas as pd

from .common import DATA_DIR, FINAL_DIR
from .plotting import (
    ARM_COLORS,
    ARM_LABELS,
    ARM_LINESTYLES,
    ARM_MARKERS,
    apply_style,
    ordered_methods,
    save_figure,
)


def _draw_arm(ax: plt.Axes, sub: pd.DataFrame, method: str) -> None:
    stats = sub.groupby("cores")["utilisation"].agg(["median", "min", "max"])
    ax.plot(
        stats.index,
        stats["median"],
        marker=ARM_MARKERS[method],
        linestyle=ARM_LINESTYLES[method],
        color=ARM_COLORS[method],
        label=ARM_LABELS.get(method, method),
    )
    ax.fill_between(
        stats.index, stats["min"], stats["max"], color=ARM_COLORS[method], alpha=0.2
    )


def _finish(ax: plt.Axes, cores: list) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(cores))
    ax.set_xticklabels([str(c) for c in sorted(cores)])
    ax.set_ylim(0, 105)
    ax.set_xlabel("Allocated CPU cores")
    ax.set_ylabel("CPU utilisation (%)")
    ax.legend(frameon=False)


def plot_ritme_only() -> None:
    df = pd.read_csv(DATA_DIR / "b2_utilisation.csv")
    if "method" in df.columns:
        df = df[df["method"] == "ritme"]
    if df.empty:
        raise SystemExit("no ritme rows in b2_utilisation.csv")
    df = df.assign(utilisation=df["utilisation"] * 100)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    _draw_arm(ax, df, "ritme")
    _finish(ax, df["cores"].unique().tolist())
    save_figure(fig, FINAL_DIR, "b3_final")


def plot_comparators() -> None:
    path = DATA_DIR / "b4_summary.csv"
    if not path.exists():
        print(f"[warn] {path} missing; skipping b3_final_comparators")
        return
    df = pd.read_csv(path)
    if "usecase" in df.columns:
        df = df[df["usecase"] == "u1"]
    df = df.dropna(subset=["total_cpu_s", "elapsed_s", "ncpus"])
    df = df[df["elapsed_s"] > 0]
    if df.empty:
        print("[warn] no u1 rows with sacct stats; skipping b3_final_comparators")
        return
    df = df.assign(
        utilisation=df["total_cpu_s"] / (df["ncpus"] * df["elapsed_s"]) * 100
    )

    methods = ordered_methods(df["method"].unique())
    missing = [m for m in ("ritme", "automl", "tpot") if m not in methods]
    if missing:
        print(f"[warn] no u1 rows yet for {missing}; drawing the arms that exist")
    if len(methods) < 2:
        print("[warn] a single arm is no comparison; skipping b3_final_comparators")
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for method in methods:
        _draw_arm(ax, df[df["method"] == method], method)
    _finish(ax, df["cores"].unique().tolist())
    save_figure(fig, FINAL_DIR, "b3_final_comparators")


def main() -> None:
    apply_style()
    plot_ritme_only()
    plot_comparators()


if __name__ == "__main__":
    main()
