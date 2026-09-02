"""B1 figure (B1_final): best running validation RMSE over wall-clock time.

One line per arm (median across seeds, min-max band) plus a vertical
marker at the end of TPE's adaptive random warm-up.

Usage: python -m benchmarking.plot_b1_final [--smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

from benchmarking.common import DATA_DIR, FINAL_DIR, RUNS_DIR
from benchmarking.plotting import (
    ARM_COLORS,
    ARM_LABELS,
    ARM_LINESTYLES,
    apply_style,
    save_figure,
)

GRID_STEP_S = 60


def step_on_grid(run: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Evaluate a run's running-min step function on a common time grid."""
    idx = np.searchsorted(run["trial_end_s"].to_numpy(), grid, side="right") - 1
    values = run["running_min"].to_numpy()
    return np.where(idx >= 0, values[np.clip(idx, 0, None)], np.nan)


def warmup_end_s(trials: pd.DataFrame, n_startup: int) -> float | None:
    """Median (across TPE seeds) time at which the n-th trial finished.

    Optuna switches from random sampling to TPE once ``n_startup_trials``
    trials have completed, so this is where the arms stop coinciding. The
    count here is over scored trials only; trials that errored are not in
    `mlflow_logs.csv` with a metric, which would push the true switch
    slightly later.
    """
    ends = []
    for _, run in trials[trials["sampler"] == "tpe"].groupby("seed"):
        run = run.sort_values("trial_end_s")
        if len(run) >= n_startup:
            ends.append(run["trial_end_s"].iloc[n_startup - 1])
    return float(np.median(ends)) if ends else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--out-name",
        default=None,
        help="figure basename (default: b1_final; smoke: b1_smoke_search_efficiency)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: results/final; smoke: results/data)",
    )
    args = p.parse_args()
    benchmark = "b1_smoke" if args.smoke else "b1"

    default_dir = DATA_DIR if args.smoke else FINAL_DIR
    out_dir = Path(args.out_dir) if args.out_dir else default_dir
    default_name = f"{benchmark}_search_efficiency" if args.smoke else "b1_final"
    out_name = args.out_name or default_name
    trials = pd.read_csv(DATA_DIR / f"{benchmark}_trials.csv")
    warmup_file = RUNS_DIR / ("b1_smoke" if args.smoke else "b1") / "warmup.json"

    grid = np.arange(0, trials["trial_end_s"].max() + GRID_STEP_S, GRID_STEP_S)
    grid_h = grid / 3600

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    # TPE first so the dashed random arm is drawn on top: the two coincide
    # exactly until TPE leaves its warm-up, and the overlap must stay visible.
    arms = sorted(trials.groupby("sampler"), key=lambda kv: kv[0] != "tpe")
    for sampler, arm in arms:
        curves = np.vstack(
            [
                step_on_grid(run.sort_values("trial_end_s"), grid)
                for _, run in arm.groupby("seed")
            ]
        )
        # Grid points before the first trial of any seed completed have no
        # value yet; drop them so the curve starts at the first result.
        scored = ~np.all(np.isnan(curves), axis=0)
        color = ARM_COLORS[sampler]
        ax.plot(
            grid_h[scored],
            np.nanmedian(curves[:, scored], axis=0),
            color=color,
            linestyle=ARM_LINESTYLES[sampler],
            label=ARM_LABELS[sampler],
        )
        ax.fill_between(
            grid_h[scored],
            np.nanmin(curves[:, scored], axis=0),
            np.nanmax(curves[:, scored], axis=0),
            color=color,
            alpha=0.2,
            linewidth=0,
        )

    ax.set_xlabel("Wall-clock time since first trial start (h)")
    ax.set_ylabel("Best running\nvalidation RMSE (\u2193)")
    legend = ax.legend(title="Sampler", frameon=False)

    if warmup_file.exists():
        n_startup = json.loads(warmup_file.read_text())["n_startup_trials"]
        marker_s = warmup_end_s(trials, n_startup)
        if marker_s is not None:
            ax.axvline(marker_s / 3600, color="dimgray", linestyle="--", linewidth=1)
            # The note is centred on the legend's vertical midpoint.
            fig.canvas.draw()
            legend_box = legend.get_window_extent().transformed(
                ax.transAxes.inverted()
            )
            legend_mid = (legend_box.y0 + legend_box.y1) / 2
            ax.text(
                marker_s / 3600,
                legend_mid,
                " random warm-up ends",
                ha="left",
                va="center",
                fontsize=9,
                color="dimgray",
                transform=mtransforms.blended_transform_factory(
                    ax.transData, ax.transAxes
                ),
            )
    else:
        print(f"[warn] {warmup_file} missing; figure has no warm-up marker")

    save_figure(fig, out_dir, out_name)


if __name__ == "__main__":
    main()
