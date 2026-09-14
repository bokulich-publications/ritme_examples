"""Supplementary figures: train and test performance of all three arms.

One 3x2 figure per use case. Rows are the arms (original / ritme / TPOT),
columns are the train and the held-out test split. u1 and u2 predict a
continuous target and get true-vs-predicted scatter plots; u3 and u4 classify
and get ROC curves -- binary for u3, macro one-vs-rest with the per-class
curves behind it for u4.

Reads the CSVs written by `extract_predictions.py`; run that first.

Usage (repo root, ritme_usecases env):
    python -m final_figures.supp_scatter_roc.make_supp_figures
    python -m final_figures.supp_scatter_roc.make_supp_figures --usecase u3
"""

from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.metrics import auc, r2_score, roc_auc_score, roc_curve

try:
    from final_figures.supp_scatter_roc.arms import (
        ARM_LABELS,
        ARMS,
        OUT_DIR,
        TASK,
        TITLES,
        USECASES,
        prediction_path,
    )
except ImportError:  # running from inside final_figures/supp_scatter_roc/
    from arms import (  # type: ignore[no-redef]
        ARM_LABELS,
        ARMS,
        OUT_DIR,
        TASK,
        TITLES,
        USECASES,
        prediction_path,
    )

SPLIT_COLORS = {"train": "cornflowerblue", "test": "darkorange"}
SPLIT_TITLES = {"train": "Train set", "test": "Test set"}

# Page geometry in inches. The cell size follows from the width so that every
# panel is square, and the page height from the cell size; both figure kinds
# put a 1:1 or a 0-1 axis on each side, which only reads correctly square.
PAGE_WIDTH = 8.0  # the width of figures 4 and 5, so all three typeset alike
LEFT_MARGIN = 1.30
RIGHT_MARGIN = 0.26
TOP_MARGIN = 0.72
BOTTOM_MARGIN = 0.72
COLUMN_GAP = 0.62
ROW_GAP = 0.26
CELL = (PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN - COLUMN_GAP) / 2
PAGE_HEIGHT = TOP_MARGIN + 3 * CELL + 2 * ROW_GAP + BOTTOM_MARGIN
FIG_SIZE = (PAGE_WIDTH, PAGE_HEIGHT)

ARM_LABEL_X = 0.022  # figure fraction; the bold row names, outermost
AXIS_LABEL_X = 0.085  # figure fraction; the single shared y-axis label
X_LABEL_Y = 0.022

LABEL_SIZE = 12.5  # matches GROUP_LABEL_SIZE of figure 4
TICK_SIZE = 10
ANNOT_SIZE = 9.5
MARKER_SIZE = 12
SCATTER_MARGIN = 0.04  # padding of the shared square data range
PER_CLASS_COLOR = "0.75"

TARGET_LABELS = {"u1": "age (months)", "u2": "mean temperature (°C)"}
FPR_GRID = np.linspace(0.0, 1.0, 1001)
DPI = 600


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": LABEL_SIZE,
            "axes.titlesize": LABEL_SIZE,
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
    )


def load(usecase: str) -> dict[str, pd.DataFrame]:
    out = {}
    for arm in ARMS:
        path = prediction_path(usecase, arm)
        if not path.exists():
            raise SystemExit(
                f"missing {path}; run extract_predictions.py {usecase} {arm} first"
            )
        out[arm] = pd.read_csv(path)
    return out


def class_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("p_")]


def grid(fig: plt.Figure) -> GridSpec:
    return GridSpec(
        3,
        2,
        figure=fig,
        left=LEFT_MARGIN / PAGE_WIDTH,
        right=1 - RIGHT_MARGIN / PAGE_WIDTH,
        top=1 - TOP_MARGIN / PAGE_HEIGHT,
        bottom=BOTTOM_MARGIN / PAGE_HEIGHT,
        wspace=COLUMN_GAP / CELL,
        hspace=ROW_GAP / CELL,
    )


def scatter_range(frames: dict[str, pd.DataFrame]) -> tuple[float, float]:
    """One square data range shared by all six panels, so scales are comparable."""
    values = np.concatenate(
        [df[["y_true", "y_pred"]].to_numpy().ravel() for df in frames.values()]
    )
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * SCATTER_MARGIN
    return lo - pad, hi + pad


def draw_scatter(ax: plt.Axes, part: pd.DataFrame, split: str, lims) -> None:
    color = SPLIT_COLORS[split]
    ax.plot(lims, lims, linestyle=":", color="black", linewidth=1, zorder=1)
    ax.scatter(
        part["y_true"],
        part["y_pred"],
        s=MARKER_SIZE,
        color=color,
        linewidths=0,
        zorder=2,
    )
    rmse = float(np.sqrt(np.mean((part["y_true"] - part["y_pred"]) ** 2)))
    r2 = r2_score(part["y_true"], part["y_pred"])
    ax.text(
        0.04,
        0.96,
        f"RMSE = {rmse:.2f}\nR² = {r2:.2f}\nn = {len(part)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOT_SIZE,
        color=color,
    )
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)


def draw_roc(ax: plt.Axes, part: pd.DataFrame, split: str, columns: list[str]) -> None:
    color = SPLIT_COLORS[split]
    classes = [c.removeprefix("p_") for c in columns]
    y_true = part["y_true"].astype(str).to_numpy()
    proba = part[columns].to_numpy()
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, zorder=1)

    if len(classes) == 2:
        positive = (y_true == classes[1]).astype(int)
        fpr, tpr, _ = roc_curve(positive, proba[:, 1])
        label = f"AUROC = {roc_auc_score(positive, proba[:, 1]):.3f}\n(n = {len(part)})"
        ax.plot(fpr, tpr, color=color, linewidth=2, zorder=3, label=label)
    else:
        # Macro one-vs-rest: per-class curves behind a bold average of their
        # TPRs on a common FPR grid. A class absent from the split has no
        # defined ROC and is left out of both the curve and the mean.
        tpr_sum = np.zeros_like(FPR_GRID)
        aucs = []
        for i, cls in enumerate(classes):
            binary = (y_true == cls).astype(int)
            if binary.sum() in (0, len(binary)):
                continue
            fpr, tpr, _ = roc_curve(binary, proba[:, i])
            tpr_sum += np.interp(FPR_GRID, fpr, tpr)
            aucs.append(auc(fpr, tpr))
            ax.plot(fpr, tpr, color=PER_CLASS_COLOR, linewidth=0.7, zorder=2)
        ax.plot(
            FPR_GRID,
            tpr_sum / len(aucs),
            color=color,
            linewidth=2,
            zorder=3,
            label=(
                f"macro OvR = {np.mean(aucs):.3f}\n"
                f"({len(aucs)} classes, n = {len(part)})"
            ),
        )

    legend = ax.legend(
        loc="lower right",
        fontsize=ANNOT_SIZE,
        frameon=True,
        facecolor="white",
        edgecolor="#C8C8C8",
        framealpha=1.0,
        handlelength=1.4,
        borderpad=0.4,
    )
    legend.get_frame().set_linewidth(0.8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)


def make(usecase: str) -> None:
    frames = load(usecase)
    regression = TASK[usecase] == "regression"
    lims = scatter_range(frames) if regression else None

    fig = plt.figure(figsize=FIG_SIZE)
    gs = grid(fig)
    axes = np.empty((3, 2), dtype=object)
    for row, arm in enumerate(ARMS):
        for col, split in enumerate(("train", "test")):
            ax = fig.add_subplot(gs[row, col])
            axes[row, col] = ax
            part = frames[arm][frames[arm]["split"] == split]
            if regression:
                draw_scatter(ax, part, split, lims)
            else:
                draw_roc(ax, part, split, class_columns(frames[arm]))
            if row == 0:
                ax.set_title(SPLIT_TITLES[split], pad=8)
            if row < 2:
                ax.tick_params(labelbottom=False)
            if col > 0:
                ax.tick_params(labelleft=False)

    for row, arm in enumerate(ARMS):
        box = axes[row, 0].get_position()
        fig.text(
            ARM_LABEL_X,
            (box.y0 + box.y1) / 2,
            ARM_LABELS[arm],
            rotation=90,
            ha="left",
            va="center",
            fontsize=LABEL_SIZE,
            fontweight="bold",
        )

    if regression:
        target = TARGET_LABELS[usecase]
        x_label, y_label = f"True {target}", f"Predicted {target}"
    else:
        x_label, y_label = "False positive rate", "True positive rate"

    top, bottom = axes[0, 0].get_position(), axes[2, 0].get_position()
    left, right = axes[2, 0].get_position(), axes[2, 1].get_position()
    fig.text(
        AXIS_LABEL_X,
        (bottom.y0 + top.y1) / 2,
        y_label,
        rotation=90,
        ha="left",
        va="center",
        fontsize=LABEL_SIZE,
    )
    fig.text(
        (left.x0 + right.x1) / 2,
        X_LABEL_Y,
        x_label,
        ha="center",
        va="bottom",
        fontsize=LABEL_SIZE,
    )
    fig.suptitle(TITLES[usecase], fontsize=LABEL_SIZE + 1, fontweight="bold", y=0.995)

    stem = OUT_DIR / f"supp_{usecase}_train_test"
    for suffix in ("eps", "pdf", "png"):
        fig.savefig(f"{stem}.{suffix}", dpi=DPI)
    plt.close(fig)
    print(f"written to {stem}.eps / .pdf / .png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--usecase", choices=USECASES, help="default: all four")
    args = p.parse_args()
    style()
    for usecase in [args.usecase] if args.usecase else USECASES:
        make(usecase)


if __name__ == "__main__":
    main()
