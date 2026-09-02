"""Shared plotting style for benchmark figures.

Follows the conventions of `src/evaluate_trials.py` (whitegrid seaborn,
tableau-colorblind10, DejaVu Sans, 400 dpi).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

FIG_DPI = 400
ARM_COLORS = {
    # ritme is always orange, and TPE is ritme's own sampler; the random
    # baseline and the comparators take the blues, mAML grey.
    "tpe": "#FF800E",
    "random": "#006BA4",
    "ritme": "#FF800E",
    # A second ritme build keeps the orange family, one shade darker.
    "ritme_efficient": "#C85200",
    "automl": "#006BA4",
    "tpot": "#5F9ED1",
    "maml": "#08306B",
}
# Draw (and therefore legend) order wherever methods share an axis: ritme
# first, then the comparators. B1's TPE/random arms are both ritme and keep
# their own pairing.
METHOD_ORDER = ["ritme", "ritme_efficient", "automl", "tpot", "maml"]


def ordered_methods(present) -> list:
    """`METHOD_ORDER` filtered to the methods actually present."""
    present = set(present)
    return [m for m in METHOD_ORDER if m in present]


ARM_LABELS = {
    "tpe": "TPE",
    "random": "Random",
    "ritme": "ritme",
    "ritme_efficient": "ritme_efficient",
    "automl": "auto-sklearn",
    "tpot": "TPOT",
    "maml": "mAML",
}
# Arms coincide exactly while TPE is still in its random warm-up (same seed,
# same draws), so the lines must stay distinguishable where they overlap.
ARM_MARKERS = {
    "ritme": "o",
    "ritme_efficient": "v",
    "automl": "s",
    "tpot": "^",
    "maml": "D",
}
ARM_LINESTYLES = {
    "tpe": "-",
    "random": "--",
    "ritme": "-",
    "ritme_efficient": "-",
    "automl": "--",
    "tpot": "-.",
    "maml": ":",
}


def apply_style() -> None:
    plt.style.use("tableau-colorblind10")
    plt.rcParams["font.family"] = "DejaVu Sans"
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight", dpi=FIG_DPI)
    print(f"Wrote {out_dir / stem}.pdf/.png")
