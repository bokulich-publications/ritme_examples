"""Shared plotting style for benchmark figures.

Follows the conventions of `src/evaluate_trials.py` (whitegrid seaborn,
tableau-colorblind10, DejaVu Sans) at the 600 dpi and embedded-TrueType
settings the manuscript figures in `final_figures/` use.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

FIG_DPI = 600
BAND_ALPHA = 0.2
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
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["pdf.fonttype"] = 42
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)


def draw_band(ax: plt.Axes, x, low, high, color: str) -> None:
    """Translucent min-max band across seeds."""
    ax.fill_between(x, low, high, color=color, alpha=BAND_ALPHA, linewidth=0)


@contextlib.contextmanager
def _opaque_bands(fig: plt.Figure):
    """Flatten every translucent band onto white for the duration of a save.

    EPS has no alpha channel, so a band saved as-is comes out fully opaque
    in whatever colour it was given and swamps the panel. Pre-blending it
    with the white background reproduces how it looks over the background in
    the PDF and keeps the file vector; where two bands overlap the one drawn
    last wins, which the median lines on top still disambiguate.
    """
    saved = []
    for ax in fig.axes:
        for band in ax.collections:
            alpha = band.get_alpha()
            if alpha is None or alpha >= 1:
                continue
            saved.append((band, alpha, band.get_facecolor()))
            band.set_facecolor(
                [
                    [alpha * channel + (1 - alpha) for channel in rgba[:3]] + [1.0]
                    for rgba in band.get_facecolor()
                ]
            )
            band.set_alpha(1.0)
    try:
        yield
    finally:
        for band, alpha, facecolor in saved:
            band.set_facecolor(facecolor)
            band.set_alpha(alpha)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight", dpi=FIG_DPI)
    with _opaque_bands(fig):
        fig.savefig(out_dir / f"{stem}.eps", bbox_inches="tight", dpi=FIG_DPI)
    print(f"Wrote {out_dir / stem}.eps/.pdf/.png")
