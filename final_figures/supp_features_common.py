"""Shared pieces of the supplementary feature figures and the named stability plots.

Feature naming turns ritme's design-matrix column names into readable labels:
OTU / ASV ids into their deepest classified taxon, aggregated taxa into
"name (rank)", transformed columns into "label [transform]", and ilr balances
into the taxon that defines them. With skbio's default (Gram-Schmidt) ilr
basis, balance k contrasts the first k + 1 retained features against feature
k + 2, so the balance is named after that one feature; the retained order is
read from the deployed model.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import pandas as pd

from src.launch_models import USECASES

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "use_cases" / "ritme_runs" / "local"
OUT_DIR = REPO_ROOT / "final_figures"

COLORS = {  # Set3, as in figures 4-5; orange = metadata involved, blue = not
    "with": "#FDB462",
    "without": "#80B1D3",
    "metadata": "#FDB462",
    "microbial": "#80B1D3",
    "original": "#BEBADA",
    "metadata_only": "#D9D9D9",
}
METADATA_HATCH = "////"
BAR_EDGE = "black"
LABEL_SIZE = 12.5  # GROUP_LABEL_SIZE of figure 4
TICK_SIZE = 10
NOTE_SIZE = 9
LEGEND_KW = dict(frameon=True, edgecolor="#C8C8C8", facecolor="white", framealpha=1.0)
DPI = 600

RANK_NAMES = {
    "k": "kingdom",
    "p": "phylum",
    "c": "class",
    "o": "order",
    "f": "family",
    "g": "genus",
    "s": "species",
}
TRANSFORMS = ("clr", "alr", "pa", "rank", "ilr")
SPECIAL = {
    "shannon_entropy": "Shannon entropy",
    "F_low_abun": "low-abundance features (summed)",
    "F_low_var": "low-variance features (summed)",
    "age": "Age",
    "bmi": "BMI",
    "gender_m": "Gender (male)",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11,
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
            "hatch.linewidth": 0.6,
            # EPS has no alpha channel: keep every element opaque.
            "ps.fonttype": 42,
            "pdf.fonttype": 42,
        }
    )


def load_taxonomy(usecase: str) -> pd.Series:
    """Feature id -> lineage string for a use case, from its taxonomy table."""
    path = REPO_ROOT / USECASES[usecase]["path_tax"]
    return pd.read_csv(path, sep="\t", index_col=0)["Taxon"].astype(str)


def taxon_label(lineage: str, rank_tag: bool = True) -> str:
    """Deepest classified rank of a lineage, e.g. `Roseburia`.

    Species are joined with their genus; ranks left empty, `unclassified` or
    `undef` are skipped. Unassigned lineages give `unassigned`.
    """
    parsed = []
    for rank in str(lineage).split(";"):
        rank = rank.strip()
        if len(rank) <= 3 or rank[1:3] != "__":
            continue
        name = rank[3:]
        if not name or name.startswith("undef") or name.lower() == "unclassified":
            continue
        parsed.append((rank[0], name))
    if not parsed:
        return "unassigned"
    level, name = parsed[-1]
    if level == "s" and len(parsed) > 1 and parsed[-2][0] == "g":
        name = f"{parsed[-2][1]} {name}"
    elif name.endswith("_unclassified"):
        name = f"{name.removesuffix('_unclassified')} (uncl.)"
    elif rank_tag and level not in ("g", "s"):  # say how coarse a lineage is
        name = f"{name} ({RANK_NAMES[level]})"
    return name.replace("_", " ")


def short_id(feature_id: str) -> str:
    """Compact form of a feature id: OTU number, accession, or ASV prefix."""
    if re.fullmatch(r"Otu\d+", feature_id):
        return f"OTU {int(feature_id[3:])}"
    if re.fullmatch(r"[ACGT]{30,}", feature_id):
        return f"ASV {feature_id[:8]}…"
    if re.fullmatch(r"[0-9a-f]{32,}", feature_id):
        return f"{feature_id[:8]}…"
    if re.fullmatch(r"\d+", feature_id):
        return f"OTU {feature_id}"
    return feature_id.split(".")[0]


def _core_label(core: str, taxonomy: pd.Series | None, rank_tag: bool) -> str:
    if core in SPECIAL:
        return SPECIAL[core]
    feature_id = core[1:] if core.startswith("F") else core
    if taxonomy is not None and feature_id in taxonomy.index:
        return f"{taxon_label(taxonomy[feature_id], rank_tag)} ({short_id(feature_id)})"
    match = re.fullmatch(r"([kpcofgs])__(.+)", feature_id)
    if match:  # a taxonomic aggregate is named by its taxon
        return f"{match.group(2).replace('_', ' ')} ({RANK_NAMES[match.group(1)]})"
    return core.replace("_", " ").capitalize()


def feature_label(
    feature: str,
    taxonomy: pd.Series | None = None,
    retained: list[str] | None = None,
    rank_tag: bool = True,
) -> str:
    """Readable label for a ritme feature name.

    ``retained`` is the deployed model's retained feature order
    (`snapshot_selected_map`), needed to name ilr balances; without it a
    balance keeps its index only. ``rank_tag`` appends the rank to lineages
    classified no deeper than family.
    """
    match = re.fullmatch(rf"({'|'.join(TRANSFORMS)})_(.+)", feature)
    if not match:
        return _core_label(feature, taxonomy, rank_tag)
    transform, core = match.groups()
    if transform == "ilr":
        k = int(core)
        if retained is None or k + 1 >= len(retained):
            return f"ilr balance {k}"
        part = _core_label(retained[k + 1], taxonomy, rank_tag)
        return f"{part} vs. {k + 1} preceding [ilr]"
    return f"{_core_label(core, taxonomy, rank_tag)} [{transform}]"
