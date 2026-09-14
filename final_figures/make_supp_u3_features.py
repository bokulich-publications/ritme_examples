"""Supplementary figure: u3 top features with and without metadata.

Contrasts the two ritme logistic regressions of use case 3 -- the winner of
the search with metadata enrichment (`u3_logreg_tpe_no_fit`: age, gender,
BMI) and the winner of the same search without it (`u3_logreg_tpe_no_enrich`).

a  features carrying the largest standardised coefficients in either model,
   side by side
b  every feature's coefficient in one model against the other
c  test ROC AUC of both models, of the published setup and of age alone,
   for the two lesion types that make up the positive class
   (screen-relevant neoplasia, SRN)

Coefficients are read from the `feature_importance_logreg.csv` that
`ritme explain-features` wrote next to each deployed model. Panel c reloads
the two models and scores the held-out split; the published setup's scores
come from `supp_scatter_roc/extract_predictions.py u3 original`, run first.

Also writes `supp_u3_features_caption.txt`, the figure caption with the
summary statistics filled in.

Usage (repo root, ritme_usecases env):
    python -m final_figures.make_supp_u3_features
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "use_cases" / "ritme_runs" / "local"
DATA_DIR = REPO_ROOT / "data" / "u3_topcuoglu20_baxter"
SPLITS_DIR = (
    REPO_ROOT / "use_cases" / "u3_amplicon_crc_classification" / "data_splits_u3"
)
OUT_DIR = REPO_ROOT / "final_figures"
ORIGINAL_PREDICTIONS = OUT_DIR / "supp_scatter_roc" / "predictions" / "u3_original.csv"
STEM = "supp_u3_features"

MODELS = {
    "with": ("u3_logreg_tpe_no_fit", "ritme with metadata"),
    "without": ("u3_logreg_tpe_no_enrich", "ritme without metadata"),
}
METADATA_FEATURES = {"age": "Age", "gender_m": "Gender (male)", "bmi": "BMI"}
TOP_N = 15  # per model; panel a shows the union
TARGET = "srn"

COLORS = {  # Set3, as in figures 4-5; orange = metadata involved, blue = not
    "with": "#FDB462",
    "without": "#80B1D3",
    "metadata": "#FDB462",
    "microbial": "#80B1D3",
    "original": "#BEBADA",
    "age": "#D9D9D9",
}
METADATA_HATCH = "////"
BAR_EDGE = "black"
LABEL_SIZE = 12.5  # GROUP_LABEL_SIZE of figure 4
TICK_SIZE = 10
NOTE_SIZE = 9
MARKER_SIZE = 28
Y_TOP = 0.99  # panel c; leaves room for the group headers above the bars
HEADER_Y = 0.985
DPI = 600

# Page geometry in inches: panel a spans the width, b and c share the row
# below. The panel-a height follows from its row count.
PAGE_WIDTH = 8.0
A_LEFT, A_RIGHT, A_TOP, ROW_HEIGHT = 2.75, 7.75, 0.62, 0.255
ROW_GAP = 1.2
B_LEFT, B_SIZE = 0.95, 2.85
C_LEFT, C_WIDTH = 4.85, 2.90
BOTTOM_MARGIN = 0.80
LEGEND_KW = dict(frameon=True, edgecolor="#C8C8C8", facecolor="white", framealpha=1.0)


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


def load_coefficients() -> pd.DataFrame:
    """Signed coefficient per feature and model; NaN where the model lacks it."""
    frames = {}
    for key, (tag, _) in MODELS.items():
        path = RUNS_DIR / tag / "feature_importance_logreg.csv"
        frames[key] = pd.read_csv(path).set_index("feature")["coefficient"]
    return pd.DataFrame(frames)


def feature_names() -> dict[str, str]:
    """`FOtu00367` -> `Peptostreptococcus (OTU 367)`, at the deepest named rank."""
    taxa = pd.read_csv(DATA_DIR / "taxonomy_baxter.tsv", sep="\t", index_col=0)["Taxon"]
    names = {}
    for otu, lineage in taxa.items():
        ranks = [r.strip() for r in lineage.split(";")]
        named = [r[3:] for r in ranks if len(r) > 3]
        label = named[-1] if named else "unclassified"
        if label.endswith("_unclassified"):
            label = f"{label.removesuffix('_unclassified')} (uncl.)"
        names[f"F{otu}"] = f"{label} (OTU {int(otu.removeprefix('Otu'))})"
    return {**names, **METADATA_FEATURES}


def top_union(coefs: pd.DataFrame) -> list[str]:
    """Features in either model's top-N by |coefficient|, in with-metadata order."""
    chosen = set()
    for key in MODELS:
        chosen |= set(coefs[key].abs().nlargest(TOP_N).index)
    order = coefs["with"].abs().fillna(-1)
    return sorted(chosen, key=lambda f: -order[f])


def test_scores() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Test-split metadata and each model's P(SRN), plus age as a bare score."""
    md = pd.read_csv(DATA_DIR / "md_baxter.tsv", sep="\t", index_col=0)
    train = pd.read_pickle(SPLITS_DIR / "train_val.pkl")
    test = pd.read_pickle(SPLITS_DIR / "test.pkl")
    scores = {}
    for key, (tag, _) in MODELS.items():
        with open(RUNS_DIR / tag / "logreg_best_model.pkl", "rb") as fh:
            model = pickle.load(fh)
        # The train pass pins the engineered column set the test pass reuses.
        model.predict_proba(train, "train")
        proba, classes = model.predict_proba(test, "test")
        scores[key] = pd.Series(proba[:, list(classes).index(1)], index=test.index)
    if not ORIGINAL_PREDICTIONS.exists():
        raise SystemExit(
            f"missing {ORIGINAL_PREDICTIONS}; run "
            "`python -m final_figures.supp_scatter_roc.extract_predictions u3 original`"
        )
    original = pd.read_csv(ORIGINAL_PREDICTIONS).query("split == 'test'")
    scores["original"] = original.set_index("sample_id")["p_1"].reindex(test.index)
    scores["age"] = md.loc[test.index, "age"].astype(float)
    return md.loc[test.index], scores


def subset_labels(md: pd.DataFrame) -> dict[str, pd.Series]:
    """Binary labels per panel-c comparison; NaN drops a sample from it.

    SRN (screen-relevant neoplasia) pools carcinomas with advanced adenomas;
    non-advanced adenomas count as negatives in the task. The two lesion-type
    comparisons score each positive subtype against normal colons only.
    """
    srn, dx = md[TARGET], md["dx"]
    normal = dx == "normal"
    carcinoma = dx == "cancer"
    advanced_adenoma = (dx == "adenoma") & (srn == 1)

    def versus_normal(positive):
        labels = pd.Series(np.nan, index=md.index)
        labels[positive] = 1
        labels[normal] = 0
        return labels

    return {
        "SRN vs.\nnon-SRN": srn.astype(float),
        "carcinoma\nvs. normal": versus_normal(carcinoma),
        "advanced\nadenoma\nvs. normal": versus_normal(advanced_adenoma),
    }


def subset_auc(labels: pd.Series, score: pd.Series) -> float:
    kept = labels.dropna()
    return roc_auc_score(kept.astype(int), score[kept.index])


def draw_bars(ax: plt.Axes, coefs: pd.DataFrame, features: list[str], names: dict):
    y = np.arange(len(features))
    height = 0.38
    for i, key in enumerate(MODELS):
        values = coefs.loc[features, key]
        offset = (0.5 - i) * height
        ax.barh(
            y + offset,
            values.fillna(0),
            height=height,
            color=COLORS[key],
            edgecolor=BAR_EDGE,
            linewidth=0.6,
            label=MODELS[key][1],
        )
        # A feature the model never saw gets a note rather than a zero-length
        # bar, which would read as "selected but weightless".
        for yi, value in zip(y + offset, values):
            if pd.isna(value):
                ax.text(
                    0.0015,
                    yi,
                    "not in model",
                    va="center",
                    ha="left",
                    fontsize=NOTE_SIZE,
                    color="0.35",
                    style="italic",
                )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([names[f] for f in features])
    ax.set_ylim(len(features) - 0.5, -0.5)
    ax.set_xlabel("Standardised coefficient (log-odds of SRN)")
    ax.grid(axis="y")
    ax.tick_params(axis="y", length=0)
    lo, hi = ax.get_xlim()
    ax.text(lo, -0.85, "← lower SRN risk", ha="left", va="bottom", fontsize=TICK_SIZE)
    ax.text(hi, -0.85, "higher SRN risk →", ha="right", va="bottom", fontsize=TICK_SIZE)
    ax.legend(loc="center right", fontsize=TICK_SIZE, **LEGEND_KW)


def draw_scatter(ax: plt.Axes, coefs: pd.DataFrame, names: dict) -> dict:
    shared = coefs.dropna()
    active = shared[(shared != 0).any(axis=1)]
    rho = spearmanr(shared["with"].abs(), shared["without"].abs()).statistic
    both = active[(active != 0).all(axis=1)]
    agree = (np.sign(both["with"]) == np.sign(both["without"])).mean()

    lim = float(coefs.abs().max().max()) * 1.15
    ax.plot(
        [-lim, lim], [-lim, lim], linestyle=":", color="black", linewidth=1, zorder=1
    )
    ax.axhline(0, color="0.6", linewidth=0.6, zorder=1)
    ax.axvline(0, color="0.6", linewidth=0.6, zorder=1)
    ax.scatter(
        active["with"],
        active["without"],
        s=MARKER_SIZE,
        color=COLORS["microbial"],
        edgecolor="0.35",
        linewidth=0.5,
        zorder=3,
        label="microbial feature",
    )
    # Metadata features exist in one model only: drawn on the y = 0 line.
    meta = coefs.loc[[f for f in METADATA_FEATURES if f in coefs.index]]
    meta = meta[meta["with"] != 0]
    ax.scatter(
        meta["with"],
        np.zeros(len(meta)),
        s=MARKER_SIZE + 12,
        marker="D",
        color=COLORS["metadata"],
        edgecolor="0.35",
        linewidth=0.5,
        zorder=4,
        label="metadata feature",
    )
    for i, (feature, row) in enumerate(meta.sort_values("with").iterrows()):
        ax.annotate(
            names[feature],
            (row["with"], 0),
            xytext=(0, -9 - 11 * (i % 2)),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=NOTE_SIZE,
        )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Coefficient with metadata")
    ax.set_ylabel("Coefficient without metadata")
    ax.legend(loc="lower right", fontsize=NOTE_SIZE, **LEGEND_KW)
    return {"shared": len(shared), "rho": rho, "both": len(both), "agree": agree}


def draw_subsets(
    ax: plt.Axes, md: pd.DataFrame, scores: dict[str, pd.Series]
) -> dict[str, tuple[int, int]]:
    arms = [
        ("with", MODELS["with"][1], None),
        ("without", MODELS["without"][1], None),
        ("original", "original setup", None),
        ("age", "age alone", METADATA_HATCH),
    ]
    subsets = subset_labels(md)
    x = np.arange(len(subsets))
    width = 0.2
    for i, (key, label, hatch) in enumerate(arms):
        aucs = [subset_auc(labels, scores[key]) for labels in subsets.values()]
        bars = ax.bar(
            x + (i - 1.5) * width,
            aucs,
            width=width,
            color=COLORS[key],
            edgecolor=BAR_EDGE,
            linewidth=0.6,
            hatch=hatch,
            label=label,
        )
        for bar, value in zip(bars, aucs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=NOTE_SIZE - 1,
            )
        summary = ", ".join(f"{k} {v:.3f}" for k, v in zip(subsets, aucs))
        print(f"{label}: {summary}".replace("\n", " "))
    labels = [
        f"{name}\n(n = {int((lab == 1).sum())} / {int((lab == 0).sum())})"
        for name, lab in subsets.items()
    ]
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, zorder=1)
    # The first group is the prediction task itself; the rest dissect it.
    x_lo, x_hi = -0.65, len(subsets) - 0.35
    ax.axvline(0.5, color="0.3", linewidth=0.8, zorder=1)
    headers = (
        ((x_lo + 0.5) / 2, "prediction task"),
        ((0.5 + x_hi) / 2, "by lesion type"),
    )
    for x_text, header in headers:
        ax.text(
            x_text,
            HEADER_Y,
            header,
            ha="center",
            va="top",
            fontsize=NOTE_SIZE - 0.5,
            style="italic",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=NOTE_SIZE)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.45, Y_TOP)
    ax.set_ylabel("ROC AUC, test split")
    ax.grid(axis="x")
    counts = {
        name: (int((lab == 1).sum()), int((lab == 0).sum()))
        for name, lab in subsets.items()
    }
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1, 1.02),
        ncol=2,
        columnspacing=1.0,
        handlelength=1.4,
        fontsize=NOTE_SIZE,
        **LEGEND_KW,
    )
    return counts


def write_caption(stats: dict, counts: dict, n_rows: int, overlap: int) -> Path:
    task, carcinoma, adenoma = counts.values()
    text = (
        "Top features of the use case 3 ritme models with and without metadata. "
        "(a) Standardised logistic-regression coefficients (log-odds of "
        "screen-relevant neoplasia, SRN) of the features ranking in either "
        f"model's top {TOP_N} by absolute coefficient ({n_rows} features), ordered "
        "by the with-metadata model; positive values indicate higher SRN risk. "
        "'Not in model' marks the metadata covariates unavailable to the "
        "without-metadata search; BMI received a zero coefficient in the "
        "with-metadata model. (b) Coefficient of every feature with a non-zero "
        "weight in at least one model, with metadata (x) against without "
        "metadata (y); dotted line, identity. Metadata features exist in one "
        f"model only and are drawn at y = 0. Over the {stats['shared']} features "
        "shared by the two models, Spearman's rho of the absolute coefficients is "
        f"{stats['rho']:.2f}, the {stats['both']} features non-zero in both models "
        f"agree in sign ({stats['agree']:.0%}) and {overlap} of the {TOP_N} top "
        "features coincide. (c) Test-split ROC AUC of the two ritme models, of "
        "the published setup (random forest on the rarefied OTU table, "
        "microbiome only) and of age alone, for the SRN prediction task (left "
        f"of the divider; n = {task[0]} SRN / {task[1]} non-SRN samples) and, right of "
        "it, for each SRN lesion type against normal colons "
        f"(carcinoma n = {carcinoma[0]} / {carcinoma[1]}, "
        f"advanced adenoma n = {adenoma[0]} / {adenoma[1]}); non-advanced adenomas "
        "are SRN-negative and therefore enter only the task comparison. Models: "
        f"ritme with metadata = {MODELS['with'][0]} (age, gender, BMI; "
        "elastic-net logistic regression), ritme without metadata = "
        f"{MODELS['without'][0]} (the same search without metadata enrichment).\n"
    )
    path = OUT_DIR / f"{STEM}_caption.txt"
    path.write_text(text)
    return path


def main() -> None:
    style()
    coefs = load_coefficients()
    names = feature_names()
    features = top_union(coefs)
    overlap = set.intersection(
        *[set(coefs[k].abs().nlargest(TOP_N).index) for k in MODELS]
    )
    print(f"top-{TOP_N} overlap: {len(overlap)} of {TOP_N}")
    md, scores = test_scores()

    a_height = ROW_HEIGHT * len(features)
    page_height = A_TOP + a_height + ROW_GAP + B_SIZE + BOTTOM_MARGIN
    fig = plt.figure(figsize=(PAGE_WIDTH, page_height))

    def add_axes(left, bottom, width, height):
        return fig.add_axes(
            [
                left / PAGE_WIDTH,
                bottom / page_height,
                width / PAGE_WIDTH,
                height / page_height,
            ]
        )

    ax_a = add_axes(
        A_LEFT, BOTTOM_MARGIN + B_SIZE + ROW_GAP, A_RIGHT - A_LEFT, a_height
    )
    ax_b = add_axes(B_LEFT, BOTTOM_MARGIN, B_SIZE, B_SIZE)
    ax_c = add_axes(C_LEFT, BOTTOM_MARGIN, C_WIDTH, B_SIZE)
    draw_bars(ax_a, coefs, features, names)
    stats = draw_scatter(ax_b, coefs, names)
    counts = draw_subsets(ax_c, md, scores)

    for ax, letter, dx in (
        (ax_a, "a", -A_LEFT + 0.15),
        (ax_b, "b", -0.75),
        (ax_c, "c", -0.6),
    ):
        box = ax.get_position()
        fig.text(
            box.x0 + dx / PAGE_WIDTH,
            box.y1 + 0.12 / page_height,
            letter,
            fontsize=LABEL_SIZE + 1,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    for suffix in ("eps", "pdf", "png"):
        fig.savefig(OUT_DIR / f"{STEM}.{suffix}", dpi=DPI)
    plt.close(fig)
    caption = write_caption(stats, counts, len(features), len(overlap))
    print(f"written to {OUT_DIR / STEM}.eps / .pdf / .png and {caption.name}")


if __name__ == "__main__":
    main()
