"""Supplementary figure: u2 top features with and without metadata.

Contrasts the two ritme elastic-net regressions of use case 2 (ocean
temperature from Tara Oceans OTU profiles) -- the winner of the search with
metadata enrichment (`u2_linreg_tpe`: depth, layer, latitude, longitude) and
the winner of the same search without it (`u2_linreg_tpe_no_enrich`). Both
act on within-sample abundance ranks (rank 1 = most abundant), so a positive
coefficient means the taxon is less abundant in warmer water.

a  top features of either model plus the metadata covariates, side by side
b  every shared feature's coefficient in one model against the other
c  test RMSE of both models, of the published elastic net and of a
   metadata-only baseline, overall and per depth layer

Usage (repo root, ritme_usecases env):
    python -m final_figures.make_supp_u2_features
"""

from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from final_figures.supp_features_common import (
    BAR_EDGE,
    COLORS,
    DPI,
    LABEL_SIZE,
    LEGEND_KW,
    METADATA_HATCH,
    NOTE_SIZE,
    OUT_DIR,
    REPO_ROOT,
    RUNS_DIR,
    TICK_SIZE,
    feature_label,
    load_taxonomy,
    style,
)

STEM = "supp_u2_features"
USECASE = "u2"
MODELS = {
    "with": ("u2_linreg_tpe", "ritme with metadata"),
    "without": ("u2_linreg_tpe_no_enrich", "ritme without metadata"),
}
DATA_DIR = REPO_ROOT / "data" / "u2_tara_ocean"
SPLITS_DIR = REPO_ROOT / "use_cases" / "u2_metagenome_ocean" / "data_splits_u2"
PREDICTIONS = OUT_DIR / "supp_scatter_roc" / "predictions"
TARGET = "temperature_mean_degc"
METADATA = ["sampling_depth_m", "latitude", "longitude", "env_feature"]
METADATA_LABELS = {
    "sampling_depth_m": "Sampling depth",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "shannon_entropy": "Shannon entropy",
}
TOP_N = 10  # per model; panel a shows the union plus the metadata rows
LAYER_ORDER = ["SRF", "DCM", "MES"]  # MIX has two test samples and is left out

MARKER_SIZE = 28
Y_TOP = 4.3
HEADER_Y = 4.2

PAGE_WIDTH = 8.0
A_LEFT, A_RIGHT, A_TOP, ROW_HEIGHT = 3.2, 7.75, 0.62, 0.245
ROW_GAP = 1.2
B_LEFT, B_SIZE = 0.95, 2.85
C_LEFT, C_WIDTH = 4.85, 2.90
BOTTOM_MARGIN = 0.80


def load_coefficients() -> pd.DataFrame:
    frames = {}
    for key, (tag, _) in MODELS.items():
        path = RUNS_DIR / tag / "feature_importance_linreg.csv"
        frames[key] = pd.read_csv(path).set_index("feature")["coefficient"]
    return pd.DataFrame(frames)


def is_metadata(feature: str) -> bool:
    return not feature.startswith("rank_")


def labels_for(features: list[str], taxonomy: pd.Series) -> dict[str, str]:
    out = {}
    for f in features:
        if f in METADATA_LABELS:
            out[f] = METADATA_LABELS[f]
        elif f.startswith("env_feature_"):
            out[f] = "Layer: " + f.removeprefix("env_feature_").split(")")[0].strip("(")
        else:  # every microbial feature is rank-transformed; the tag is implied
            out[f] = feature_label(f, taxonomy, rank_tag=False).removesuffix(" [rank]")
    return out


def panel_a_rows(coefs: pd.DataFrame) -> list[str]:
    """Union of both models' top-N microbial features, then the metadata rows."""
    microbial = coefs[[not is_metadata(f) for f in coefs.index]]
    chosen = set()
    for key in MODELS:
        chosen |= set(microbial[key].abs().nlargest(TOP_N).index)
    order = coefs["with"].abs().fillna(-1)
    rows = sorted(chosen, key=lambda f: -order[f])
    # Continuous covariates and Shannon entropy; the layer dummies are left to
    # the caption to keep the panel on one page.
    metadata = [
        f for f in METADATA_LABELS if f in coefs.index and coefs.loc[f, "with"] != 0
    ]
    return rows + sorted(metadata, key=lambda f: -order[f])


def test_predictions() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    md = pd.read_csv(DATA_DIR / "md_tara_ocean.tsv", sep="\t", index_col=0)
    train = pd.read_pickle(SPLITS_DIR / "train_val.pkl")
    test = pd.read_pickle(SPLITS_DIR / "test.pkl")
    preds = {}
    for arm, tag in (("with", "u2_ritme"), ("original", "u2_original")):
        frame = pd.read_csv(PREDICTIONS / f"{tag}.csv").query("split == 'test'")
        preds[arm] = frame.set_index("sample_id")["y_pred"].reindex(test.index)
    with open(RUNS_DIR / MODELS["without"][0] / "linreg_best_model.pkl", "rb") as fh:
        model = pickle.load(fh)
    model.predict(train, "train")  # pins the engineered columns for the test pass
    preds["without"] = pd.Series(model.predict(test, "test"), index=test.index)
    # What the covariates alone predict: a random forest on depth, position
    # and layer, fitted on the training split.
    design = pd.concat(
        [md[METADATA[:3]], pd.get_dummies(md[METADATA[3]], dtype=float)], axis=1
    )
    forest = RandomForestRegressor(n_estimators=500, random_state=0).fit(
        design.loc[train.index], md.loc[train.index, TARGET]
    )
    preds["metadata_only"] = pd.Series(
        forest.predict(design.loc[test.index]), index=test.index
    )
    return md.loc[test.index], preds


def rmse(pred: pd.Series, truth: pd.Series) -> float:
    return float(np.sqrt(((pred - truth) ** 2).mean()))


def draw_bars(ax, coefs, rows, labels):
    y = np.arange(len(rows))
    height = 0.38
    for i, key in enumerate(MODELS):
        values = coefs.loc[rows, key]
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
        for yi, value in zip(y + offset, values):
            if pd.isna(value):
                ax.text(
                    0.002,
                    yi,
                    "not in model",
                    va="center",
                    ha="left",
                    fontsize=NOTE_SIZE,
                    color="0.35",
                    style="italic",
                )
    n_microbial = sum(not is_metadata(f) for f in rows)
    ax.axhline(n_microbial - 0.5, color="0.3", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[f] for f in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("Standardised coefficient (°C per SD of abundance rank)")
    ax.grid(axis="y")
    ax.tick_params(axis="y", length=0)
    lo, hi = ax.get_xlim()
    ax.text(
        lo,
        -0.85,
        "← more abundant when warmer",
        ha="left",
        va="bottom",
        fontsize=TICK_SIZE,
    )
    ax.text(
        hi,
        -0.85,
        "less abundant when warmer →",
        ha="right",
        va="bottom",
        fontsize=TICK_SIZE,
    )
    ax.legend(loc="lower left", fontsize=TICK_SIZE, **LEGEND_KW)


def draw_scatter(ax, coefs, labels) -> dict:
    shared = coefs.dropna()
    shared = shared[[not is_metadata(f) for f in shared.index]]
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
        s=MARKER_SIZE * 0.5,
        color=COLORS["microbial"],
        edgecolor="0.35",
        linewidth=0.4,
        zorder=3,
        label="microbial feature",
    )
    meta = coefs[[is_metadata(f) and not f.startswith("env_") for f in coefs.index]]
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
    ax.text(
        0.04,
        0.96,
        "metadata: depth, latitude,\nlongitude, Shannon entropy\n"
        f"(|coef.| ≤ {meta['with'].abs().max():.2f})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=NOTE_SIZE - 1,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Coefficient with metadata")
    ax.set_ylabel("Coefficient without metadata")
    ax.legend(loc="lower right", fontsize=NOTE_SIZE, **LEGEND_KW)
    return {
        "shared": len(shared),
        "rho": rho,
        "both": len(both),
        "agree": agree,
        "nonzero": {k: int(((coefs[k] != 0) & coefs[k].notna()).sum()) for k in MODELS},
        "n_features": {k: int(coefs[k].notna().sum()) for k in MODELS},
    }


def draw_layers(ax, md, preds) -> dict:
    truth = md[TARGET]
    layer = md["env_feature"].str.extract(r"\((\w+)\)")[0]
    groups = {"all test\nsamples": truth.index}
    for name in LAYER_ORDER:
        groups[name] = truth.index[layer == name]
    arms = [
        ("with", MODELS["with"][1], None),
        ("without", MODELS["without"][1], None),
        ("original", "original setup", None),
        ("metadata_only", "metadata alone", METADATA_HATCH),
    ]
    x = np.arange(len(groups))
    width = 0.2
    table = {}
    for i, (key, label, hatch) in enumerate(arms):
        values = [rmse(preds[key].loc[idx], truth.loc[idx]) for idx in groups.values()]
        table[label] = dict(zip([g.replace("\n", " ") for g in groups], values))
        bars = ax.bar(
            x + (i - 1.5) * width,
            values,
            width=width,
            color=COLORS[key],
            edgecolor=BAR_EDGE,
            linewidth=0.6,
            hatch=hatch,
            label=label,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.03,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=NOTE_SIZE - 1,
            )
    x_lo, x_hi = -0.65, len(groups) - 0.35
    ax.axvline(0.5, color="0.3", linewidth=0.8, zorder=1)
    headers = (
        ((x_lo + 0.5) / 2, "prediction task"),
        ((0.5 + x_hi) / 2, "by depth layer"),
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
    ax.set_xticklabels(
        [f"{name}\n(n = {len(idx)})" for name, idx in groups.items()],
        fontsize=NOTE_SIZE,
    )
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, Y_TOP)
    ax.set_ylabel("RMSE (°C), test split")
    ax.grid(axis="x")
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1, 1.02),
        ncol=2,
        columnspacing=1.0,
        handlelength=1.4,
        fontsize=NOTE_SIZE,
        **LEGEND_KW,
    )
    return {
        "n": {g.replace("\n", " "): len(idx) for g, idx in groups.items()},
        "rmse": table,
    }


def write_caption(stats, rows, n_microbial_rows, overlap, panel_c) -> None:
    n = panel_c["n"]
    text = (
        "Top features of the use case 2 ritme models with and without metadata. "
        "(a) Standardised elastic-net coefficients (°C per standard deviation) "
        "of the microbial features ranking in either model's top "
        f"{TOP_N} by absolute coefficient ({n_microbial_rows} features, above the "
        "line, ordered by the with-metadata model) and of the metadata covariates "
        "the with-metadata model kept (below the line; its depth-layer "
        "indicators, not shown, carry coefficients between -0.04 and 0.05). "
        "Features are "
        "within-sample abundance ranks (rank 1 = most abundant), so a positive "
        "coefficient marks a taxon that is less abundant in warmer water. 'Not in "
        "model' marks features the other search did not retain. (b) Coefficient "
        "of every microbial feature with a non-zero weight in at least one model, "
        "with metadata (x) against without (y); dotted line, identity; metadata "
        f"features are drawn at y = 0. Over the {stats['shared']} microbial "
        "features shared by the two models, Spearman's rho of the absolute "
        f"coefficients is {stats['rho']:.2f} and the {stats['both']} features "
        f"non-zero in both models agree in sign in {stats['agree']:.0%} of cases; "
        f"{overlap} of the {TOP_N} top features coincide. The models keep "
        f"{stats['nonzero']['with']} of {stats['n_features']['with']} and "
        f"{stats['nonzero']['without']} of {stats['n_features']['without']} "
        "coefficients non-zero. (c) Test-split RMSE of the two ritme models, of "
        "the published setup (cross-validated elastic net, microbiome only) and "
        "of a random forest on the metadata alone (depth, latitude, longitude, "
        "layer), for the prediction task (left of the divider; "
        f"n = {n['all test samples']}) "
        "and, right of it, per depth layer: surface (SRF, "
        f"n = {n['SRF']}), deep chlorophyll maximum (DCM, n = {n['DCM']}) and "
        f"mesopelagic (MES, n = {n['MES']}); the two mixed-layer samples are "
        "omitted. Models: ritme with metadata = "
        f"{MODELS['with'][0]}, ritme without metadata = {MODELS['without'][0]}.\n"
    )
    (OUT_DIR / f"{STEM}_caption.txt").write_text(text)


def main() -> None:
    style()
    coefs = load_coefficients()
    taxonomy = load_taxonomy(USECASE)
    rows = panel_a_rows(coefs)
    labels = labels_for(list(coefs.index), taxonomy)
    microbial = coefs[[not is_metadata(f) for f in coefs.index]]
    overlap = len(
        set.intersection(
            *[set(microbial[k].abs().nlargest(TOP_N).index) for k in MODELS]
        )
    )
    n_microbial_rows = sum(not is_metadata(f) for f in rows)
    print(f"top-{TOP_N} overlap: {overlap} of {TOP_N}; panel a rows: {len(rows)}")
    md, preds = test_predictions()

    a_height = ROW_HEIGHT * len(rows)
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
    draw_bars(ax_a, coefs, rows, labels)
    stats = draw_scatter(ax_b, coefs, labels)
    panel_c = draw_layers(ax_c, md, preds)
    print(
        f"shared {stats['shared']} | rho {stats['rho']:.3f} | "
        f"sign agreement {stats['agree']:.3f}"
    )
    for arm, values in panel_c["rmse"].items():
        print(f"{arm}: " + ", ".join(f"{k} {v:.3f}" for k, v in values.items()))

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
    write_caption(stats, rows, n_microbial_rows, overlap, panel_c)
    print(f"written to {OUT_DIR / STEM}.eps / .pdf / .png and {STEM}_caption.txt")


if __name__ == "__main__":
    main()
