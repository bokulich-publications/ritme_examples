"""Supplementary figure: u1 top features with and without metadata.

Contrasts the two ritme XGBoost regressions of use case 1 (infant age from
gut 16S profiles) -- the winner of the search with metadata enrichment
(`u1_xgb_tpe`: health status, milk diet, weaning, recent antibiotics) and the
winner of the same search without it (`u1_xgb_tpe_no_enrich`). Both act on
ilr balances of the retained OTUs. Under skbio's Gram-Schmidt basis, balance k
contrasts one OTU against all OTUs retained before it, so every balance is
named after, and matched across the two models by, that defining OTU.

Feature importance is the mean absolute SHAP value on the test split, read
from the `shap_values_xgb.pkl` that `ritme explain-features` wrote.

a  top features of either model, side by side
b  every matched balance's importance in one model against the other
c  test RMSE of both models, of the published random forest and of a
   metadata-only baseline, overall and per age bracket

Usage (repo root, ritme_usecases env):
    python -m final_figures.make_supp_u1_features
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

STEM = "supp_u1_features"
USECASE = "u1"
MODELS = {
    "with": ("u1_xgb_tpe", "ritme with metadata"),
    "without": ("u1_xgb_tpe_no_enrich", "ritme without metadata"),
}
DATA_DIR = REPO_ROOT / "data" / "u1_subramanian14"
SPLITS_DIR = REPO_ROOT / "use_cases" / "u1_amplicon_age_prediction" / "data_splits_u1"
PREDICTIONS = OUT_DIR / "supp_scatter_roc" / "predictions"
TARGET = "age_months"
METADATA = ["health_status_at_sampling", "diet_milk", "diet_weaning", "abx_7d_prior"]
METADATA_LABELS = {
    "diet_weaning": "Diet: weaned",
    "diet_milk_mixed": "Diet milk: mixed",
    "diet_milk_fd": "Diet milk: formula",
    "diet_milk_no milk": "Diet milk: none",
    "diet_milk_bd": "Diet milk: breast",
    "health_status_at_sampling_healthy": "Health status: healthy",
    "abx_7d_prior": "Antibiotics (7 d prior)",
    "shannon_entropy": "Shannon entropy",
}
TOP_N = 15
AGE_BINS = [0, 3, 6, 12, 24]
AGE_LABELS = ["0-3", "3-6", "6-12", "12-24"]

MARKER_SIZE = 28
Y_TOP = 7.0
HEADER_Y = 6.85

PAGE_WIDTH = 8.0
A_LEFT, A_RIGHT, A_TOP, ROW_HEIGHT = 3.2, 7.75, 0.62, 0.245
ROW_GAP = 1.2
B_LEFT, B_SIZE = 0.95, 2.85
C_LEFT, C_WIDTH = 4.85, 2.90
BOTTOM_MARGIN = 0.80


def load_model(key: str):
    with open(RUNS_DIR / MODELS[key][0] / "xgb_best_model.pkl", "rb") as fh:
        return pickle.load(fh)


def importances() -> tuple[pd.DataFrame, dict[str, str]]:
    """Mean |SHAP| per feature and model, balances keyed by their defining OTU.

    Returns the table plus, per key, the ilr balance name the feature had in
    the with-metadata model (for the caption).
    """
    frames, balance_names = {}, {}
    for key in MODELS:
        with open(RUNS_DIR / MODELS[key][0] / "shap_values_xgb.pkl", "rb") as fh:
            explanation = pickle.load(fh)
        retained = load_model(key).snapshot_selected_map["t0"]
        values = pd.Series(
            np.abs(explanation.values).mean(axis=0), index=explanation.feature_names
        )
        keys = []
        for feature in values.index:
            if feature.startswith("ilr_"):
                part = retained[int(feature[4:]) + 1]
                keys.append(part)
                if key == "with":
                    balance_names[part] = feature
            else:
                keys.append(feature)
        frames[key] = pd.Series(values.to_numpy(), index=keys)
    return pd.DataFrame(frames), balance_names


def is_metadata(feature: str) -> bool:
    return feature in METADATA_LABELS


def labels_for(features: list[str], taxonomy: pd.Series) -> dict[str, str]:
    return {
        f: (
            METADATA_LABELS[f]
            if is_metadata(f)
            else feature_label(f, taxonomy)  # an ilr balance named by its OTU
        )
        for f in features
    }


def panel_a_rows(imp: pd.DataFrame) -> list[str]:
    chosen = set()
    for key in MODELS:
        chosen |= set(imp[key].nlargest(TOP_N).index)
    order = imp["with"].fillna(-1)
    return sorted(chosen, key=lambda f: -order[f])


def test_predictions() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    md = pd.read_csv(DATA_DIR / "md_subr14.tsv", sep="\t", index_col=0)
    train = pd.read_pickle(SPLITS_DIR / "train_val.pkl")
    test = pd.read_pickle(SPLITS_DIR / "test.pkl")
    preds = {}
    for arm, tag in (("with", "u1_ritme"), ("original", "u1_original")):
        frame = pd.read_csv(PREDICTIONS / f"{tag}.csv").query("split == 'test'")
        preds[arm] = frame.set_index("sample_id")["y_pred"].reindex(test.index)
    model = load_model("without")
    model.predict(train, "train")  # pins the engineered columns for the test pass
    preds["without"] = pd.Series(model.predict(test, "test"), index=test.index)
    design = pd.get_dummies(md[METADATA].astype(str), dtype=float)
    forest = RandomForestRegressor(n_estimators=500, random_state=0).fit(
        design.loc[train.index], md.loc[train.index, TARGET]
    )
    preds["metadata_only"] = pd.Series(
        forest.predict(design.loc[test.index]), index=test.index
    )
    return md.loc[test.index], preds


def rmse(pred: pd.Series, truth: pd.Series) -> float:
    return float(np.sqrt(((pred - truth) ** 2).mean()))


def draw_bars(ax, imp, rows, labels):
    y = np.arange(len(rows))
    height = 0.38
    for i, key in enumerate(MODELS):
        values = imp.loc[rows, key]
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
                    0.02,
                    yi,
                    "not in model",
                    va="center",
                    ha="left",
                    fontsize=NOTE_SIZE,
                    color="0.35",
                    style="italic",
                )
    ax.set_yticks(y)
    ax.set_yticklabels([labels[f] for f in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, None)
    ax.set_xlabel("Mean |SHAP| on the test split (months)")
    ax.grid(axis="y")
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", fontsize=TICK_SIZE, **LEGEND_KW)


def draw_scatter(ax, imp, labels) -> dict:
    shared = imp.dropna()
    shared = shared[[not is_metadata(f) for f in shared.index]]
    rho = spearmanr(shared["with"], shared["without"]).statistic
    lim = float(imp.max().max()) * 1.12
    ax.plot([0, lim], [0, lim], linestyle=":", color="black", linewidth=1, zorder=1)
    ax.scatter(
        shared["with"],
        shared["without"],
        s=MARKER_SIZE * 0.6,
        color=COLORS["microbial"],
        edgecolor="0.35",
        linewidth=0.4,
        zorder=3,
        label="ilr balance (matched by OTU)",
    )
    meta = imp[[is_metadata(f) for f in imp.index]]
    meta = meta[meta["with"].notna() & (meta["with"] > 0)]
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
    top_meta = meta["with"].idxmax()
    ax.annotate(
        labels[top_meta],
        (meta.loc[top_meta, "with"], 0),
        xytext=(-4, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=NOTE_SIZE - 1,
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(-lim * 0.08, lim)
    ax.set_xlabel("Mean |SHAP| with metadata")
    ax.set_ylabel("Mean |SHAP| without metadata")
    ax.legend(loc="upper right", fontsize=NOTE_SIZE, **LEGEND_KW)
    share = {
        k: float(
            imp.loc[[f for f in imp.index if is_metadata(f)], k].sum() / imp[k].sum()
        )
        for k in MODELS
    }
    return {"matched": len(shared), "rho": rho, "share": share}


def draw_age_bins(ax, md, preds) -> dict:
    truth = md[TARGET]
    bins = pd.cut(truth, AGE_BINS, labels=AGE_LABELS, include_lowest=True)
    groups = {"all test\nsamples": truth.index}
    for name in AGE_LABELS:
        groups[f"{name}\nmonths"] = truth.index[bins == name]
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
                value + 0.06,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=NOTE_SIZE - 1.5,
            )
    x_lo, x_hi = -0.65, len(groups) - 0.35
    ax.axvline(0.5, color="0.3", linewidth=0.8, zorder=1)
    headers = (
        ((x_lo + 0.5) / 2, "prediction task"),
        ((0.5 + x_hi) / 2, "by age bracket"),
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
        fontsize=NOTE_SIZE - 1,
    )
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, Y_TOP)
    ax.set_ylabel("RMSE (months), test split")
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


def write_caption(stats, n_rows, overlap, panel_c) -> None:
    n = panel_c["n"]
    brackets = ", ".join(
        f"{k.split()[0]} months n = {v}"
        for k, v in n.items()
        if k != "all test samples"
    )
    text = (
        "Top features of the use case 1 ritme models with and without metadata. "
        "(a) Mean absolute SHAP value on the test split (in months of predicted "
        f"age) of the features ranking in either model's top {TOP_N} ({n_rows} "
        "features, ordered by the with-metadata model). Microbial features are "
        "ilr balances; under the Gram-Schmidt basis each balance contrasts one "
        "OTU against every OTU retained before it and is labelled by that OTU, "
        "which also matches balances between the two models. 'Not in model' "
        "marks the metadata covariates unavailable to the without-metadata "
        "search. (b) Importance of every balance defined by an OTU both models "
        f"retained ({stats['matched']} balances), with metadata (x) against "
        "without (y); dotted line, identity; metadata features are drawn at "
        f"y = 0. Spearman's rho is {stats['rho']:.2f}; the metadata covariates carry "
        f"{stats['share']['with']:.0%} of the with-metadata model's total importance. "
        "(c) Test-split RMSE of the two ritme models, of the published setup "
        "(random forest on the rarefied OTU table, microbiome only) and of a "
        "random forest on the metadata alone (health status, milk diet, weaning, "
        "antibiotics), for the prediction task (left of the divider; "
        f"n = {n['all test samples']}) and, right of it, per age bracket "
        f"({brackets}). "
        f"Models: ritme with metadata = {MODELS['with'][0]}, ritme without "
        f"metadata = {MODELS['without'][0]}.\n"
    )
    (OUT_DIR / f"{STEM}_caption.txt").write_text(text)


def main() -> None:
    style()
    imp, balance_names = importances()
    taxonomy = load_taxonomy(USECASE)
    rows = panel_a_rows(imp)
    labels = labels_for(list(imp.index), taxonomy)
    overlap = len(
        set.intersection(*[set(imp[k].nlargest(TOP_N).index) for k in MODELS])
    )
    print(f"top-{TOP_N} overlap (balances matched by OTU): {overlap} of {TOP_N}")
    for f in rows:
        print(
            f"    {labels[f]:<50} with {imp.loc[f, 'with']:.3f}  "
            f"without {imp.loc[f, 'without']:.3f}  {balance_names.get(f, '')}"
        )
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
    draw_bars(ax_a, imp, rows, labels)
    stats = draw_scatter(ax_b, imp, labels)
    panel_c = draw_age_bins(ax_c, md, preds)
    print(
        f"matched balances {stats['matched']} | rho {stats['rho']:.3f} | "
        f"metadata share {stats['share']}"
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
    write_caption(stats, len(rows), overlap, panel_c)
    print(f"written to {OUT_DIR / STEM}.eps / .pdf / .png and {STEM}_caption.txt")


if __name__ == "__main__":
    main()
