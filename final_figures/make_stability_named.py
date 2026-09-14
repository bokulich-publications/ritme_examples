"""Re-draw every stability plot with a trimmed legend and readable names.

`ritme explain-stability` labels the heatmap rows with design-matrix column
names (`FOtu00367`, `pa_FTACGGAGG...`, `ilr_127`) and prints the full cell
legend whether or not a figure uses every symbol. This reads the CSVs it
wrote next to each plot and calls ritme's own `plot_stability` twice: once
with the original names (`stability_plot.png`, replaced in place) and once
with the names mapped through `supp_features_common.feature_label`
(`stability_plot_named.png`). Both keep only the legend entries whose match
kind occurs in that figure.

ilr balances are named after the feature that defines them, read from the
deployed model's retained-feature order; every other transform keeps its
taxon plus a `[transform]` tag.

Usage (repo root, an environment with ritme, e.g. ritme_usecases_new):
    python -m final_figures.make_stability_named
"""

from __future__ import annotations

import pickle
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from ritme import explain_stability  # noqa: E402
from ritme.explain_stability import plot_stability  # noqa: E402

from src.launch_models import USECASES  # noqa: E402

from final_figures.supp_features_common import (  # noqa: E402
    RUNS_DIR,
    feature_label,
    load_taxonomy,
)

METRIC = {"regression": "rmse_val", "classification": "roc_auc_macro_ovr_val"}
# Leading symbol of the legend entry that explains each match kind; plain
# ranks ("N = ...") occur in every figure.
LEGEND_SYMBOL = {
    "contained": "◆",
    "lumped": "L",
    "split": "s",
    "absent": "–",
    "not_attributable": "grey n/a",
}


def used_legend_entries(ranks: pd.DataFrame) -> tuple:
    symbols = {"N"} | {
        LEGEND_SYMBOL[k] for k in set(ranks["match_kind"]) & set(LEGEND_SYMBOL)
    }
    return tuple(
        entry
        for entry in explain_stability._CELL_LEGEND_ENTRIES
        if entry.split(" =")[0] in symbols
    )


def draw(manifest, ranks, metric, top_n, path) -> None:
    """ritme's stability figure with the legend cut to the symbols it uses.

    `plot_stability` reads the legend entries from its module global, so the
    trimmed tuple is patched in for the duration of the call.
    """
    entries = used_legend_entries(ranks)
    with patch.object(explain_stability, "_CELL_LEGEND_ENTRIES", entries):
        fig = plot_stability(manifest, ranks, metric, top_n=top_n, show=False)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def retained_order(run_dir, model_type: str) -> list[str] | None:
    """Retained feature order of the deployed model, for naming ilr balances."""
    path = run_dir / f"{model_type}_best_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    return model.snapshot_selected_map.get("t0")


def unique_labels(features: list[str], taxonomy, retained) -> dict[str, str]:
    """Label per feature; a label used twice gets the raw name appended."""
    labels = {f: feature_label(f, taxonomy, retained) for f in features}
    counts = pd.Series(list(labels.values())).value_counts()
    return {
        f: (label if counts[label] == 1 else f"{label} ({f})")
        for f, label in labels.items()
    }


def main() -> None:
    for stability_dir in sorted(RUNS_DIR.glob("u*/stability_*")):
        if not stability_dir.is_dir() or stability_dir.name.endswith("_pre_pr126"):
            continue
        run_dir = stability_dir.parent
        model_type = stability_dir.name.removeprefix("stability_")
        manifest = pd.read_csv(stability_dir / "stability_trials.csv")
        ranks = pd.read_csv(stability_dir / "stability_ranks.csv")

        taxonomy = load_taxonomy(run_dir.name[:2])
        retained = retained_order(run_dir, model_type)
        labels = unique_labels(ranks["feature"].unique().tolist(), taxonomy, retained)
        reference = manifest.loc[manifest["is_reference"], "run_id"].iloc[0]
        top_n = int((ranks["run_id"] == reference).sum())

        metric = METRIC[USECASES[run_dir.name[:2]]["task"]]
        draw(manifest, ranks, metric, top_n, stability_dir / "stability_plot.png")
        draw(
            manifest,
            ranks.assign(feature=ranks["feature"].map(labels)),
            metric,
            top_n,
            stability_dir / "stability_plot_named.png",
        )
        kinds = ", ".join(sorted(set(ranks["match_kind"])))
        print(f"{run_dir.name}/{stability_dir.name}: legend for [{kinds}]")
        for feature in ranks.loc[ranks["run_id"] == reference].sort_values(
            "reference_rank"
        )["feature"]:
            print(f"    {feature:<28} -> {labels[feature]}")


if __name__ == "__main__":
    main()
