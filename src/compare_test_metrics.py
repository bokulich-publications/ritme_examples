"""Held-out test metrics of the ritme winners and the comparator arms, side by side.

The ritme model per use case is the manuscript's choice: ritme's 1-SE +
simplicity pick per model type, ranked across model types by mean validation
score (the selection `use_cases/evaluate_all_trials.ipynb` makes). Its point
estimates come from the experiment's `best_metrics.csv`, the 95% percentile
bootstrap CIs from its `bootstrap_test_metrics.csv`; the comparator CSVs carry
both inline. Import-safe from every comparator environment.
"""

from __future__ import annotations

import re
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from src.launch_models import REPO_ROOT, USECASES

RITME_RUNS = REPO_ROOT / "use_cases/ritme_runs/local"
COMPARATORS = REPO_ROOT / "comparators"
AUTOML = REPO_ROOT / "automl"
ORIGINALS = REPO_ROOT / "use_cases/results_original_setup.csv"

RITME_WINNERS = {
    "u1": "u1_xgb_tpe",
    "u2": "u2_linreg_tpe",
    "u3": "u3_logreg_tpe_no_fit",
}

METRICS = {
    "regression": ["rmse_test", "r2_test"],
    "classification": ["auroc_test", "f1_test", "f1_macro_test"],
}
METRIC_LABELS = {
    "rmse_test": "RMSE (↓)",
    "r2_test": "R² (↑)",
    "auroc_test": "AUROC (↑)",
    "f1_test": "F1 (↑)",
    "f1_macro_test": "macro-F1 (↑)",
}
# ritme and the original baselines store the binary AUROC under its objective name.
ALIASES = {"auroc_test": "roc_auc_macro_ovr_test"}

ARM_ORDER = ["ritme", "tabpfn", "tabpfn_top", "automl", "tpot", "maml", "original"]
ARM_LABELS = {
    "ritme": "ritme",
    "tabpfn": "TabPFN (all features)",
    "automl": "auto-sklearn",
    "tpot": "TPOT",
    "maml": "mAML",
    "original": "original study",
}
# Categorical slots in fixed order; the remaining arms are context, in gray.
ARM_COLORS = {"ritme": "#2a78d6", "tabpfn": "#eb6834", "tabpfn_top": "#1baf7a"}
CONTEXT_COLOR = "#8a8984"


def arm_label(arm: str) -> str:
    top = re.fullmatch(r"tabpfn_top(\d+)", arm)
    if top:
        return f"TabPFN (top-{top.group(1)} features)"
    return ARM_LABELS.get(arm, arm)


def _arm_family(arm: str) -> str:
    return "tabpfn_top" if arm.startswith("tabpfn_top") else arm


def _arm_rank(arm: str) -> int:
    family = _arm_family(arm)
    return ARM_ORDER.index(family) if family in ARM_ORDER else len(ARM_ORDER)


def _records_from_row(
    row: pd.Series, metrics: Iterable[str], aliases: dict | None = None
) -> list[dict]:
    """Tidy records for ``metrics`` found in a flat row with optional CI columns."""
    aliases = aliases or {}
    records = []
    for metric in metrics:
        column = metric if metric in row.index else aliases.get(metric)
        if column is None or column not in row.index or pd.isna(row[column]):
            continue
        records.append(
            {
                "metric": metric,
                "point": float(row[column]),
                "ci_low": float(row.get(f"{column}_ci_low", float("nan"))),
                "ci_high": float(row.get(f"{column}_ci_high", float("nan"))),
            }
        )
    return records


def ritme_metrics(usecase: str, experiment: str | None = None) -> list[dict]:
    """Test metrics of one ritme experiment, default the use case's model of record."""
    exp_dir = RITME_RUNS / (experiment or RITME_WINNERS[usecase])
    best = pd.read_csv(exp_dir / "best_metrics.csv", index_col=0)
    boot = pd.read_csv(exp_dir / "bootstrap_test_metrics.csv").set_index("metric")
    base = {
        "usecase": usecase,
        "arm": "ritme",
        "model": best.index[0],
        "source": exp_dir.name,
    }
    records = []
    for metric in METRICS[USECASES[usecase]["task"]]:
        if metric in boot.index:
            values = boot.loc[metric]
            records.append(
                {
                    **base,
                    "metric": metric,
                    "point": float(values["point"]),
                    "ci_low": float(values["ci_low"]),
                    "ci_high": float(values["ci_high"]),
                }
            )
        else:
            for rec in _records_from_row(best.iloc[0], [metric], ALIASES):
                records.append({**base, **rec})
    return records


def comparator_metrics(usecase: str) -> list[dict]:
    """Test metrics of every comparator arm and the original study for a use case."""
    task = USECASES[usecase]["task"]
    files = {
        path.stem.removeprefix(f"{usecase}_").removesuffix("_metrics"): path
        for path in sorted(COMPARATORS.glob(f"{usecase}_*_metrics.csv"))
    }
    automl = AUTOML / f"{usecase}_metrics.csv"
    if automl.exists():
        files["automl"] = automl

    records = []
    for arm, path in files.items():
        row = pd.read_csv(path).iloc[0]
        model = next(
            (row[c] for c in ("restricted_model", "winner") if c in row.index), ""
        )
        base = {"usecase": usecase, "arm": arm, "model": model, "source": path.name}
        records += [{**base, **rec} for rec in _records_from_row(row, METRICS[task])]

    if ORIGINALS.exists():
        originals = pd.read_csv(ORIGINALS)
        match = originals[originals["experiment"] == f"{usecase}_original"]
        if not match.empty:
            row = match.iloc[0]
            base = {
                "usecase": usecase,
                "arm": "original",
                "model": row["model"],
                "source": ORIGINALS.name,
            }
            records += [
                {**base, **rec}
                for rec in _records_from_row(row, METRICS[task], ALIASES)
            ]
    return records


def collect(usecases: Iterable[str] = ("u1", "u2", "u3")) -> pd.DataFrame:
    """Tidy frame: one row per (use case, arm, metric) with point and 95% CI."""
    records = []
    for usecase in usecases:
        records += ritme_metrics(usecase)
        records += comparator_metrics(usecase)
    tidy = pd.DataFrame(records)
    tidy["arm_rank"] = tidy["arm"].map(_arm_rank)
    return tidy.sort_values(["usecase", "arm_rank", "arm"]).drop(columns="arm_rank")


def _format_value(row: pd.Series) -> str:
    text = f"{row['point']:.3f}"
    if pd.notna(row["ci_low"]) and pd.notna(row["ci_high"]):
        text += f" [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
    return text


def format_table(tidy: pd.DataFrame) -> pd.DataFrame:
    """One row per (use case, arm); metrics as ``point [ci_low, ci_high]`` strings."""
    frame = tidy.assign(
        value=tidy.apply(_format_value, axis=1),
        arm_rank=tidy["arm"].map(_arm_rank),
        arm=tidy["arm"].map(arm_label),
        metric=tidy["metric"].map(METRIC_LABELS),
    )
    order = [METRIC_LABELS[m] for metrics in METRICS.values() for m in metrics]
    table = (
        frame.pivot_table(
            index=["usecase", "arm_rank", "arm", "model"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index(level="arm_rank", drop=True)
        .fillna("")
    )
    return table[[c for c in order if c in table.columns]]


def tabpfn_settings(usecases: Iterable[str] = ("u1", "u2", "u3")) -> pd.DataFrame:
    """How each TabPFN run was configured and how long it took."""
    columns = [
        "n_features",
        "n_metadata_features",
        "n_estimators",
        "feature_subsampling",
        "fit_seconds",
        "predict_seconds",
        "n_threads",
        "slurm_job_id",
    ]
    rows = []
    for usecase in usecases:
        for path in sorted(COMPARATORS.glob(f"{usecase}_tabpfn*_metrics.csv")):
            row = pd.read_csv(path).iloc[0]
            arm = path.stem.removeprefix(f"{usecase}_").removesuffix("_metrics")
            rows.append(
                {
                    "usecase": usecase,
                    "arm": arm_label(arm),
                    **{c: row[c] for c in columns if c in row.index},
                }
            )
    return pd.DataFrame(rows).set_index(["usecase", "arm"])


def plot_comparison(tidy: pd.DataFrame, usecases: Iterable[str] = ("u1", "u2", "u3")):
    """Dot-and-whisker panels: the headline test metrics per use case and arm."""
    usecases = list(usecases)
    headline = {
        "regression": ["rmse_test", "r2_test"],
        "classification": ["auroc_test", "f1_test"],
    }
    fig, axes = plt.subplots(
        2, len(usecases), figsize=(5.2 * len(usecases), 5.6), squeeze=False
    )
    for col, usecase in enumerate(usecases):
        metrics = headline[USECASES[usecase]["task"]]
        subset = tidy[tidy["usecase"] == usecase]
        arms = list(dict.fromkeys(subset["arm"]))
        for row_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col]
            data = subset[subset["metric"] == metric].set_index("arm")
            for y, arm in enumerate(arms):
                if arm not in data.index:
                    continue
                values = data.loc[arm]
                color = ARM_COLORS.get(_arm_family(arm), CONTEXT_COLOR)
                if pd.notna(values["ci_low"]):
                    ax.plot(
                        [values["ci_low"], values["ci_high"]],
                        [y, y],
                        color=color,
                        linewidth=2,
                        solid_capstyle="round",
                    )
                ax.scatter(values["point"], y, s=60, color=color, zorder=3)
                ax.annotate(
                    f"{values['point']:.2f}",
                    (values["point"], y),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="#52514e",
                )
            ax.set_yticks(range(len(arms)))
            ax.set_yticklabels([arm_label(a) for a in arms])
            ax.invert_yaxis()
            ax.set_xlabel(METRIC_LABELS[metric])
            ax.grid(axis="x", color="#e5e4e0", linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row_idx == 0:
                ax.set_title(f"Use case {usecase[1:]}", loc="left", fontsize=11)
    families = {}
    for arm in dict.fromkeys(tidy["arm"]):
        family = _arm_family(arm)
        label = arm_label(arm) if family in ARM_COLORS else "other arms"
        families.setdefault(label, ARM_COLORS.get(family, CONTEXT_COLOR))
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=label)
        for label, color in families.items()
    ]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
