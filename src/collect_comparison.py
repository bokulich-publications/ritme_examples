"""Build the ritme-vs-AutoML comparison table for one or all use cases.

Model selection is on the **validation** set, never on test. ritme picks the
best trial within one model type by its own objective
(`ritme.tune_models.TASK_METRICS`: `rmse_val` min for regression,
`roc_auc_macro_ovr_val` max for classification) and reports that trial's test
metrics in `best_metrics.csv`. Selecting *across* model types is done here, on
the same validation metric read from `mlflow_logs.csv`.

Usage (from the repo root, ritme_usecases env):
    python -m src.collect_comparison            # all use cases
    python -m src.collect_comparison --usecase u3
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

from src.launch_models import REPO_ROOT, USECASES

RITME_RUNS = REPO_ROOT / "use_cases/ritme_runs/local"
COMPARATORS = REPO_ROOT / "comparators"
AUTOML = REPO_ROOT / "automl"

# Validation metric per task, mirroring ritme.tune_models.TASK_METRICS.
VAL_METRIC = {
    "regression": ("metrics.rmse_val_mean", "min"),
    "classification": ("metrics.roc_auc_macro_ovr_val_mean", "max"),
}
# Headline test metric per task.
TEST_METRIC = {
    "regression": "rmse_test",
    "classification": "roc_auc_macro_ovr_test",
}

# ritme experiment suffixes and the arm each belongs to. Order matters: the
# first matching suffix wins.
RITME_ARMS = [
    ("_tpe_reduced", "ritme-reduced"),
    ("_tpe_no_fit", "ritme (no fit_result)"),
    ("_tpe_no_enrich", None),  # separate ablation, excluded from this table
    ("_tpe", "ritme"),
]


def _read_csv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _as_float(value: Optional[str]) -> Optional[float]:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


def best_validation_score(exp_dir: Path, task: str) -> tuple[Optional[float], int]:
    """Best validation score over all trials of one ritme experiment."""
    log = exp_dir / "mlflow_logs.csv"
    if not log.exists():
        return None, 0
    column, direction = VAL_METRIC[task]
    scores = [s for s in (_as_float(r.get(column)) for r in _read_csv(log)) if s]
    if not scores:
        return None, 0
    return (min(scores) if direction == "min" else max(scores)), len(scores)


def ritme_rows(usecase: str, task: str) -> list[dict]:
    """One row per ritme experiment of this use case, with its val and test scores."""
    rows = []
    for exp_dir in sorted(RITME_RUNS.glob(f"{usecase}_*")):
        if not (exp_dir / "experiment_config.json").exists():
            continue
        name = exp_dir.name
        if name.startswith(f"{usecase}_dynamic"):
            continue
        arm = next((a for suf, a in RITME_ARMS if name.endswith(suf)), None)
        if arm is None:
            continue
        val, n_trials = best_validation_score(exp_dir, task)
        metrics_path = exp_dir / "best_metrics.csv"
        test = None
        if metrics_path.exists():
            records = _read_csv(metrics_path)
            if records:
                test = _as_float(records[0].get(TEST_METRIC[task]))
        cfg = json.loads((exp_dir / "experiment_config.json").read_text())
        rows.append(
            {
                "arm": arm,
                "experiment": name,
                "model": (cfg.get("ls_model_types") or [""])[0],
                "val": val,
                "test": test,
                "n_trials": n_trials,
            }
        )
    return rows


def comparator_rows(usecase: str, task: str) -> list[dict]:
    """One row per external comparator arm (TPOT, mAML, auto-sklearn)."""
    rows = []
    for path in sorted(COMPARATORS.glob(f"{usecase}_*_metrics.csv")):
        method = path.stem.removeprefix(f"{usecase}_").removesuffix("_metrics")
        records = _read_csv(path)
        if not records:
            continue
        r = records[0]
        rows.append(
            {
                "arm": method,
                "experiment": path.stem.removesuffix("_metrics"),
                "model": r.get("restricted_model") or r.get("best_classifier", ""),
                "val": None,  # comparators use their own internal CV protocol
                "test": _as_float(r.get(TEST_METRIC[task])),
                "n_trials": int(_as_float(r.get("n_configs_evaluated")) or 0),
            }
        )
    automl_path = AUTOML / f"{usecase}_metrics.csv"
    if automl_path.exists():
        records = _read_csv(automl_path)
        if records:
            rows.append(
                {
                    "arm": "auto-sklearn",
                    "experiment": f"{usecase}_automl",
                    "model": "",
                    "val": None,
                    "test": _as_float(records[0].get(TEST_METRIC[task])),
                    "n_trials": 0,
                }
            )
    return rows


def report(usecase: str) -> None:
    task = USECASES[usecase]["task"]
    column, direction = VAL_METRIC[task]
    better = min if direction == "min" else max
    label = TEST_METRIC[task]

    rows = ritme_rows(usecase, task)
    print(f"\n=== {usecase} ({task}) ===")
    print(f"ritme model selection on {column} ({direction})\n")
    print(f"{'experiment':34s} {'arm':22s} {'val':>9s} {'test':>9s} {'trials':>7s}")
    for r in sorted(rows, key=lambda x: (x["arm"], x["experiment"])):
        v = f"{r['val']:.4f}" if r["val"] is not None else "    -"
        t = f"{r['test']:.4f}" if r["test"] is not None else "    -"
        print(
            f"{r['experiment']:34s} {r['arm']:22s} {v:>9s} {t:>9s} {r['n_trials']:7d}"
        )

    print(f"\n--- selected per arm, reported on {label} ---")
    for arm in sorted({r["arm"] for r in rows}):
        candidates = [r for r in rows if r["arm"] == arm and r["val"] is not None]
        if not candidates:
            continue
        win = better(candidates, key=lambda x: x["val"])
        print(
            f"{arm:22s} {win['model']:12s} val={win['val']:.4f} "
            f"test={win['test'] if win['test'] is not None else float('nan'):.4f} "
            f"({win['experiment']})"
        )
    for r in comparator_rows(usecase, task):
        t = f"{r['test']:.4f}" if r["test"] is not None else "    -"
        n = f"{r['n_trials']} configs" if r["n_trials"] else ""
        print(f"{r['arm']:22s} {str(r['model'])[:12]:12s} val=    -    test={t}  {n}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--usecase", default=None, help="u1 | u2 | u3 (default: all)")
    args = p.parse_args()
    for usecase in [args.usecase] if args.usecase else ["u1", "u2", "u3"]:
        report(usecase)


if __name__ == "__main__":
    main()
