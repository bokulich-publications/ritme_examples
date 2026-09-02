"""Collect B4 into one tidy CSV: one row per (usecase, method, cores, seed).

Configurations explored and best validation score come from each arm's own
output; CPU time, elapsed and peak RSS from sacct. Every row carries the
metric it reports (`rmse` for U1/U2, `roc_auc` for U3) and its direction, and
a `n_configs_source` / `best_val_source` telling where the number came from --
TPOT runs that died mid-search are counted from the job log and scored from
their checkpoints, and are flagged as such.

The (U1, ritme) cell is B2's, read from `runs/b2` with B2's job ids. mAML
rows count only the grid configurations that completed inside the budget
(`completed_at_s <= budget_s`), so a run killed at walltime mid-grid is exact.

Usage: python -m benchmarking.collect_comparators [--smoke]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.launch_models import USECASES

from benchmarking.common import (
    B4_REUSE_FROM_B2,
    B4_TASK_METRIC,
    REPO_ROOT,
    DATA_DIR,
    RUNS_DIR,
    SACCT_COLUMNS,
    load_manifests,
    query_sacct,
)

# ritme reports `<metric>_val_mean` from its K-fold aggregate and the bare
# `<metric>_val` from single-report paths; the classification metric carries
# ritme's own name.
RITME_VAL_COLUMNS = {
    "rmse": ["metrics.rmse_val_mean", "metrics.rmse_val"],
    "roc_auc": ["metrics.roc_auc_macro_ovr_val_mean", "metrics.roc_auc_macro_ovr_val"],
}

# TPOT's tqdm line survives in the job log when the process dies and carries
# the number of individuals evaluated so far. It counts every individual the
# GP proposes, while `evaluated_individuals_` deduplicates re-proposals, so on
# clean manuscript runs the log ran +2% (u3) to +26% (u2) above the exact
# count -- an upper bound with no usable correction factor, kept in its own
# column so it is never compared against exact counts. tqdm switches format
# once the counter passes the initial total, so both forms are matched:
#     Optimization Progress:  98%|####| 6303/6400 [...]
#     Optimization Progress: 6401pipeline [...]
# launch_comparators opens each log with --open-mode=truncate, so a file holds exactly
# one attempt and the last match is the run's final count.
TPOT_PROGRESS_RES = (
    re.compile(r"Optimization Progress:[^|]*\|[^|]*\|\s*(\d+)/\d+"),
    re.compile(r"Optimization Progress:\s*(\d+)pipeline"),
)
TPOT_CHECKPOINT_SCORE_RE = re.compile(
    r"Average CV score on the training set was:\s*(-?[0-9.]+)"
)


def _metric_spec(usecase: str) -> dict:
    return B4_TASK_METRIC[USECASES[usecase]["task"]]


def ritme_summary(run_dir: Path, usecase: str) -> Optional[dict]:
    log = run_dir / "mlflow_logs.csv"
    if not log.exists():
        return None
    frame = pd.read_csv(log)
    spec = _metric_spec(usecase)
    mean_col, bare_col = RITME_VAL_COLUMNS[spec["metric"]]
    present = [c for c in (mean_col, bare_col) if c in frame]
    if not present:
        print(f"[warn] {log}: no validation column for {spec['metric']}")
        return None
    finished = frame[frame["status"] == "FINISHED"]
    # Explored = any validation estimate, including ASHA-pruned trials that
    # report only the running K-fold mean; scored = full-fold trials only, so
    # the best score stays a full K-fold mean whatever the build prunes.
    explored = finished[finished[present].notna().any(axis=1)]
    score_col = bare_col if bare_col in frame else mean_col
    full = finished[finished[score_col].notna()]
    values = full[score_col]
    if values.empty:
        return None
    return {
        "n_configs": int(len(explored)),
        "n_configs_full": int(len(full)),
        "n_configs_pruned": int(len(explored) - len(full)),
        "n_configs_source": "mlflow_logs",
        "best_val": float(values.min() if spec["mode"] == "min" else values.max()),
        "best_val_source": score_col,
    }


def automl_summary(run_dir: Path) -> Optional[dict]:
    files = sorted(run_dir.glob("automl_*_metrics.json"))
    if not files:
        return None
    metrics = json.loads(files[-1].read_text())
    return {
        "n_configs": metrics["n_configs_success"],
        "n_configs_source": "runhistory",
        "best_val": metrics["best_val"],
        "best_val_source": "runhistory",
        "file_job_id": metrics.get("slurm_job_id"),
    }


def _tpot_log_count(log: Path) -> Optional[int]:
    if not log.exists():
        return None
    counts = []
    for line in log.read_text(errors="ignore").splitlines():
        for pattern in TPOT_PROGRESS_RES:
            counts.extend(int(m) for m in pattern.findall(line))
    return counts[-1] if counts else None


def _tpot_checkpoint_best(checkpoints: Path, mode: str) -> Optional[float]:
    """Best CV score across the pareto-front checkpoints. TPOT scores
    regression as negative RMSE, so the sign is undone for `min` metrics."""
    scores = []
    for path in glob.glob(str(checkpoints / "pipeline_gen_*.py")):
        match = TPOT_CHECKPOINT_SCORE_RE.search(Path(path).read_text(errors="ignore"))
        if match:
            scores.append(float(match.group(1)))
    if not scores:
        return None
    # TPOT maximises its scorer; for regression the scorer is neg-RMSE.
    best_neg = max(scores)
    return -best_neg if mode == "min" else best_neg


def tpot_summary(run_dir: Path, usecase: str, log: Path) -> Optional[dict]:
    spec = _metric_spec(usecase)
    metrics_file = run_dir / f"{usecase}_tpot_metrics.csv"
    configs_file = run_dir / f"{usecase}_tpot_configs.csv"
    if metrics_file.exists() and configs_file.exists():
        metrics = pd.read_csv(metrics_file).iloc[0]
        configs = pd.read_csv(configs_file)
        scores = configs["internal_cv_score"].dropna()
        # internal_cv_score is TPOT's maximised scorer: roc_auc, or neg-RMSE.
        best = scores.max()
        return {
            "n_configs": int(metrics["n_configs_evaluated"]),
            "n_configs_source": "evaluated_individuals",
            "best_val": float(-best if spec["mode"] == "min" else best),
            "best_val_source": "configs_csv",
            "file_job_id": (
                str(int(metrics["slurm_job_id"]))
                if pd.notna(metrics.get("slurm_job_id"))
                else None
            ),
        }
    # No clean exit: recover what the log and checkpoints hold.
    count = _tpot_log_count(log)
    best = _tpot_checkpoint_best(run_dir / "checkpoints", spec["mode"])
    if count is None and best is None:
        return None
    return {
        "n_configs": None,
        "n_configs_source": None,
        "n_configs_upper_bound": count,
        "best_val": best,
        "best_val_source": "checkpoint" if best is not None else None,
        "crashed": True,
    }


def _b2_job_ids() -> dict[str, str]:
    """experiment_tag -> job id for B2's ritme runs, newest manifest winning."""
    ids: dict[str, str] = {}
    for manifest in load_manifests("b2"):
        for job in manifest["jobs"]:
            if job["method"] == "ritme":
                ids[job["experiment_tag"]] = job["job_id"]
    return ids


def reused_rows(benchmark: str) -> list[dict]:
    if benchmark != "b4":
        return []
    b2_ids = _b2_job_ids()
    rows = []
    for (usecase, method), pattern in B4_REUSE_FROM_B2.items():
        spec = _metric_spec(usecase)
        for run_dir in sorted(
            (RUNS_DIR / "b2" / method).glob(pattern.format(cores="*", seed="*"))
        ):
            match = re.search(r"_c(\d+)_s(\d+)$", run_dir.name)
            if not match:
                continue
            summary = ritme_summary(run_dir, usecase)
            if summary is None:
                continue
            rows.append(
                {
                    "usecase": usecase,
                    "method": method,
                    "cores": int(match.group(1)),
                    "seed": int(match.group(2)),
                    "job_id": b2_ids.get(run_dir.name),
                    "metric": spec["metric"],
                    "mode": spec["mode"],
                    "reused_from": "b2",
                    **summary,
                }
            )
    return rows


def maml_summary(run_dir: Path, budget_s: int) -> Optional[dict]:
    files = sorted(run_dir.glob("maml_*_configs.csv"))
    if not files:
        return None
    configs = pd.read_csv(files[-1])
    within = configs[configs["completed_at_s"] <= budget_s]
    scores = within["mean_cv_roc_auc"].dropna()
    metrics_files = sorted(run_dir.glob("maml_*_metrics.json"))
    metrics = json.loads(metrics_files[-1].read_text()) if metrics_files else {}
    return {
        "n_configs": int(len(within)),
        "n_configs_source": "grid_prefix",
        "best_val": float(scores.max()) if len(scores) else None,
        "best_val_source": "configs_csv",
        "grid_total": metrics.get("grid_total"),
        "grid_complete": metrics.get("grid_complete"),
        "file_job_id": metrics.get("slurm_job_id"),
        # No metrics file means the job was killed mid-grid; the count above
        # is still exact because it only uses grids that finished in budget.
        "crashed": not bool(metrics_files),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    benchmark = "b4_smoke" if args.smoke else "b4"

    manifests = load_manifests(benchmark)
    if not manifests and benchmark == "b4_smoke":
        raise SystemExit(f"No {benchmark} manifests found; launch first.")

    rows = reused_rows(benchmark)
    for manifest in manifests:
        runs_root = REPO_ROOT / manifest["params"]["runs_root"]
        for job in manifest["jobs"]:
            method, usecase = job["method"], job["usecase"]
            run_dir = runs_root / method / job["experiment_tag"]
            log = runs_root / method / "logs" / f"{job['experiment_tag']}_out.txt"
            if method == "ritme":
                summary = ritme_summary(run_dir, usecase)
            elif method == "automl":
                summary = automl_summary(run_dir)
            elif method == "maml":
                summary = maml_summary(run_dir, manifest["params"]["time_budget_s"])
            else:
                summary = tpot_summary(run_dir, usecase, log)
            if summary is None:
                print(f"[warn] no outputs yet for {job['experiment_tag']}; skipping")
                continue
            file_job_id = summary.pop("file_job_id", None)
            if file_job_id and file_job_id != job["job_id"]:
                print(
                    f"[warn] {job['experiment_tag']}: outputs written by job "
                    f"{file_job_id}, manifest says {job['job_id']}; using the file"
                )
            spec = _metric_spec(usecase)
            rows.append(
                {
                    "usecase": usecase,
                    "method": method,
                    "cores": job["cores"],
                    "seed": job["seed"],
                    "job_id": file_job_id or job["job_id"],
                    "budget_s": manifest["params"]["time_budget_s"],
                    "metric": spec["metric"],
                    "mode": spec["mode"],
                    **summary,
                }
            )

    if not rows:
        raise SystemExit("No finished runs found.")

    # A relaunched point appears in more than one manifest; keep the newest.
    summary = pd.DataFrame(rows).drop_duplicates(
        subset=["usecase", "method", "cores", "seed"], keep="last"
    )

    job_ids = summary["job_id"].dropna().astype(str).tolist()
    sacct = pd.DataFrame(query_sacct(job_ids), columns=SACCT_COLUMNS)
    summary["job_id"] = summary["job_id"].astype("string")
    summary = summary.merge(sacct, on="job_id", how="left")
    summary["max_rss_gb"] = summary["max_rss_mb"] / 1024
    summary["total_cpu_h"] = summary["total_cpu_s"] / 3600

    out_dir = DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{benchmark}_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(summary)} rows)")
    counts = summary.groupby(["usecase", "method"]).size().unstack(fill_value=0)
    print(counts.to_string())


if __name__ == "__main__":
    main()
