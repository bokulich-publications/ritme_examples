"""Collect B2 results into one tidy CSV: one row per (method, cores, seed).

Configurations explored and best validation RMSE come from each method's
own output (ritme: mlflow_logs.csv; auto-sklearn: the per-run metrics
JSON); peak RSS, CPU time and elapsed time come from sacct.

Usage: python -m benchmarking.collect_b2 [--smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmarking.launch_b2 import benchmark_name
from benchmarking.common import (
    REPO_ROOT,
    DATA_DIR,
    SACCT_COLUMNS,
    load_manifests,
    query_sacct,
)


def ritme_run_summary(experiment_dir: Path) -> dict:
    """Configurations explored and best validation RMSE from a trial log.

    A trial ASHA stops early still reports the running K-fold mean
    (`rmse_val_mean`) but never the final `rmse_val`. Such a trial *was*
    evaluated and steered the search, so it counts as explored; its
    partial-fold mean is not comparable to a full 5-fold one, so the best
    score is taken over full-fold trials only. Builds that never prune
    (every B2 baseline run) give identical numbers under both definitions.
    """
    df = pd.read_csv(experiment_dir / "mlflow_logs.csv")
    finished = df[df["status"] == "FINISHED"]
    mean_col = "metrics.rmse_val_mean"
    has_estimate = finished["metrics.rmse_val"].notna()
    if mean_col in finished:
        has_estimate |= finished[mean_col].notna()
    explored = finished[has_estimate]
    full = finished[finished["metrics.rmse_val"].notna()]
    return {
        "n_configs": int(len(explored)),
        "n_configs_full": int(len(full)),
        "n_configs_pruned": int(len(explored) - len(full)),
        "best_rmse_val": full["metrics.rmse_val"].min(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--ritme-label", default="ritme")
    args = p.parse_args()
    benchmark = benchmark_name(args.ritme_label, args.smoke)

    manifests = load_manifests(benchmark)
    if not manifests:
        raise SystemExit(f"No {benchmark} manifests found; launch first.")

    rows = []
    for manifest in manifests:
        runs_root = REPO_ROOT / manifest["params"]["runs_root"]
        for job in manifest["jobs"]:
            row = {
                "method": job["method"],
                "cores": job["cores"],
                "seed": job["seed"],
                "job_id": job["job_id"],
                "budget_s": manifest["params"]["time_budget_s"],
            }
            if job["method"] != "automl":
                exp_dir = runs_root / "ritme" / job["experiment_tag"]
                if not (exp_dir / "mlflow_logs.csv").exists():
                    print(f"[warn] no mlflow_logs.csv yet in {exp_dir}; skipping")
                    continue
                row.update(ritme_run_summary(exp_dir))
            else:
                metrics_file = (
                    runs_root
                    / "automl"
                    / f"automl_c{job['cores']}_s{job['seed']}_metrics.json"
                )
                if not metrics_file.exists():
                    print(f"[warn] no metrics yet: {metrics_file}; skipping")
                    continue
                metrics = json.loads(metrics_file.read_text())
                if metrics["slurm_job_id"] != job["job_id"]:
                    print(
                        f"[warn] {metrics_file.name} was written by job "
                        f"{metrics['slurm_job_id']}, manifest says "
                        f"{job['job_id']}; using the file on disk"
                    )
                    row["job_id"] = metrics["slurm_job_id"]
                row.update(
                    {
                        "n_configs": metrics["n_configs_success"],
                        "best_rmse_val": metrics["best_rmse_val"],
                        "budget_s": metrics["budget_s"],
                    }
                )
            rows.append(row)

    if not rows:
        raise SystemExit("No finished runs found.")

    # A run's outputs are keyed by (method, cores, seed), so a relaunched
    # point appears in more than one manifest. Keep the newest entry, whose
    # job id is the one that produced the outputs on disk.
    summary = pd.DataFrame(rows).drop_duplicates(
        subset=["method", "cores", "seed"], keep="last"
    )
    sacct = pd.DataFrame(query_sacct(summary["job_id"].tolist()), columns=SACCT_COLUMNS)
    summary = summary.merge(sacct, on="job_id", how="left")
    summary["max_rss_gb"] = summary["max_rss_mb"] / 1024
    summary["total_cpu_h"] = summary["total_cpu_s"] / 3600

    out_dir = DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{benchmark}_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(summary)} runs)")


if __name__ == "__main__":
    main()
