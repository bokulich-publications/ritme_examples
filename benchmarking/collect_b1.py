"""Collect B1 trial-level results into tidy CSVs for plotting.

Reads each run's mlflow_logs.csv, keeps finished trials, computes the
running minimum of the validation RMSE over wall-clock time, and joins the
SLURM accounting of every job.

Usage: python -m benchmarking.collect_b1 [--smoke]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from benchmarking.common import (
    REPO_ROOT,
    DATA_DIR,
    SACCT_COLUMNS,
    load_manifests,
    query_sacct,
)


def load_trials(experiment_dir: Path, metric: str = "metrics.rmse_val") -> pd.DataFrame:
    """Finished trials of one run, sorted by end time, with running minimum."""
    df = pd.read_csv(experiment_dir / "mlflow_logs.csv")
    df["start_time"] = pd.to_datetime(
        df["start_time"], format="ISO8601", errors="coerce"
    )
    df["end_time"] = pd.to_datetime(df["end_time"], format="ISO8601", errors="coerce")
    t0 = df["start_time"].min()
    df = df[(df["status"] == "FINISHED") & df[metric].notna()].copy()
    df = df.sort_values("end_time").reset_index(drop=True)
    df["trial_end_s"] = (df["end_time"] - t0).dt.total_seconds()
    df["running_min"] = df[metric].cummin()
    return df[["trial_end_s", metric, "running_min"]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    benchmark = "b1_smoke" if args.smoke else "b1"

    manifests = load_manifests(benchmark)
    if not manifests:
        raise SystemExit(f"No {benchmark} manifests found; launch first.")

    trials_frames, job_records = [], []
    for manifest in manifests:
        logs_dir = REPO_ROOT / manifest["params"]["logs_dir"]
        for job in manifest["jobs"]:
            exp_dir = logs_dir / job["experiment_tag"]
            if not (exp_dir / "mlflow_logs.csv").exists():
                print(f"[warn] no mlflow_logs.csv yet in {exp_dir}; skipping")
                continue
            trials = load_trials(exp_dir)
            trials.insert(0, "sampler", job["sampler"])
            trials.insert(1, "seed", job["seed"])
            trials_frames.append(trials)
            job_records.append(job)

    if not trials_frames:
        raise SystemExit("No finished runs found.")

    out_dir = DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_trials = pd.concat(trials_frames, ignore_index=True)
    all_trials.to_csv(out_dir / f"{benchmark}_trials.csv", index=False)

    sacct = pd.DataFrame(
        query_sacct([j["job_id"] for j in job_records]), columns=SACCT_COLUMNS
    )
    jobs = pd.DataFrame(job_records).merge(sacct, on="job_id", how="left")
    jobs.to_csv(out_dir / f"{benchmark}_jobs.csv", index=False)

    print(
        f"Wrote {out_dir / f'{benchmark}_trials.csv'} "
        f"({len(all_trials)} trials, {len(job_records)} runs)"
    )


if __name__ == "__main__":
    main()
