"""Launch B1: TPE vs. random search on U1/xgb at an identical time budget.

One SLURM job per (sampler, seed); everything except
``optuna_searchspace_sampler`` and the seeds is held constant. Runs use the
search-only template so the jobs end after ``find-best-model-config``.

Usage (from the repo root, ritme_usecases env):
    python -m benchmarking.launch_b1 [--smoke] [--dry-run]
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from src import cluster_config
from src.launch_models import REPO_ROOT, submit_model

from benchmarking.common import (
    ensure_launcher_env,
    B1_CPUS,
    B1_MAX_CONCURRENT_TRIALS,
    B1_MEM_PER_CPU_MB,
    B1_SAMPLERS,
    B1_TIME_BUDGET_S,
    RUNS_DIR,
    SEARCH_ONLY_TEMPLATE,
    SEEDS,
    account_flags,
    constraint_flags,
    parse_job_id,
    repo_relative,
    slurm_time,
    write_manifest,
)

# Buffer on top of time_budget_s for data loading, trial teardown and the
# mlflow_logs.csv extraction at the end of the search.
WALLTIME_BUFFER_S = 5400


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--time-budget-s", type=int, default=B1_TIME_BUDGET_S)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run (10 min budget, 8 CPUs, 1 seed) to validate the setup",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be submitted without calling sbatch",
    )
    return p.parse_args()


def _submit_warmup_job(logs_dir: Path) -> str:
    """Compute the adaptive TPE warm-up length in a small SLURM job."""
    out_log = logs_dir / "logs" / "b1_warmup_out.txt"
    out_log.parent.mkdir(parents=True, exist_ok=True)
    inner = [
        "python",
        "-m",
        "benchmarking.compute_warmup",
        "--usecase",
        "u1",
        "--model-type",
        "xgb",
        "--out",
        str(logs_dir / "warmup.json"),
    ]
    cmd = [
        "sbatch",
        *account_flags(),
        "--job-name=b1_warmup",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--mem-per-cpu=4096",
        "--time=00:30:00",
        f"--output={out_log}",
        "--open-mode=append",
        f"--chdir={REPO_ROOT}",
        f"--wrap={' '.join(shlex.quote(c) for c in inner)}",
    ]
    print("submitting:", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for stream in (proc.stdout, proc.stderr):
        if stream:
            print(stream.strip())
    proc.check_returncode()
    return parse_job_id(proc.stdout)


def main() -> None:
    args = parse_args()
    ensure_launcher_env()
    budget_s = 600 if args.smoke else args.time_budget_s
    cpus = 8 if args.smoke else B1_CPUS
    max_concurrent = 2 if args.smoke else B1_MAX_CONCURRENT_TRIALS
    seeds = [args.seeds[0]] if args.smoke else args.seeds
    tag_prefix = "b1smoke" if args.smoke else "b1"
    logs_dir = RUNS_DIR / ("b1_smoke" if args.smoke else "b1")

    jobs = []
    for sampler in B1_SAMPLERS:
        for seed in seeds:
            tag = f"{tag_prefix}_u1_xgb_{sampler}_s{seed}"
            exp_dir = logs_dir / tag
            if (exp_dir / "mlflow_logs.csv").exists():
                print(f"[skip] trial log exists: {exp_dir}")
                continue
            if exp_dir.exists():
                # ritme refuses to reuse an experiment_tag directory, so an
                # incomplete run needs clearing before it can be redone.
                print(f"[skip] incomplete run; remove to relaunch: {exp_dir}")
                continue
            if args.dry_run:
                print(f"[dry-run] would submit {tag}")
                continue
            proc = submit_model(
                "u1",
                "xgb",
                sampler=sampler,
                logs_dir=logs_dir,
                template=SEARCH_ONLY_TEMPLATE,
                cpus=cpus,
                mem_per_cpu_mb=B1_MEM_PER_CPU_MB,
                max_concurrent_trials=max_concurrent,
                slurm_time=slurm_time(budget_s + WALLTIME_BUFFER_S),
                slurm_account=cluster_config.slurm_account(),
                sbatch_extra=constraint_flags(),
                config_overrides={
                    "experiment_tag": tag,
                    "seed_data": seed,
                    "seed_model": seed,
                    "time_budget_s": budget_s,
                },
            )
            jobs.append(
                {
                    "job_id": parse_job_id(proc.stdout),
                    "experiment_tag": tag,
                    "sampler": sampler,
                    "seed": seed,
                }
            )

    warmup_job_id = None
    if jobs and not (logs_dir / "warmup.json").exists():
        warmup_job_id = _submit_warmup_job(logs_dir)

    if jobs:
        manifest = write_manifest(
            "b1_smoke" if args.smoke else "b1",
            params={
                "usecase": "u1",
                "model_type": "xgb",
                "time_budget_s": budget_s,
                "cpus": cpus,
                "mem_per_cpu_mb": B1_MEM_PER_CPU_MB,
                "max_concurrent_trials": max_concurrent,
                "logs_dir": repo_relative(logs_dir),
                "warmup_job_id": warmup_job_id,
            },
            jobs=jobs,
        )
        print(f"Submitted {len(jobs)} jobs; manifest: {manifest}")


if __name__ == "__main__":
    main()
