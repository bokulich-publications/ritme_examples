"""Launch B2: resource-efficiency sweep over allocated CPU cores on U1.

One SLURM job per (method, cores, seed) at a fixed wall-clock search budget,
all pinned to one node type. Methods:

Each method fills the allocation the way it is designed to:

- ritme: xgb only, concurrent trials = cores / 4 at 4 CPUs each, search-only
  template.
- auto-sklearn: gradient_boosting only, n_jobs = cores single-threaded
  workers, grouped 5-fold CV on host_id (see benchmarking/automl_b2.py), run
  in the `autosklearn` env.

Usage (from the repo root, ritme_usecases env):
    python -m benchmarking.launch_b2 [--methods ritme automl] [--smoke] [--dry-run]
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src import cluster_config
from src.launch_models import REPO_ROOT, USECASES, submit_model

from benchmarking.common import (
    ensure_launcher_env,
    env_python,
    B2_AUTOML_MEMORY_LIMIT_MB,
    B2_AUTOML_MODEL,
    B2_AUTOML_N_JOBS_PER_CORE,
    B2_AUTOML_THREADS_PER_JOB,
    B2_CORES,
    B2_CPUS_PER_TRIAL,
    B2_MEM_PER_CPU_MB,
    B2_TIME_BUDGET_S,
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

# Walltime = search budget + buffer for setup, trials still running at the
# budget cutoff (ritme lets in-flight trials finish) and log extraction.
RITME_WALLTIME_BUFFER_S = 10800
# auto-sklearn's budget covers the search only; cluster startup, ensemble
# handling and shutdown add real time on top, and that overhead is both large
# (tens of minutes) and variable -- one run of 18 exceeded a 90 minute buffer
# and was killed inside fit().
AUTOML_WALLTIME_BUFFER_S = 12600

# The auto-sklearn arm is invoked through its env's interpreter rather than
# `mamba run`: --no-capture-output is broken in mamba 2.0.4 here
# (`exec: --: invalid option`), and without it the wrapper relays the child's
# output through itself, which hung this arm when that output was consumed
# through a pipe. Plain `mamba run` does work under `sbatch --output=` (see
# src/launch_automl.py); calling the interpreter directly avoids both.
#
# The interpreter is resolved from the env *name*, which is not
# site-specific, so no absolute path under anyone's home or project
# directory is stored in the repo. Override with RITME_AUTOSKLEARN_PYTHON
# (or an `autosklearn_python` key in .cluster.json) when the env lives
# somewhere the sibling-env rule cannot find.
AUTOSKLEARN_ENV_NAME = "autosklearn"


def autosklearn_python() -> str:
    """Path to the auto-sklearn env's interpreter (see ``common.env_python``)."""
    return env_python(AUTOSKLEARN_ENV_NAME, "autosklearn_python")


# BLAS/OpenMP threads per auto-sklearn worker. These must be set explicitly:
# at their defaults the libraries size themselves from the node's core count,
# which oversubscribes every worker and aborts the 128-core runs. See
# "Design notes" in benchmarking/README.md for the reasoning.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", default=["ritme", "automl"])
    p.add_argument("--cores", type=int, nargs="+", default=B2_CORES)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--time-budget-s", type=int, default=B2_TIME_BUDGET_S)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run (10 min budget, 8 cores, 1 seed) to validate the setup",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--max-trial-failure-rate",
        type=float,
        default=None,
        help=(
            "Override ritme's post-search failure-rate policy for the ritme arm. "
            "The default (0.005) discards a run's trial log when a single trial "
            "actor dies at 4 cores, where the allocation is memory-bound; "
            "raising it changes only whether the log is written, not the search."
        ),
    )
    p.add_argument(
        "--ritme-label",
        default="ritme",
        help=(
            "Method label for the ritme arm, e.g. `ritme_efficient` to run the "
            "same sweep with another ritme build. Any label other than `ritme` "
            "keys its own manifests, runs/ and results/ directories, and must be "
            "launched from the conda env of the same name so the jobs run that "
            "build."
        ),
    )
    return p.parse_args()


def _ensure_parquet_splits() -> None:
    """Materialize parquet copies of the U1 splits for the auto-sklearn env.

    Its NumPy 1.x cannot read pickles written under NumPy 2.x; converting
    once here also avoids concurrent jobs racing the on-the-fly conversion
    in `src.generic_automl._read_split`.
    """
    splits_dir = REPO_ROOT / USECASES["u1"]["data_splits"]
    for name in ("train_val", "test"):
        parquet = splits_dir / f"{name}.parquet"
        if not parquet.exists():
            pd.read_pickle(splits_dir / f"{name}.pkl").to_parquet(parquet, index=True)
            print(f"Converted {parquet}")


RITME_CONFIG_OVERRIDES: dict = {}


def _submit_ritme(
    cores: int, seed: int, budget_s: int, tag: str, logs_dir: Path
) -> str:
    proc = submit_model(
        "u1",
        "xgb",
        sampler="tpe",
        logs_dir=logs_dir,
        template=SEARCH_ONLY_TEMPLATE,
        cpus=cores,
        mem_per_cpu_mb=B2_MEM_PER_CPU_MB,
        max_concurrent_trials=cores // B2_CPUS_PER_TRIAL,
        slurm_time=slurm_time(budget_s + RITME_WALLTIME_BUFFER_S),
        slurm_account=cluster_config.slurm_account(),
        sbatch_extra=constraint_flags(),
        config_overrides={
            "experiment_tag": tag,
            "seed_data": seed,
            "seed_model": seed,
            "time_budget_s": budget_s,
            **RITME_CONFIG_OVERRIDES,
        },
    )
    return parse_job_id(proc.stdout)


def _submit_automl(
    cores: int, seed: int, budget_s: int, job_name: str, out_dir: Path
) -> str:
    out_log = out_dir / "logs" / f"{job_name}_out.txt"
    out_log.parent.mkdir(parents=True, exist_ok=True)
    inner = [
        "env",
        *(f"{var}={B2_AUTOML_THREADS_PER_JOB}" for var in THREAD_ENV_VARS),
        autosklearn_python(),
        "-u",
        "-m",
        "benchmarking.automl_b2",
        "--budget-s",
        str(budget_s),
        "--cores",
        str(cores),
        "--n-jobs",
        str(cores * B2_AUTOML_N_JOBS_PER_CORE),
        "--threads-per-job",
        str(B2_AUTOML_THREADS_PER_JOB),
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
        "--memory-limit-mb",
        str(B2_AUTOML_MEMORY_LIMIT_MB),
        "--restricted-model",
        B2_AUTOML_MODEL,
    ]
    cmd = [
        "sbatch",
        *account_flags(),
        f"--job-name={job_name}",
        "--ntasks=1",
        f"--cpus-per-task={cores}",
        f"--mem-per-cpu={B2_MEM_PER_CPU_MB}",
        f"--time={slurm_time(budget_s + AUTOML_WALLTIME_BUFFER_S)}",
        *constraint_flags(),
        "--tmp=8192",
        f"--output={out_log}",
        # Truncate rather than append: a rerun only happens after its metrics
        # file is deleted, and one attempt per log keeps them unambiguous.
        "--open-mode=truncate",
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


def benchmark_name(ritme_label: str, smoke: bool) -> str:
    """`b2`, `b2_smoke`, or `b2_<label>[_smoke]` for an alternative ritme build."""
    base = "b2" if ritme_label == "ritme" else f"b2_{ritme_label}"
    return f"{base}_smoke" if smoke else base


def _check_ritme_build(label: str) -> dict:
    """Refuse to run a relabelled ritme arm from the wrong env, and record
    which build the jobs will get. The label is the claim; the interpreter's
    env is the fact, and the two must agree before 18 jobs are submitted."""
    import ritme

    env_name = Path(sys.executable).resolve().parents[1].name
    if label != "ritme" and env_name != label:
        raise SystemExit(
            f"--ritme-label {label!r} but this launcher runs from env "
            f"{env_name!r} ({sys.executable}); the jobs would run that build "
            f"under the wrong name. Launch from the {label!r} env."
        )
    return {"ritme_label": label, "ritme_version": ritme.__version__, "env": env_name}


def main() -> None:
    args = parse_args()
    ensure_launcher_env()
    build = _check_ritme_build(args.ritme_label) if not args.dry_run else {}
    if args.max_trial_failure_rate is not None:
        RITME_CONFIG_OVERRIDES["max_trial_failure_rate"] = args.max_trial_failure_rate
    benchmark = benchmark_name(args.ritme_label, args.smoke)
    budget_s = 600 if args.smoke else args.time_budget_s
    cores_sweep = [8] if args.smoke else args.cores
    seeds = [args.seeds[0]] if args.smoke else args.seeds
    prefix = f"{'b2smoke' if args.smoke else 'b2'}"
    if args.ritme_label != "ritme":
        prefix = f"{prefix}_{args.ritme_label}"
    runs_root = RUNS_DIR / benchmark
    if args.ritme_label != "ritme" and "automl" in args.methods:
        # The comparator arm is unchanged by a ritme build; B2's runs stand.
        print("[info] relabelled ritme arm: skipping automl (reuse B2's)")
        args.methods = [m for m in args.methods if m != "automl"]

    invalid = [c for c in cores_sweep if c % B2_CPUS_PER_TRIAL != 0]
    if invalid:
        raise ValueError(
            f"Core counts {invalid} are not multiples of "
            f"B2_CPUS_PER_TRIAL={B2_CPUS_PER_TRIAL}."
        )
    if "automl" in args.methods and not args.dry_run:
        _ensure_parquet_splits()

    jobs = []
    for cores in cores_sweep:
        for seed in seeds:
            if "ritme" in args.methods:
                tag = (
                    f"{prefix}_c{cores}_s{seed}"
                    if args.ritme_label != "ritme"
                    else f"{prefix}_u1_xgb_c{cores}_s{seed}"
                )
                exp_dir = runs_root / "ritme" / tag
                if (exp_dir / "mlflow_logs.csv").exists():
                    print(f"[skip] trial log exists: {exp_dir}")
                elif exp_dir.exists():
                    # ritme refuses to reuse an experiment_tag directory, so an
                    # incomplete run needs clearing before it can be redone.
                    print(f"[skip] incomplete run; remove to relaunch: {exp_dir}")
                elif args.dry_run:
                    print(f"[dry-run] would submit ritme {tag}")
                else:
                    job_id = _submit_ritme(
                        cores, seed, budget_s, tag, runs_root / "ritme"
                    )
                    jobs.append(
                        {
                            "job_id": job_id,
                            "method": args.ritme_label,
                            "cores": cores,
                            "seed": seed,
                            "experiment_tag": tag,
                        }
                    )
            if "automl" in args.methods:
                job_name = f"{prefix}_automl_c{cores}_s{seed}"
                metrics_file = (
                    runs_root / "automl" / f"automl_c{cores}_s{seed}_metrics.json"
                )
                if metrics_file.exists():
                    print(f"[skip] metrics exist: {metrics_file}")
                elif args.dry_run:
                    print(f"[dry-run] would submit automl {job_name}")
                else:
                    job_id = _submit_automl(
                        cores, seed, budget_s, job_name, runs_root / "automl"
                    )
                    jobs.append(
                        {
                            "job_id": job_id,
                            "method": "automl",
                            "cores": cores,
                            "seed": seed,
                            "job_name": job_name,
                        }
                    )

    if jobs:
        manifest = write_manifest(
            benchmark,
            params={
                **build,
                "ritme_config_overrides": RITME_CONFIG_OVERRIDES,
                "usecase": "u1",
                "time_budget_s": budget_s,
                "cores": cores_sweep,
                "seeds": seeds,
                "mem_per_cpu_mb": B2_MEM_PER_CPU_MB,
                "cpus_per_trial": B2_CPUS_PER_TRIAL,
                "automl_memory_limit_mb": B2_AUTOML_MEMORY_LIMIT_MB,
                "automl_model": B2_AUTOML_MODEL,
                "automl_n_jobs_per_core": B2_AUTOML_N_JOBS_PER_CORE,
                "automl_threads_per_job": B2_AUTOML_THREADS_PER_JOB,
                "runs_root": repo_relative(runs_root),
            },
            jobs=jobs,
        )
        print(f"Submitted {len(jobs)} jobs; manifest: {manifest}")


if __name__ == "__main__":
    main()
