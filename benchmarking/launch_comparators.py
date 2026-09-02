"""Launch B4: compute scaling of ritme, auto-sklearn and TPOT on all use cases.

The B2 design -- {4..128} cores x 3 seeds at a fixed 2 h search budget -- run
for U1, U2 and U3 with three arms:

- ritme, on the model its own validation objective selects per use case,
  via `run_ritme_search_only.sh` with cores / CPUS_PER_TRIAL concurrent trials;
- auto-sklearn (`automl_sweep.py`), one single-threaded worker per core,
  restricted to the family closest to ritme's model;
- TPOT (`src.comparator_tpot`, the manuscript comparator worker), `n_jobs =
  cores`, restricted likewise;
- mAML (U3 only): the manuscript grid walked in its published order with
  `GridSearchCV(n_jobs=cores)` until the budget is spent. Its worker,
  maml_sweep.py, is ARCHIVED (archive/code/) and incompatible with the
  reworked src/comparator_maml.py, so this arm refuses to launch.

All arms receive the same metadata enrichment ritme searches over
(`src.launch_automl._read_enrich_with`), so ritme's own configs are aligned to
that list too -- for U3 this drops `fit_result`, a screening readout of the
outcome that the comparator arms exclude.

The (U1, ritme) cell is taken over from B2 unchanged -- same config,
enrichment and allocation -- and is not relaunched.

Usage (repo root, ritme_usecases env):
    python -m benchmarking.launch_comparators [--smoke] [--dry-run]
        [--methods ritme automl tpot] [--usecases u1 u2 u3]
        [--cores 4 8 ...] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from src import cluster_config
from src.launch_automl import _read_base_config, _read_enrich_with, _read_target
from src.launch_models import REPO_ROOT, USECASES, submit_model

from benchmarking.common import (
    ensure_launcher_env,
    B4_AUTOML_MODEL,
    B4_CORES,
    B4_CPUS_PER_TRIAL,
    B4_MAML_ENV_NAME,
    B4_MEM_PER_CPU_MB,
    B4_METHOD_USECASES,
    B4_METHODS,
    B4_REUSE_FROM_B2,
    B4_RITME_MODEL,
    B4_TIME_BUDGET_S,
    B4_TPOT_MODEL,
    B4_USECASES,
    BENCH_DIR,
    RUNS_DIR,
    SEARCH_ONLY_TEMPLATE,
    SEEDS,
    account_flags,
    b4_run_dir,
    b4_tag,
    b4_tpot_max_eval_time_mins,
    constraint_flags,
    env_python,
    parse_job_id,
    repo_relative,
    slurm_time,
    write_manifest,
)
from benchmarking.launch_b2 import (
    AUTOML_WALLTIME_BUFFER_S,
    RITME_WALLTIME_BUFFER_S,
    THREAD_ENV_VARS,
    autosklearn_python,
)

TPOT_ENV_NAME = "tpot_bench"
# TPOT stops at a generation boundary after `max_time_mins`, so the run can
# overshoot by one generation plus the final refit; with the per-evaluation
# cap set above the budget a single slow pipeline can extend that generation
# by up to the cap. Both scale with the budget, so the buffer does too.
TPOT_WALLTIME_MARGIN_S = 3600
MAML_WALLTIME_MARGIN_S = 3600
# Per-worker BLAS/OpenMP threads. auto-sklearn workers were pinned in B2
# already; TPOT's XGBoost workers get the same treatment because at 128
# `n_jobs` an unpinned library sizes itself from the node's core count and
# oversubscribes every worker (the manuscript TPOT runs at 50 cores did not
# pin, which is recorded in the manifest as a difference).
THREADS_PER_WORKER = 1

SMOKE_BUDGET_S = 600
SMOKE_CORES = [4]
SMOKE_SEEDS = [0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--methods", nargs="+", default=B4_METHODS, choices=B4_METHODS)
    p.add_argument("--usecases", nargs="+", default=B4_USECASES, choices=B4_USECASES)
    p.add_argument("--cores", nargs="+", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    return p.parse_args()


def _ritme_hyperparameters(usecase: str) -> dict:
    """The base config's `model_hyperparameters` with enrichment aligned to
    what the comparator arms receive. `submit_model` merges overrides at the
    top level only, so the whole block is passed back."""
    hparams = dict(_read_base_config(usecase).get("model_hyperparameters") or {})
    hparams["data_enrich_with"] = _read_enrich_with(usecase)
    return hparams


def _data_args(usecase: str) -> list[str]:
    spec = USECASES[usecase]
    args = [
        "--usecase",
        usecase,
        "--task",
        spec["task"],
        "--data-splits-folder",
        str(REPO_ROOT / spec["data_splits"]),
        "--path-to-features",
        str(REPO_ROOT / spec["path_ft"]),
        "--path-to-md",
        str(REPO_ROOT / spec["path_md"]),
        "--target",
        _read_target(usecase),
    ]
    for feature in _read_enrich_with(usecase):
        args += ["--enrich-with", feature]
    return args


def _sbatch(
    job_name: str,
    cores: int,
    walltime_s: int,
    out_log: Path,
    inner: list[str],
    raise_ulimits: bool = False,
) -> str:
    out_log.parent.mkdir(parents=True, exist_ok=True)
    wrapped = " ".join(shlex.quote(c) for c in inner)
    if raise_ulimits:
        # One process per core plus forkservers exhausts the per-user
        # defaults at high core counts (as src/run_ritme_model.sh does for Ray).
        wrapped = f"ulimit -u 60000; ulimit -n 524288; {wrapped}"
    cmd = [
        "sbatch",
        *account_flags(),
        f"--job-name={job_name}",
        "--ntasks=1",
        f"--cpus-per-task={cores}",
        f"--mem-per-cpu={B4_MEM_PER_CPU_MB}",
        f"--time={slurm_time(walltime_s)}",
        *constraint_flags(),
        "--tmp=8192",
        f"--output={out_log}",
        # Truncate rather than append: a rerun only happens after its outputs
        # are deleted, and one attempt per log keeps them unambiguous.
        "--open-mode=truncate",
        f"--chdir={REPO_ROOT}",
        f"--wrap={wrapped}",
    ]
    print("submitting:", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for stream in (proc.stdout, proc.stderr):
        if stream:
            print(stream.strip())
    proc.check_returncode()
    return parse_job_id(proc.stdout)


def submit_ritme(
    usecase: str, cores: int, seed: int, budget_s: int, runs_root: Path
) -> str:
    tag = b4_tag(usecase, "ritme", cores, seed)
    proc = submit_model(
        usecase,
        B4_RITME_MODEL[usecase],
        sampler="tpe",
        logs_dir=runs_root / "ritme",
        template=SEARCH_ONLY_TEMPLATE,
        cpus=cores,
        mem_per_cpu_mb=B4_MEM_PER_CPU_MB,
        max_concurrent_trials=max(1, cores // B4_CPUS_PER_TRIAL),
        slurm_time=slurm_time(budget_s + RITME_WALLTIME_BUFFER_S),
        slurm_account=cluster_config.slurm_account(),
        sbatch_extra=constraint_flags(),
        config_overrides={
            "experiment_tag": tag,
            "seed_data": seed,
            "seed_model": seed,
            "time_budget_s": budget_s,
            "model_hyperparameters": _ritme_hyperparameters(usecase),
        },
    )
    return parse_job_id(proc.stdout)


def submit_automl(
    usecase: str, cores: int, seed: int, budget_s: int, run_dir: Path
) -> str:
    inner = [
        "env",
        *(f"{var}={THREADS_PER_WORKER}" for var in THREAD_ENV_VARS),
        autosklearn_python(),
        "-u",
        "-m",
        "benchmarking.automl_sweep",
        *_data_args(usecase),
        "--budget-s",
        str(budget_s),
        "--cores",
        str(cores),
        "--n-jobs",
        str(cores),
        "--threads-per-job",
        str(THREADS_PER_WORKER),
        "--seed",
        str(seed),
        "--out-dir",
        str(run_dir),
        "--memory-limit-mb",
        str(B4_MEM_PER_CPU_MB),
        "--restricted-model",
        B4_AUTOML_MODEL[usecase],
    ]
    spec = USECASES[usecase]
    if spec.get("group_by_column"):
        inner += ["--group-by-column", spec["group_by_column"]]
    tag = b4_tag(usecase, "automl", cores, seed)
    return _sbatch(
        tag,
        cores,
        budget_s + AUTOML_WALLTIME_BUFFER_S,
        run_dir.parent / "logs" / f"{tag}_out.txt",
        inner,
    )


def submit_tpot(
    usecase: str, cores: int, seed: int, budget_s: int, run_dir: Path
) -> str:
    inner = [
        "env",
        *(f"{var}={THREADS_PER_WORKER}" for var in THREAD_ENV_VARS),
        env_python(TPOT_ENV_NAME, "tpot_python"),
        "-u",
        "-m",
        "src.comparator_tpot",
        *_data_args(usecase),
        "--seed",
        str(seed),
        "--n-jobs",
        str(cores),
        "--out-dir",
        str(run_dir),
        "--total-time-s",
        str(budget_s),
        "--max-eval-time-mins",
        str(b4_tpot_max_eval_time_mins(budget_s)),
        "--restricted-model",
        B4_TPOT_MODEL[usecase],
        # Pareto-front pipelines are written once per generation, so a run that
        # dies still leaves its best score recoverable (collect_comparators.py).
        "--checkpoint-dir",
        str(run_dir / "checkpoints"),
    ]
    tag = b4_tag(usecase, "tpot", cores, seed)
    return _sbatch(
        tag,
        cores,
        budget_s + b4_tpot_max_eval_time_mins(budget_s) * 60 + TPOT_WALLTIME_MARGIN_S,
        run_dir.parent / "logs" / f"{tag}_out.txt",
        inner,
        raise_ulimits=True,
    )


def submit_maml(
    usecase: str, cores: int, seed: int, budget_s: int, run_dir: Path
) -> str:
    inner = [
        "env",
        *(f"{var}={THREADS_PER_WORKER}" for var in THREAD_ENV_VARS),
        env_python(B4_MAML_ENV_NAME, "maml_python"),
        "-u",
        "-m",
        "benchmarking.maml_sweep",
        *_data_args(usecase),
        "--budget-s",
        str(budget_s),
        "--cores",
        str(cores),
        "--n-jobs",
        str(cores),
        "--seed",
        str(seed),
        "--out-dir",
        str(run_dir),
    ]
    tag = b4_tag(usecase, "maml", cores, seed)
    # The worker stops *starting* grids at the budget; the one in flight may
    # run on, and the collector excludes it if it finishes late, so being
    # killed at walltime loses nothing that would have counted.
    return _sbatch(
        tag,
        cores,
        budget_s + MAML_WALLTIME_MARGIN_S,
        run_dir.parent / "logs" / f"{tag}_out.txt",
        inner,
    )


def _status(method: str, run_dir: Path, usecase: str) -> str:
    """'done' | 'incomplete' | 'missing' for one run's outputs."""
    if method == "ritme":
        done = (run_dir / "mlflow_logs.csv").exists()
    elif method == "automl":
        done = any(run_dir.glob("automl_*_metrics.json"))
    elif method == "maml":
        done = any(run_dir.glob("maml_*_metrics.json"))
    else:
        done = (run_dir / f"{usecase}_tpot_metrics.csv").exists()
    if done:
        return "done"
    return "incomplete" if run_dir.exists() else "missing"


def main() -> None:
    args = parse_args()
    if "maml" in args.methods:
        raise SystemExit(
            "The mAML arm is archived (archive/code/maml_sweep.py) and is "
            "incompatible with the current src/comparator_maml.py; restore "
            "and fix it before launching maml."
        )
    ensure_launcher_env()
    benchmark = "b4_smoke" if args.smoke else "b4"
    budget_s = SMOKE_BUDGET_S if args.smoke else B4_TIME_BUDGET_S
    cores_list = args.cores or (SMOKE_CORES if args.smoke else B4_CORES)
    seeds = args.seeds or (SMOKE_SEEDS if args.smoke else SEEDS)
    runs_root = RUNS_DIR / benchmark

    jobs = []
    for usecase in args.usecases:
        for method in args.methods:
            if usecase not in B4_METHOD_USECASES.get(method, B4_USECASES):
                continue
            for cores in cores_list:
                for seed in seeds:
                    reused = (usecase, method) in B4_REUSE_FROM_B2 and not args.smoke
                    run_dir = (
                        b4_run_dir(usecase, method, cores, seed)
                        if reused
                        else runs_root / method / b4_tag(usecase, method, cores, seed)
                    )
                    status = _status(method, run_dir, usecase)
                    if reused:
                        if status != "done":
                            print(f"[warn] B2 cell to reuse is {status}: {run_dir}")
                        else:
                            print(f"[reuse] B2 outputs: {run_dir}")
                        continue
                    if status == "done":
                        print(f"[skip] outputs exist: {run_dir}")
                        continue
                    if status == "incomplete":
                        print(f"[skip] incomplete run; remove to relaunch: {run_dir}")
                        continue
                    if args.dry_run:
                        print(
                            f"[dry-run] would submit {method} {usecase} c{cores} s{seed}"
                        )
                        continue
                    if method == "ritme":
                        job_id = submit_ritme(usecase, cores, seed, budget_s, runs_root)
                    elif method == "automl":
                        job_id = submit_automl(usecase, cores, seed, budget_s, run_dir)
                    elif method == "maml":
                        job_id = submit_maml(usecase, cores, seed, budget_s, run_dir)
                    else:
                        job_id = submit_tpot(usecase, cores, seed, budget_s, run_dir)
                    jobs.append(
                        {
                            "job_id": job_id,
                            "method": method,
                            "usecase": usecase,
                            "cores": cores,
                            "seed": seed,
                            "experiment_tag": b4_tag(usecase, method, cores, seed),
                        }
                    )

    if jobs:
        source_commit_file = BENCH_DIR / ".comparator_source_commit"
        manifest = write_manifest(
            benchmark,
            params={
                "time_budget_s": budget_s,
                "cores": cores_list,
                "seeds": seeds,
                "mem_per_cpu_mb": B4_MEM_PER_CPU_MB,
                "cpus_per_trial": B4_CPUS_PER_TRIAL,
                "ritme_model": B4_RITME_MODEL,
                "automl_model": B4_AUTOML_MODEL,
                "tpot_model": B4_TPOT_MODEL,
                "tpot_max_eval_time_mins": b4_tpot_max_eval_time_mins(budget_s),
                "method_usecases": B4_METHOD_USECASES,
                "threads_per_worker": THREADS_PER_WORKER,
                "enrich_with": {u: _read_enrich_with(u) for u in args.usecases},
                "comparator_source_commit": (
                    source_commit_file.read_text().strip()
                    if source_commit_file.exists()
                    else None
                ),
                "runs_root": repo_relative(runs_root),
            },
            jobs=jobs,
        )
        print(f"Submitted {len(jobs)} jobs; manifest: {manifest}")
    else:
        print("Nothing submitted.")


if __name__ == "__main__":
    main()
