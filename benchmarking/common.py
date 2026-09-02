"""Shared constants and helpers for the ritme benchmarks.

See benchmarking/README.md for how to launch, collect and plot the three
final figures (original design plan: archive/docs/plan_benchmark_design.md).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from src import cluster_config

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "benchmarking"
RUNS_DIR = BENCH_DIR / "runs"
MANIFESTS_DIR = BENCH_DIR / "manifests"
RESULTS_DIR = BENCH_DIR / "results"
DATA_DIR = RESULTS_DIR / "data"  # collector outputs (gitignored working data)
FINAL_DIR = RESULTS_DIR / "final"  # the tracked final figures
SEARCH_ONLY_TEMPLATE = BENCH_DIR / "run_ritme_search_only.sh"

# All benchmark jobs are pinned to one node type so timings, trial counts and
# memory stay comparable across arms and core counts; pick one with >=128
# cores that fits 128 x B2_MEM_PER_CPU_MB. The node type and the SLURM account
# identify a specific site, so they are never hardcoded here -- see
# src/cluster_config.py and .cluster.example.json. When either is unset the
# corresponding sbatch flag is omitted and SLURM applies its own default.


def account_flags() -> list[str]:
    """``--account`` for sbatch, or nothing when no account is configured."""
    account = cluster_config.slurm_account()
    return [f"--account={account}"] if account else []


def constraint_flags() -> list[str]:
    """``--constraint`` for sbatch, or nothing when no node type is set."""
    constraint = cluster_config.node_constraint()
    return [f"--constraint={constraint}"] if constraint else []


SEEDS = [0, 1, 2]

# B1: TPE vs random search on U1/xgb, identical allocation per arm (the
# registered u1/xgb allocation from src/launch_models.py).
B1_SAMPLERS = ["tpe", "random"]
B1_TIME_BUDGET_S = 43200
B1_CPUS = 50
B1_MEM_PER_CPU_MB = 4096
B1_MAX_CONCURRENT_TRIALS = 10

# B2: resource sweep on U1. Both methods spend the allocation as
# cores / CPUS_PER_TRIAL concurrent units of CPUS_PER_TRIAL cores each; 3584
# MB/cpu is the largest per-core request for which 128 cores still fit on an
# node of the pinned type. See "Design notes" in benchmarking/README.md.
B2_TIME_BUDGET_S = 7200
B2_CORES = [4, 8, 16, 32, 64, 128]
B2_MEM_PER_CPU_MB = 3584
B2_CPUS_PER_TRIAL = 4
B2_AUTOML_MODEL = "gradient_boosting"
# auto-sklearn parallelises across configurations rather than within a fit, so
# it fills the allocation with one single-threaded worker per core; measured
# against the alternatives in benchmarking/README.md. Each worker is then
# capped at one core's memory, summing to the allocation.
B2_AUTOML_N_JOBS_PER_CORE = 1
B2_AUTOML_THREADS_PER_JOB = 1
B2_AUTOML_MEMORY_LIMIT_MB = B2_MEM_PER_CPU_MB


# B4: the B2 design (budget, cores, memory, seeds) on all three use cases, with
# TPOT as a third arm and mAML as a reference. ritme's model per use case is the
# one its own validation objective selects -- `rmse_val` (min) / `roc_auc_val`
# (max), never the test score -- and each comparator is restricted to the
# closest family in its own catalogue (TPOT's pairing follows
# src/comparator_tpot.py:ESTIMATOR_FOR_USECASE; auto-sklearn has no ElasticNet,
# so `sgd` stands in for linreg as in the manuscript's own comparison).
B4_TIME_BUDGET_S = B2_TIME_BUDGET_S
B4_CORES = B2_CORES
B4_MEM_PER_CPU_MB = B2_MEM_PER_CPU_MB
B4_CPUS_PER_TRIAL = B2_CPUS_PER_TRIAL
B4_USECASES = ["u1", "u2", "u3"]
B4_METHODS = ["ritme", "automl", "tpot", "maml"]
# Arms that only apply to some use cases (mAML is classification-only).
B4_METHOD_USECASES = {"maml": ["u3"]}
B4_RITME_MODEL = {"u1": "xgb", "u2": "linreg", "u3": "xgb_class"}
B4_AUTOML_MODEL = {"u1": "gradient_boosting", "u2": "sgd", "u3": "gradient_boosting"}
B4_MAML_ENV_NAME = "maml_bench"
B4_TPOT_MODEL = {
    "u1": "xgboost.XGBRegressor",
    "u2": "sklearn.linear_model.ElasticNetCV",
    "u3": "xgboost.XGBClassifier",
}
# TPOT's per-evaluation cap is a `stopit` timeout that, when it fires inside
# XGBoost's C code, corrupts the heap and kills the run (3 of 3 manuscript u1
# jobs). It cannot be disabled, but set above the search budget it can never
# fire before the search ends. One pathological pipeline is then unbounded,
# which the walltime buffer has to absorb.
B4_TPOT_CAP_MARGIN_MINS = 10


def b4_tpot_max_eval_time_mins(budget_s: int) -> int:
    """TPOT's per-evaluation cap for a run of ``budget_s``: just above the
    search budget, so it can never fire before the search ends."""
    return budget_s // 60 + B4_TPOT_CAP_MARGIN_MINS


# The (usecase, method) cells B4 can take over from B2 unchanged: same config,
# same enrichment, same allocation. B2's auto-sklearn arm is *not* reused --
# it predates the metadata enrichment the comparator arms now receive.
B4_REUSE_FROM_B2 = {("u1", "ritme"): "b2_u1_xgb_c{cores}_s{seed}"}
# mAML is an exhaustive grid with no budget (781 configurations, 11 h 41 min
# at 50 cores in the manuscript run), so under the sweep's budget it evaluates
# a prefix of the grid in its published order -- maml_sweep.py (archived) stops
# starting new (scaler, classifier) grids once the budget is spent, and the
# collector counts only configurations that completed inside it.
# Panel-2 quantity per task; every arm must report its best validation score
# under this metric and the same resampling protocol as ritme.
B4_TASK_METRIC = {
    "regression": {"metric": "rmse", "mode": "min"},
    "classification": {"metric": "roc_auc", "mode": "max"},
}


def b4_tag(usecase: str, method: str, cores: int, seed: int) -> str:
    return f"b4_{usecase}_{method}_c{cores}_s{seed}"


def b4_run_dir(usecase: str, method: str, cores: int, seed: int) -> Path:
    """Where one B4 run writes; reused B2 cells resolve into runs/b2 instead."""
    pattern = B4_REUSE_FROM_B2.get((usecase, method))
    if pattern:
        return RUNS_DIR / "b2" / method / pattern.format(cores=cores, seed=seed)
    return RUNS_DIR / "b4" / method / b4_tag(usecase, method, cores, seed)


def ensure_launcher_env() -> None:
    """Put the launching interpreter's ``bin`` first on PATH before submitting.

    ritme jobs run ``ritme`` from PATH inside the run template, and
    ``submit_model`` forwards the launcher's environment with
    ``--export=ALL``. A shell where another ``python``/``ritme`` shadows the
    env's therefore hands every job the wrong interpreter -- the B4 smoke run
    died in base miniforge's scipy this way. Refuses to continue when the
    launcher itself is not running from an env that has ``ritme``.
    """
    bin_dir = Path(sys.executable).resolve().parent
    if not (bin_dir / "ritme").exists():
        raise SystemExit(
            f"Run this launcher with the ritme_usecases interpreter: "
            f"{sys.executable} has no `ritme` beside it."
        )
    # A `ritme` binary beside the interpreter is not enough: a stale install
    # in another env passes that test and then fails at import time inside the
    # job. Import the stack the job needs here, in the interpreter the job
    # will inherit, so a broken one is refused before anything is submitted.
    try:
        import ray  # noqa: F401
        import scipy.stats  # noqa: F401
    except Exception as exc:  # any import failure means the wrong interpreter
        raise SystemExit(
            f"{sys.executable} cannot import ritme's dependencies "
            f"({type(exc).__name__}: {exc}); run this launcher from the "
            "ritme_usecases interpreter."
        ) from exc
    os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
    print(f"jobs will run with {bin_dir}/ritme")


def env_python(env_name: str, config_key: str) -> str:
    """Interpreter of a sibling conda env, by *name* rather than absolute path.

    Resolved next to the env this launcher runs in (where conda/mamba put envs
    created with ``-n``), from ``sys.executable`` rather than ``CONDA_PREFIX``
    because the launcher may be invoked by an interpreter whose env was never
    activated in the calling shell. ``RITME_<CONFIG_KEY>`` or a
    ``<config_key>`` entry in .cluster.json overrides it.
    """
    configured = cluster_config.get(config_key)
    if configured:
        return configured
    envs_dir = Path(sys.executable).resolve().parent.parent.parent
    candidate = envs_dir / env_name / "bin" / "python"
    if not candidate.exists():
        raise SystemExit(
            f"Cannot locate the {env_name!r} interpreter: {candidate} does not "
            f"exist. Set RITME_{config_key.upper()} to its path, or add a "
            f"{config_key!r} key to .cluster.json."
        )
    return str(candidate)


def slurm_time(seconds: int) -> str:
    """Format seconds as a SLURM ``HH:MM:SS`` walltime string."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_job_id(sbatch_stdout: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", sbatch_stdout)
    if not match:
        raise ValueError(f"Could not parse job id from: {sbatch_stdout!r}")
    return match.group(1)


def repo_relative(path: Path) -> str:
    """Path as a repo-relative string, for values that get written to disk.

    Manifests are tracked, and an absolute path on a shared cluster embeds
    the account name of whoever ran the batch. Paths outside the repo are
    returned unchanged rather than raising, since that is still better than
    losing the record.
    """
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_manifest(benchmark: str, params: dict, jobs: list[dict]) -> Path:
    """Persist one submission batch (parameters + job ids) as JSON."""
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    path = MANIFESTS_DIR / f"{benchmark}_{stamp}.json"
    payload = {
        "benchmark": benchmark,
        "created": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "jobs": jobs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_manifests(benchmark: str) -> list[dict]:
    """Load all manifests of one benchmark, oldest first.

    The date pattern is spelled out so that ``b2`` does not also pick up
    ``b2_smoke`` manifests and merge smoke-test runs into real results.
    """
    paths = sorted(MANIFESTS_DIR.glob(f"{benchmark}_{'[0-9]' * 6}_*.json"))
    return [json.loads(p.read_text()) for p in paths]


def parse_slurm_duration(value: str) -> Optional[float]:
    """Parse a sacct duration ([DD-]HH:MM:SS[.ms] or MM:SS[.ms]) to seconds."""
    if not value or value.strip() in {"", "INVALID", "UNLIMITED"}:
        return None
    days = 0
    rest = value.strip()
    if "-" in rest:
        day_part, rest = rest.split("-", 1)
        days = int(day_part)
    parts = rest.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, (m, s) = 0, parts
    else:
        return None
    return days * 86400 + int(h) * 3600 + int(m) * 60 + float(s)


def parse_slurm_mem_mb(value: str) -> Optional[float]:
    """Parse a sacct memory value (e.g. ``79041728K``, ``200Gn``) to MB."""
    match = re.match(r"^([0-9.]+)\s*([KMGT]?)", value.strip())
    if not match:
        return None
    number = float(match.group(1))
    # SLURM reports RSS in KB when it omits the suffix.
    factor = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2, "": 1 / 1024}
    return number * factor[match.group(2)]


_SACCT_FIELDS = [
    "JobID",
    "JobName",
    "State",
    "Submit",
    "Start",
    "End",
    "Elapsed",
    "TotalCPU",
    "NCPUS",
    "ReqMem",
    "MaxRSS",
    "NodeList",
]


SACCT_COLUMNS = [
    "job_id",
    "job_name",
    "state",
    "submit",
    "start",
    "end",
    "elapsed_s",
    "total_cpu_s",
    "ncpus",
    "req_mem",
    "node_list",
    "max_rss_mb",
]


def query_sacct(job_ids: list[str], starttime: str = "2026-05-01") -> list[dict]:
    """Return one record per job with peak MaxRSS aggregated over its steps.

    ``starttime`` bounds the query because Euler's job-id counter was reset
    in May 2026, so bare ``-j`` lookups of small ids can be ambiguous.
    """
    cmd = [
        "sacct",
        "-j",
        ",".join(str(j) for j in job_ids),
        "-S",
        starttime,
        "--parsable2",
        "--noheader",
        f"--format={','.join(_SACCT_FIELDS)}",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    records: dict[str, dict] = {}
    for line in out.strip().splitlines():
        values = dict(zip(_SACCT_FIELDS, line.split("|")))
        base_id = values["JobID"].split(".")[0]
        step_rss = parse_slurm_mem_mb(values["MaxRSS"]) if values["MaxRSS"] else None
        if values["JobID"] == base_id:
            records[base_id] = {
                "job_id": base_id,
                "job_name": values["JobName"],
                "state": values["State"],
                "submit": values["Submit"],
                "start": values["Start"],
                "end": values["End"],
                "elapsed_s": parse_slurm_duration(values["Elapsed"]),
                "total_cpu_s": parse_slurm_duration(values["TotalCPU"]),
                "ncpus": int(values["NCPUS"]),
                "req_mem": values["ReqMem"],
                "node_list": values["NodeList"],
                "max_rss_mb": None,
            }
        if step_rss is not None and base_id in records:
            current = records[base_id]["max_rss_mb"]
            records[base_id]["max_rss_mb"] = (
                step_rss if current is None else max(current, step_rss)
            )
    missing = [str(j) for j in job_ids if str(j) not in records]
    if missing:
        print(
            f"[warn] {len(missing)} job(s) absent from sacct (outside the "
            f"retention window or the -S bound?): {', '.join(missing)}"
        )
    return [records[str(j)] for j in job_ids if str(j) in records]
