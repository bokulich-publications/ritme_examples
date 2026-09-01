"""Launch ritme experiments for one or more (usecase, model_type) combos."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

from src import cluster_config

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "src/run_ritme_model.sh"
STABILITY_TEMPLATE = REPO_ROOT / "src/run_ritme_stability.sh"

USECASES: dict[str, dict] = {
    "u1": {
        "config_prefix": "u1",
        "use_case_dir": "use_cases/u1_amplicon_age_prediction",
        "data_splits": "use_cases/u1_amplicon_age_prediction/data_splits_u1",
        "path_md": "data/u1_subramanian14/md_subr14.tsv",
        "path_ft": "data/u1_subramanian14/otu_table_subr14_wq.tsv",
        "path_tax": "data/u1_subramanian14/taxonomy_subr14.tsv",
        "path_phylo": "data/u1_subramanian14/fasttree_tree_rooted_subr14.nwk",
        "group_by_column": "host_id",
        "stratify_by": None,
        "task": "regression",
        "time_col": None,
        "host_col": None,
        "n_prev": None,
        "qza_inputs": [
            (
                "feature-table",
                "data/u1_subramanian14/otu_table_subr14_wq.qza",
                "data/u1_subramanian14/otu_table_subr14_wq.tsv",
            ),
            (
                "taxonomy",
                "data/u1_subramanian14/taxonomy_subr14.qza",
                "data/u1_subramanian14/taxonomy_subr14.tsv",
            ),
            (
                "tree",
                "data/u1_subramanian14/fasttree_tree_rooted_subr14.qza",
                "data/u1_subramanian14/fasttree_tree_rooted_subr14.nwk",
            ),
        ],
    },
    "u1_dynamic": {
        "config_prefix": "u1_dynamic",
        "base_config_prefix": "u1",
        "use_case_dir": "use_cases/u1_amplicon_age_prediction",
        "data_splits": "use_cases/u1_amplicon_age_prediction/data_splits_u1_dynamic",
        "path_md": "data/u1_subramanian14/md_subr14.tsv",
        "path_ft": "data/u1_subramanian14/otu_table_subr14_wq.tsv",
        "path_tax": "data/u1_subramanian14/taxonomy_subr14.tsv",
        "path_phylo": "data/u1_subramanian14/fasttree_tree_rooted_subr14.nwk",
        "group_by_column": "host_id",
        "stratify_by": None,
        "task": "regression",
        "time_col": "age_months_rounded1",
        "host_col": "host_id",
        "n_prev": 1,
        "qza_inputs": [
            (
                "feature-table",
                "data/u1_subramanian14/otu_table_subr14_wq.qza",
                "data/u1_subramanian14/otu_table_subr14_wq.tsv",
            ),
            (
                "taxonomy",
                "data/u1_subramanian14/taxonomy_subr14.qza",
                "data/u1_subramanian14/taxonomy_subr14.tsv",
            ),
            (
                "tree",
                "data/u1_subramanian14/fasttree_tree_rooted_subr14.qza",
                "data/u1_subramanian14/fasttree_tree_rooted_subr14.nwk",
            ),
        ],
    },
    "u2": {
        "config_prefix": "u2",
        "use_case_dir": "use_cases/u2_metagenome_ocean",
        "data_splits": "use_cases/u2_metagenome_ocean/data_splits_u2",
        "path_md": "data/u2_tara_ocean/md_tara_ocean.tsv",
        "path_ft": "data/u2_tara_ocean/otu_table_tara_ocean.tsv",
        "path_tax": "data/u2_tara_ocean/taxonomy_tara_ocean.tsv",
        "path_phylo": "data/u2_tara_ocean/fasttree_tree_rooted_proc_suna15.nwk",
        "group_by_column": None,
        "stratify_by": None,
        "task": "regression",
        "time_col": None,
        "host_col": None,
        "n_prev": None,
        "qza_inputs": [
            (
                "taxonomy",
                "data/u2_tara_ocean/taxonomy_tara_ocean.qza",
                "data/u2_tara_ocean/taxonomy_tara_ocean.tsv",
            ),
            (
                "tree",
                "data/u2_tara_ocean/fasttree_tree_rooted_proc_suna15.qza",
                "data/u2_tara_ocean/fasttree_tree_rooted_proc_suna15.nwk",
            ),
        ],
    },
    "u3": {
        "config_prefix": "u3",
        "use_case_dir": "use_cases/u3_amplicon_crc_classification",
        "data_splits": "use_cases/u3_amplicon_crc_classification/data_splits_u3",
        "path_md": "data/u3_topcuoglu20_baxter/md_baxter.tsv",
        "path_ft": "data/u3_topcuoglu20_baxter/otu_table_baxter_unrarefied.tsv",
        "path_tax": "data/u3_topcuoglu20_baxter/taxonomy_baxter.tsv",
        "path_phylo": None,
        "group_by_column": None,
        "stratify_by": "srn",
        "task": "classification",
        "time_col": None,
        "host_col": None,
        "n_prev": None,
        "qza_inputs": [],
    },
    "u4": {
        "config_prefix": "u4",
        "use_case_dir": "use_cases/u4_amplicon_emp_classification",
        "data_splits": "use_cases/u4_amplicon_emp_classification/data_splits_u4",
        "path_md": "data/u4_emp/md_emp.tsv",
        # not read: the splits are pre-staged, so the split step is skipped
        "path_ft": "data/u4_emp/emp_deblur_90bp.qc_filtered.biom",
        "path_tax": "data/u4_emp/taxonomy_emp.tsv",
        "path_phylo": None,
        "group_by_column": None,
        # CV stratification is set in the config JSON (`stratify_by`), which is
        # what `find_best_model_config` reads; this entry only feeds the split
        # step, which u4 skips.
        "stratify_by": None,
        "task": "classification",
        "time_col": None,
        "host_col": None,
        "n_prev": None,
        "qza_inputs": [],
    },
}

SLURM_RESOURCES: dict[tuple[str, str], dict] = {
    ("u1", "linreg"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1", "rf"): {
        "cpus": 50,
        "mem_per_cpu_mb": 5120,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1", "trac"): {
        "cpus": 60,
        "mem_per_cpu_mb": 6144,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1", "xgb"): {
        "cpus": 50,
        "mem_per_cpu_mb": 4096,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1", "nn_reg"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1", "nn_class"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1", "nn_corn"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1_dynamic", "linreg"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1_dynamic", "rf"): {
        "cpus": 50,
        "mem_per_cpu_mb": 5120,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1_dynamic", "trac"): {
        "cpus": 60,
        "mem_per_cpu_mb": 6144,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1_dynamic", "xgb"): {
        "cpus": 50,
        "mem_per_cpu_mb": 4096,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1_dynamic", "nn_reg"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1_dynamic", "nn_class"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1_dynamic", "nn_corn"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u2", "linreg"): {
        "cpus": 50,
        "mem_per_cpu_mb": 4096,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u2", "rf"): {
        "cpus": 60,
        "mem_per_cpu_mb": 6144,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u2", "trac"): {
        "cpus": 80,
        "mem_per_cpu_mb": 8192,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u2", "xgb"): {
        "cpus": 60,
        "mem_per_cpu_mb": 6144,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u2", "nn_reg"): {
        "cpus": 50,
        "mem_per_cpu_mb": 4096,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u2", "nn_class"): {
        "cpus": 50,
        "mem_per_cpu_mb": 4096,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u2", "nn_corn"): {
        "cpus": 50,
        "mem_per_cpu_mb": 4096,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u3", "logreg"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u3", "rf_class"): {
        "cpus": 50,
        "mem_per_cpu_mb": 5120,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u3", "xgb_class"): {
        "cpus": 50,
        "mem_per_cpu_mb": 4096,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u3", "nn_class"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u4", "logreg"): {
        "cpus": 50,
        "mem_per_cpu_mb": 10240,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u4", "rf_class"): {
        "cpus": 50,
        "mem_per_cpu_mb": 10240,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u4", "xgb_class"): {
        # 8 concurrent trials at full width hit 524 GB and had Ray actors
        # OOM-killed; XGBoost's quantised DMatrix copies dominate.
        "cpus": 50,
        "mem_per_cpu_mb": 20480,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u4", "nn_class"): {
        "cpus": 50,
        "mem_per_cpu_mb": 10240,
        "gpus": 0,
        "slurm_account": None,
    },
}

MAX_CONCURRENT_TRIALS: dict[str, int] = {
    "linreg": 10,
    "rf": 10,
    "trac": 10,
    "xgb": 10,
    "nn_reg": 5,
    "nn_class": 5,
    "nn_corn": 5,
    "logreg": 10,
    "rf_class": 10,
    "xgb_class": 10,
}


def _experiment_tag(
    usecase: str, model_type: str, sampler: str, variant: Optional[str]
) -> str:
    """Experiment tag of the run ``submit_model`` launches for these arguments."""
    spec = USECASES[usecase]
    prefix = spec["config_prefix"]
    if variant and variant != "no_enrich":
        config_dir = REPO_ROOT / spec["use_case_dir"] / "config"
        path = config_dir / f"{prefix}_{model_type}_{sampler}_{variant}.json"
        return json.loads(path.read_text())["experiment_tag"]
    tag_suffix = f"_{variant}" if variant else ""
    return f"{prefix}_{model_type}_{sampler}{tag_suffix}"


def _resolve_config_for_run(
    usecase: str,
    model_type: str,
    sampler: str,
    variant: Optional[str],
    logs_dir: Path,
    config_overrides: Optional[dict] = None,
    max_concurrent_trials: Optional[int] = None,
) -> Path:
    """Return the JSON config to hand to ritme for this run."""
    spec = USECASES[usecase]
    config_dir = REPO_ROOT / spec["use_case_dir"] / "config"
    prefix = spec["config_prefix"]

    if variant and variant != "no_enrich":
        path = config_dir / f"{prefix}_{model_type}_{sampler}_{variant}.json"
        if not path.exists():
            raise FileNotFoundError(f"Variant config not found: {path}")
        if not config_overrides and max_concurrent_trials is None:
            return path
        cfg = json.loads(path.read_text())
    else:
        base_prefix = spec.get("base_config_prefix", prefix)
        base_path = config_dir / f"{base_prefix}_base_{sampler}.json"
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")
        cfg = json.loads(base_path.read_text())
        cfg["experiment_tag"] = _experiment_tag(usecase, model_type, sampler, variant)
        cfg["ls_model_types"] = [model_type]
        if variant == "no_enrich":
            mh = cfg.get("model_hyperparameters")
            if isinstance(mh, dict):
                mh.pop("data_enrich_with", None)
                if not mh:
                    cfg.pop("model_hyperparameters")

    if max_concurrent_trials is not None:
        cfg["max_cuncurrent_trials"] = max_concurrent_trials
    else:
        try:
            cfg["max_cuncurrent_trials"] = MAX_CONCURRENT_TRIALS[model_type]
        except KeyError as e:
            raise KeyError(
                f"No MAX_CONCURRENT_TRIALS entry for model_type={model_type!r}. "
                f"Add an entry to src/launch_models.py:MAX_CONCURRENT_TRIALS or "
                f"pass max_concurrent_trials= explicitly on submit_model."
            ) from e

    if config_overrides:
        cfg.update(config_overrides)

    resolved_dir = logs_dir / "_resolved_configs"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolved_dir / f"{cfg['experiment_tag']}.json"
    resolved.write_text(json.dumps(cfg, indent=2) + "\n")
    return resolved


def _pin_launcher_env(env: dict) -> None:
    """Pin the launching interpreter's environment for the submitted job.

    ``run_ritme_model.sh`` activates no environment, and ``sbatch
    --export=ALL`` forwards the caller's ``PATH``, so a job runs whatever
    ``ritme`` the *launching shell* resolves. Submitted from a shell without
    the ritme env active that is silently a different interpreter -- the job
    dies seconds in, or worse, runs against the wrong dependencies.

    Putting the launching interpreter's ``bin`` first makes the job use the
    environment the submission was made from, and a missing ``ritme`` beside
    it becomes a hard error rather than a late, confusing one.
    """
    bin_dir = Path(sys.executable).resolve().parent
    if not (bin_dir / "ritme").exists():
        raise RuntimeError(
            f"No 'ritme' executable beside the launching interpreter "
            f"({bin_dir}). Submit from the ritme environment, e.g. "
            f"`mamba run -n ritme_usecases python -m ...`; otherwise the job "
            f"inherits PATH and would silently run a different ritme."
        )
    # Presence is not health: a stale environment can carry a complete ritme
    # stack whose scipy is broken, which passes the check above and then fails
    # seconds into the job. Import the dependencies that fail cheaply in that
    # state, in the very interpreter the job will inherit (~3 s).
    try:
        importlib.import_module("ray")
        importlib.import_module("scipy.stats")
    except Exception as exc:
        raise RuntimeError(
            f"{sys.executable} has a 'ritme' executable but cannot import "
            f"ritme's dependencies ({type(exc).__name__}: {exc}). The "
            f"environment is broken or stale; submit from a working ritme "
            f"environment instead."
        ) from exc
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"


def _build_env(usecase: str, config_path: Path, logs_dir: Path) -> dict:
    spec = USECASES[usecase]
    env = os.environ.copy()
    _pin_launcher_env(env)
    env["CONFIG"] = str(config_path)
    env["PATH_MD"] = str(REPO_ROOT / spec["path_md"])
    env["PATH_FT"] = str(REPO_ROOT / spec["path_ft"])
    env["PATH_DATA_SPLITS"] = str(REPO_ROOT / spec["data_splits"])
    env["LOGS_DIR"] = str(logs_dir)
    if spec["path_tax"]:
        env["PATH_TAX"] = str(REPO_ROOT / spec["path_tax"])
    if spec["path_phylo"]:
        env["PATH_PHYLO"] = str(REPO_ROOT / spec["path_phylo"])
    if spec["group_by_column"]:
        env["GROUP_BY_COLUMN"] = spec["group_by_column"]
    if spec["stratify_by"]:
        env["STRATIFY_BY_COLUMN"] = spec["stratify_by"]
    snapshot_fields = ("time_col", "host_col", "n_prev")
    set_fields = [f for f in snapshot_fields if spec[f] is not None]
    if 0 < len(set_fields) < 3:
        raise ValueError(
            f"USECASES[{usecase!r}] has a partial snapshot config "
            f"(set: {set_fields}). Set all of {list(snapshot_fields)} "
            f"together, or leave all three as None."
        )
    if len(set_fields) == 3:
        if not isinstance(spec["n_prev"], int) or spec["n_prev"] < 0:
            raise ValueError(
                f"USECASES[{usecase!r}]['n_prev'] must be a non-negative "
                f"int; got {spec['n_prev']!r}."
            )
        env["TIME_COL"] = spec["time_col"]
        env["HOST_COL"] = spec["host_col"]
        env["N_PREV"] = str(spec["n_prev"])
    if spec["qza_inputs"]:
        env["QZA_INPUTS"] = " ".join(
            f"{kind}:{REPO_ROOT / src}:{REPO_ROOT / dst}"
            for kind, src, dst in spec["qza_inputs"]
        )
    return env


SLURM_WALLTIME_TIERS_S: list[int] = [
    4 * 3600,
    24 * 3600,
    120 * 3600,
]
_SHAP_BOOTSTRAP_BUFFER_S: int = 3600


def _seconds_to_slurm_time(seconds: int) -> str:
    """Format ``seconds`` as the SLURM ``HH:MM:SS`` walltime string."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _default_slurm_time(time_budget_s: int) -> str:
    """Snap a ritme ``time_budget_s`` (+ SHAP/bootstrap buffer) to the
    smallest entry of :data:`SLURM_WALLTIME_TIERS_S` that fits.
    """
    needed_s = time_budget_s + _SHAP_BOOTSTRAP_BUFFER_S
    for tier_s in SLURM_WALLTIME_TIERS_S:
        if needed_s <= tier_s:
            return _seconds_to_slurm_time(tier_s)
    raise ValueError(
        f"time_budget_s={time_budget_s}s (+ {_SHAP_BOOTSTRAP_BUFFER_S}s "
        f"SHAP/bootstrap buffer) = {needed_s}s exceeds the largest SLURM "
        f"walltime tier ({SLURM_WALLTIME_TIERS_S[-1]}s = "
        f"{SLURM_WALLTIME_TIERS_S[-1] // 3600}h). Reduce time_budget_s or "
        f"pass slurm_time= directly on submit_model."
    )


# Env vars the run templates read; only these are forwarded into sbatch jobs.
_FORWARDED_ENV_VARS: frozenset[str] = frozenset(
    {
        # src/run_ritme_model.sh
        "CONFIG",
        "PATH_MD",
        "PATH_FT",
        "PATH_DATA_SPLITS",
        "LOGS_DIR",
        "PATH_TAX",
        "PATH_PHYLO",
        "GROUP_BY_COLUMN",
        "STRATIFY_BY_COLUMN",
        "TIME_COL",
        "HOST_COL",
        "N_PREV",
        "QZA_INPUTS",
        "SHAP_MAX_BACKGROUND_SAMPLES",
        # src/run_ritme_stability.sh
        "PATH_EXP",
        "MODEL_TYPE",
        "STABILITY_TOP_N",
        "STABILITY_MAX_TRIALS",
        "STABILITY_BAND_SE_FACTOR",
        "STABILITY_MAX_CONCURRENT_TRIALS",
    }
)


def _resolve_logs_path(logs_dir: str | os.PathLike) -> Path:
    logs_path = Path(logs_dir)
    if not logs_path.is_absolute():
        logs_path = REPO_ROOT / logs_path
    logs_path.mkdir(parents=True, exist_ok=True)
    return logs_path


def _launch(
    template: Path,
    env: dict,
    *,
    mode: str,
    usecase: str,
    model_type: str,
    job_name: str,
    logs_path: Path,
    slurm_time: Optional[str],
    slurm_account: Optional[str],
    cpus: Optional[int],
    mem_per_cpu_mb: Optional[int],
    sbatch_extra: Optional[Iterable[str]],
) -> subprocess.CompletedProcess:
    """Run ``template`` inline (``mode="local"``) or submit it via sbatch with
    the ``SLURM_RESOURCES`` allocation of ``(usecase, model_type)``.
    """
    if mode == "local":
        return subprocess.run(["bash", str(template)], env=env, check=True)

    if mode != "slurm":
        raise ValueError(f"Unknown mode: {mode!r}")

    res = SLURM_RESOURCES.get((usecase, model_type))
    if res is None:
        raise KeyError(
            f"No SLURM allocation registered for {(usecase, model_type)!r}. "
            f"Add an entry to src/launch_models.py:SLURM_RESOURCES or pass "
            f"cpus= and mem_per_cpu_mb= explicitly."
        )
    cpus = res["cpus"] if cpus is None else cpus
    mem_per_cpu_mb = res["mem_per_cpu_mb"] if mem_per_cpu_mb is None else mem_per_cpu_mb
    gpus = res.get("gpus", 0)
    if cpus <= 0 or mem_per_cpu_mb <= 0:
        raise ValueError(
            f"cpus and mem_per_cpu_mb must be positive; got "
            f"cpus={cpus}, mem_per_cpu_mb={mem_per_cpu_mb}."
        )
    if gpus < 0:
        raise ValueError(f"gpus must be non-negative; got gpus={gpus}.")
    if slurm_account is None:
        slurm_account = res.get("slurm_account")
    out_log = logs_path / "logs" / f"{job_name}_out.txt"
    out_log.parent.mkdir(parents=True, exist_ok=True)

    forwarded = ",".join(sorted(k for k in env if k in _FORWARDED_ENV_VARS))
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        "--ntasks=1",
        f"--cpus-per-task={cpus}",
        f"--mem-per-cpu={mem_per_cpu_mb}",
        f"--time={slurm_time}",
        f"--output={out_log}",
        "--open-mode=append",
        f"--export=ALL,{forwarded}",
    ]
    if gpus > 0:
        cmd.append(f"--gpus-per-node={gpus}")
    if slurm_account:
        cmd.insert(1, f"--account={slurm_account}")
    if sbatch_extra:
        cmd[1:1] = list(sbatch_extra)
    cmd.append(str(template))

    print("submitting:", cluster_config.redact(cmd))
    # Captured and echoed so the job id shows up in notebook cell output.
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    for stream in (proc.stdout, proc.stderr):
        if stream:
            print(stream.strip())
    proc.check_returncode()
    return proc


def submit_model(
    usecase: str,
    model_type: str,
    *,
    sampler: str = "tpe",
    variant: Optional[str] = None,
    logs_dir: str | os.PathLike = "use_cases/ritme_runs/local",
    mode: str = "slurm",
    sbatch_extra: Optional[Iterable[str]] = None,
    slurm_time: Optional[str] = None,
    slurm_account: Optional[str] = None,
    cpus: Optional[int] = None,
    mem_per_cpu_mb: Optional[int] = None,
    max_concurrent_trials: Optional[int] = None,
    config_overrides: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    """Submit (or run locally) a single ritme experiment."""
    logs_path = _resolve_logs_path(logs_dir)
    config_path = _resolve_config_for_run(
        usecase,
        model_type,
        sampler,
        variant,
        logs_path,
        config_overrides=config_overrides,
        max_concurrent_trials=max_concurrent_trials,
    )
    env = _build_env(usecase, config_path, logs_path)
    if mode == "slurm" and slurm_time is None:
        resolved_cfg = json.loads(config_path.read_text())
        slurm_time = _default_slurm_time(resolved_cfg["time_budget_s"])

    return _launch(
        TEMPLATE,
        env,
        mode=mode,
        usecase=usecase,
        model_type=model_type,
        job_name=config_path.stem,
        logs_path=logs_path,
        slurm_time=slurm_time,
        slurm_account=slurm_account,
        cpus=cpus,
        mem_per_cpu_mb=mem_per_cpu_mb,
        sbatch_extra=sbatch_extra,
    )


def submit_stability(
    usecase: str,
    model_type: str,
    *,
    sampler: str = "tpe",
    variant: Optional[str] = None,
    experiment_tag: Optional[str] = None,
    logs_dir: str | os.PathLike = "use_cases/ritme_runs/local",
    mode: str = "slurm",
    top_n: int = 15,
    max_trials: int = 15,
    band_se_factor: float = 1.0,
    max_concurrent_trials: Optional[int] = None,
    max_background_samples: Optional[int] = None,
    sbatch_extra: Optional[Iterable[str]] = None,
    slurm_time: str = "04:00:00",
    slurm_account: Optional[str] = None,
    cpus: Optional[int] = None,
    mem_per_cpu_mb: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """Run ``ritme explain-stability`` on the completed experiment that
    ``submit_model(usecase, model_type, sampler=..., variant=...)`` produced;
    pass ``experiment_tag`` for a run whose tag was set via ``config_overrides``.

    Retrains the ``model_type`` trials whose cross-validation performance is
    indistinguishable from the deployed model's and reports where its
    ``top_n`` features rank in each of them. Resources default to the
    experiment's ``SLURM_RESOURCES`` entry and ``max_concurrent_trials`` to
    ``MAX_CONCURRENT_TRIALS[model_type]``, so every retrained trial gets the
    same per-trial allocation as during the search. Outputs are written to
    ``<experiment dir>/stability_<model_type>/``.
    """
    spec = USECASES[usecase]
    logs_path = _resolve_logs_path(logs_dir)
    exp_tag = experiment_tag or _experiment_tag(usecase, model_type, sampler, variant)
    exp_dir = logs_path / exp_tag
    if not (exp_dir / "mlflow_logs.csv").exists():
        raise FileNotFoundError(
            f"No completed experiment at {exp_dir} (missing mlflow_logs.csv)."
        )
    if max_concurrent_trials is None:
        max_concurrent_trials = MAX_CONCURRENT_TRIALS[model_type]

    env = os.environ.copy()
    env["PATH_EXP"] = str(exp_dir)
    env["MODEL_TYPE"] = model_type
    env["PATH_DATA_SPLITS"] = str(REPO_ROOT / spec["data_splits"])
    env["STABILITY_TOP_N"] = str(top_n)
    env["STABILITY_MAX_TRIALS"] = str(max_trials)
    env["STABILITY_BAND_SE_FACTOR"] = str(band_se_factor)
    env["STABILITY_MAX_CONCURRENT_TRIALS"] = str(max_concurrent_trials)
    if spec["path_tax"]:
        env["PATH_TAX"] = str(REPO_ROOT / spec["path_tax"])
    if spec["path_phylo"]:
        env["PATH_PHYLO"] = str(REPO_ROOT / spec["path_phylo"])
    if max_background_samples is not None:
        env["SHAP_MAX_BACKGROUND_SAMPLES"] = str(max_background_samples)

    return _launch(
        STABILITY_TEMPLATE,
        env,
        mode=mode,
        usecase=usecase,
        model_type=model_type,
        job_name=f"stability_{exp_tag}",
        logs_path=logs_path,
        slurm_time=slurm_time,
        slurm_account=slurm_account,
        cpus=cpus,
        mem_per_cpu_mb=mem_per_cpu_mb,
        sbatch_extra=sbatch_extra,
    )
