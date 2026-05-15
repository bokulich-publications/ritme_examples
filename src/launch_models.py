"""Launch ritme experiments for one or more (usecase, model_type) combos."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "src/run_ritme_model.sh"

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
}

SLURM_RESOURCES: dict[tuple[str, str], dict] = {
    ("u1", "linreg"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1", "rf"): {
        "cpus": 50,
        "mem_per_cpu_mb": 5120,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1", "trac"): {
        "cpus": 60,
        "mem_per_cpu_mb": 6144,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1", "xgb"): {
        "cpus": 50,
        "mem_per_cpu_mb": 4096,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1", "nn_reg"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1", "nn_class"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1", "nn_corn"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1_dynamic", "linreg"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1_dynamic", "rf"): {
        "cpus": 50,
        "mem_per_cpu_mb": 5120,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1_dynamic", "trac"): {
        "cpus": 60,
        "mem_per_cpu_mb": 6144,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1_dynamic", "xgb"): {
        "cpus": 50,
        "mem_per_cpu_mb": 4096,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1_dynamic", "nn_reg"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1_dynamic", "nn_class"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u1_dynamic", "nn_corn"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u2", "linreg"): {
        "cpus": 50,
        "mem_per_cpu_mb": 4096,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u2", "rf"): {
        "cpus": 60,
        "mem_per_cpu_mb": 6144,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u2", "trac"): {
        "cpus": 80,
        "mem_per_cpu_mb": 8192,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u2", "xgb"): {
        "cpus": 60,
        "mem_per_cpu_mb": 6144,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u2", "nn_reg"): {
        "cpus": 50,
        "mem_per_cpu_mb": 4096,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u2", "nn_class"): {
        "cpus": 50,
        "mem_per_cpu_mb": 4096,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u2", "nn_corn"): {
        "cpus": 50,
        "mem_per_cpu_mb": 4096,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u3", "logreg"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u3", "rf_class"): {
        "cpus": 50,
        "mem_per_cpu_mb": 5120,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u3", "xgb_class"): {
        "cpus": 50,
        "mem_per_cpu_mb": 4096,
        "time_budget_s": 82800,
        "gpus": 0,
        "slurm_account": None,
    },
    ("u3", "nn_class"): {
        "cpus": 50,
        "mem_per_cpu_mb": 3072,
        "time_budget_s": 82800,
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


def _resolve_config_for_run(
    usecase: str,
    model_type: str,
    sampler: str,
    variant: Optional[str],
    logs_dir: Path,
    time_budget_s_default: Optional[int] = None,
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
        tag_suffix = f"_{variant}" if variant else ""
        cfg["experiment_tag"] = f"{prefix}_{model_type}_{sampler}{tag_suffix}"
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

    if time_budget_s_default is not None:
        cfg["time_budget_s"] = time_budget_s_default

    if config_overrides:
        cfg.update(config_overrides)

    resolved_dir = logs_dir / "_resolved_configs"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolved_dir / f"{cfg['experiment_tag']}.json"
    resolved.write_text(json.dumps(cfg, indent=2) + "\n")
    return resolved


def _build_env(usecase: str, config_path: Path, logs_dir: Path) -> dict:
    spec = USECASES[usecase]
    env = os.environ.copy()
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
    logs_path = Path(logs_dir)
    if not logs_path.is_absolute():
        logs_path = REPO_ROOT / logs_path
    logs_path.mkdir(parents=True, exist_ok=True)

    res = SLURM_RESOURCES.get((usecase, model_type))
    time_budget_s_default = res.get("time_budget_s") if res else None

    config_path = _resolve_config_for_run(
        usecase,
        model_type,
        sampler,
        variant,
        logs_path,
        time_budget_s_default=time_budget_s_default,
        config_overrides=config_overrides,
        max_concurrent_trials=max_concurrent_trials,
    )
    env = _build_env(usecase, config_path, logs_path)

    if mode == "local":
        cmd = ["bash", str(TEMPLATE)]
        return subprocess.run(cmd, env=env, check=True)

    if mode != "slurm":
        raise ValueError(f"Unknown mode: {mode!r}")

    if res is None:
        raise KeyError(
            f"No SLURM allocation registered for {(usecase, model_type)!r}. "
            f"Add an entry to src/launch_models.py:SLURM_RESOURCES or pass "
            f"cpus= and mem_per_cpu_mb= explicitly on submit_model."
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
    if slurm_time is None:
        resolved_cfg = json.loads(config_path.read_text())
        slurm_time = _default_slurm_time(resolved_cfg["time_budget_s"])
    job_name = config_path.stem
    out_log = logs_path / "logs" / f"{job_name}_out.txt"
    out_log.parent.mkdir(parents=True, exist_ok=True)

    forwarded = ",".join(
        sorted(
            k
            for k in env
            if k
            in {
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
            }
        )
    )

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
    cmd.append(str(TEMPLATE))

    print("submitting:", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, env=env, check=True)
