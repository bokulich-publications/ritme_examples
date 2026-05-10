"""Launch ritme experiments for one or more (usecase, model_type) combos.

Centralizes the per-use-case data paths and per-model SLURM resource defaults
that previously had to be duplicated across one shell script + one JSON config
per (usecase, model). The `n2_run_ritme_model.ipynb` notebooks call
`submit_model(...)` from here so a single template (`src/run_ritme_model.sh`)
plus one base JSON per use case cover every standard combination; genuinely
distinct hyperparameter spaces are preserved as
`<prefix>_<model>_<sampler>_<variant>.json` files alongside the base.

Two execution modes:
- ``mode="slurm"`` (default): submit via ``sbatch`` with per-model resource
  defaults from :data:`MODEL_RESOURCES`. Cluster-only.
- ``mode="local"``: run the template inline in the current process — handy
  for short local smoke tests.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "src/run_ritme_model.sh"

# Per-use-case data + config paths. Each entry's `qza_inputs` lists the
# (kind, src_qza, dst_plain) triples that the template will convert before
# running ritme; `feature-table`, `taxonomy` and `tree` are the supported
# kinds (see src/convert_qiime2_artifacts.py). `model_overrides` lets a use
# case override per-model fields baked into the ritme JSON config (e.g., a
# tighter max_cuncurrent_trials for one model class). `slurm_overrides` does
# the same for SLURM resource fields (cpus / mem_per_cpu_mb), enabling
# usecase-specific sbatch sizing without forking MODEL_RESOURCES.
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
        "model_overrides": {},
        "slurm_overrides": {},
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
        "model_overrides": {},
        # u2 has the widest feature space (~36k features); trac's matrix A
        # and the NN input layer scale with feature count, so both benefit
        # from extra headroom per CPU.
        "slurm_overrides": {
            "trac": {"mem_per_cpu_mb": 6144},
            "nn_reg": {"mem_per_cpu_mb": 6144},
            "nn_class": {"mem_per_cpu_mb": 6144},
            "nn_corn": {"mem_per_cpu_mb": 6144},
        },
    },
    "u3": {
        # u3 configs are named "u3_galaxy_log_<...>.json" — keep that prefix
        # so the launcher can resolve them.
        "config_prefix": "u3_galaxy_log",
        "use_case_dir": "use_cases/u3_mlp_prediction",
        "data_splits": "use_cases/u3_mlp_prediction/data_splits_u3_galaxy_log",
        "path_md": "data/u3_mlp_nishijima24/md_galaxy.tsv",
        "path_ft": "data/u3_mlp_nishijima24/galaxy_otu_table.tsv",
        "path_tax": "data/u3_mlp_nishijima24/u3_taxonomy.tsv",
        "path_phylo": None,
        "group_by_column": None,
        "qza_inputs": [
            (
                "taxonomy",
                "data/u3_mlp_nishijima24/u3_taxonomy.qza",
                "data/u3_mlp_nishijima24/u3_taxonomy.tsv",
            ),
        ],
        "model_overrides": {},
        # u3 has the smallest feature space (~2k), so per-trial memory
        # budgets tighten vs the wider u1/u2 datasets — roughly 40-75% of
        # the MODEL_RESOURCES defaults, depending on model class.
        "slurm_overrides": {
            "linreg": {"mem_per_cpu_mb": 2048},
            "rf": {"mem_per_cpu_mb": 3072},
            "xgb": {"mem_per_cpu_mb": 3072},
            "trac": {"mem_per_cpu_mb": 2048},
            "nn_reg": {"mem_per_cpu_mb": 3072},
            "nn_class": {"mem_per_cpu_mb": 3072},
            "nn_corn": {"mem_per_cpu_mb": 3072},
        },
    },
}

# SLURM resource + parallelism defaults per model class — sized for the
# widest of the three datasets (u1/u2; ~18-36k features). Per-usecase
# `slurm_overrides` (in USECASES) tighten these where the data is smaller.
# `max_cuncurrent_trials` keeps ritme's own (mis-)spelling so the dict can be
# merged straight into the config. trac drops 14848 -> 5120 MB/CPU because
# v1.4.0 builds matrix A as a sparse CSC matrix (PR #110).
MODEL_RESOURCES: dict[str, dict] = {
    "linreg": {"cpus": 30, "mem_per_cpu_mb": 3072, "max_cuncurrent_trials": 80},
    "rf": {"cpus": 40, "mem_per_cpu_mb": 5120, "max_cuncurrent_trials": 80},
    "trac": {"cpus": 50, "mem_per_cpu_mb": 5120, "max_cuncurrent_trials": 80},
    "xgb": {"cpus": 50, "mem_per_cpu_mb": 4096, "max_cuncurrent_trials": 80},
    "nn_reg": {"cpus": 100, "mem_per_cpu_mb": 4096, "max_cuncurrent_trials": 10},
    "nn_class": {"cpus": 100, "mem_per_cpu_mb": 4096, "max_cuncurrent_trials": 10},
    "nn_corn": {"cpus": 100, "mem_per_cpu_mb": 4096, "max_cuncurrent_trials": 10},
}


def _resolve_config_for_run(
    usecase: str,
    model_type: str,
    sampler: str,
    variant: Optional[str],
    logs_dir: Path,
    config_overrides: Optional[dict] = None,
) -> Path:
    """Return the JSON config to hand to ritme for this run.

    Variants are kept as standalone files (`<prefix>_<model>_<sampler>_<variant>.json`)
    and consumed verbatim. Without a variant, the per-usecase base config is
    merged with model-class defaults (`MODEL_RESOURCES`) and any per-usecase
    `model_overrides`; the resolved config is materialized into
    `<logs_dir>/_resolved_configs/` for inspection.

    Synthetic variant ``"no_enrich"`` derives from the base config the same
    way the no-variant path does, then strips
    ``model_hyperparameters.data_enrich_with`` so ritme runs without
    metadata enrichment. The experiment_tag gains a ``_no_enrich`` suffix
    so its outputs land in their own directory and are tracked separately
    from the enriched runs.
    """
    spec = USECASES[usecase]
    config_dir = REPO_ROOT / spec["use_case_dir"] / "config"
    prefix = spec["config_prefix"]

    if variant and variant != "no_enrich":
        path = config_dir / f"{prefix}_{model_type}_{sampler}_{variant}.json"
        if not path.exists():
            raise FileNotFoundError(f"Variant config not found: {path}")
        if not config_overrides:
            return path
        cfg = json.loads(path.read_text())
    else:
        base_path = config_dir / f"{prefix}_base_{sampler}.json"
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")
        cfg = json.loads(base_path.read_text())
        tag_suffix = f"_{variant}" if variant else ""
        cfg["experiment_tag"] = f"{prefix}_{model_type}_{sampler}{tag_suffix}"
        cfg["ls_model_types"] = [model_type]
        res = MODEL_RESOURCES.get(model_type, {})
        if "max_cuncurrent_trials" in res:
            cfg["max_cuncurrent_trials"] = res["max_cuncurrent_trials"]
        cfg.update(spec.get("model_overrides", {}).get(model_type, {}))
        if variant == "no_enrich":
            mh = cfg.get("model_hyperparameters")
            if isinstance(mh, dict):
                mh.pop("data_enrich_with", None)
                if not mh:
                    cfg.pop("model_hyperparameters")

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
    if spec["qza_inputs"]:
        env["QZA_INPUTS"] = " ".join(
            f"{kind}:{REPO_ROOT / src}:{REPO_ROOT / dst}"
            for kind, src, dst in spec["qza_inputs"]
        )
    return env


def submit_model(
    usecase: str,
    model_type: str,
    *,
    sampler: str = "tpe",
    variant: Optional[str] = None,
    logs_dir: str | os.PathLike = "use_cases/ritme_runs/local",
    mode: str = "slurm",
    sbatch_extra: Optional[Iterable[str]] = None,
    slurm_time: str = "119:59:59",
    cpus: Optional[int] = None,
    mem_per_cpu_mb: Optional[int] = None,
    config_overrides: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    """Submit (or run locally) a single ritme experiment.

    Parameters
    ----------
    usecase : "u1" | "u2" | "u3"
    model_type : ritme model class (e.g. "linreg", "xgb", "rf", "trac",
        "nn_reg", "nn_class", "nn_corn").
    sampler : Optuna sampler tag baked into the config filename (default "tpe").
    variant : optional variant suffix (e.g., "restricted", "w_start"); if set,
        the launcher uses the corresponding standalone config file. The
        synthetic ``"no_enrich"`` variant derives from the base config and
        strips ``model_hyperparameters.data_enrich_with`` so the run uses
        no metadata enrichment; outputs land under
        ``<prefix>_<model>_<sampler>_no_enrich``.
    logs_dir : parent directory for the experiment output; resolved relative
        to the repo root if not absolute.
    mode : "slurm" submits via sbatch; "local" runs the template inline.
    sbatch_extra : extra sbatch flags inserted right after `sbatch`.
    cpus, mem_per_cpu_mb : override the per-model SLURM defaults. When not
        set, the resolved budget is MODEL_RESOURCES[model_type] overlaid
        with USECASES[usecase]["slurm_overrides"].get(model_type, {}).
    """
    logs_path = Path(logs_dir)
    if not logs_path.is_absolute():
        logs_path = REPO_ROOT / logs_path
    logs_path.mkdir(parents=True, exist_ok=True)

    config_path = _resolve_config_for_run(
        usecase,
        model_type,
        sampler,
        variant,
        logs_path,
        config_overrides=config_overrides,
    )
    env = _build_env(usecase, config_path, logs_path)

    if mode == "local":
        cmd = ["bash", str(TEMPLATE)]
        return subprocess.run(cmd, env=env, check=True)

    if mode != "slurm":
        raise ValueError(f"Unknown mode: {mode!r}")

    res = dict(MODEL_RESOURCES.get(model_type, {"cpus": 30, "mem_per_cpu_mb": 4096}))
    res.update(USECASES[usecase].get("slurm_overrides", {}).get(model_type, {}))
    cpus = cpus or res["cpus"]
    mem_per_cpu_mb = mem_per_cpu_mb or res["mem_per_cpu_mb"]
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
    if sbatch_extra:
        cmd[1:1] = list(sbatch_extra)
    cmd.append(str(TEMPLATE))

    print("submitting:", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, env=env, check=True)
