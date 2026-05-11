"""Launch ritme experiments for one or more (usecase, model_type) combos.

Centralizes the per-use-case data paths and per-(usecase, model) SLURM
resources that previously had to be duplicated across one shell script + one
JSON config per (usecase, model). The `n2_run_ritme_model.ipynb` notebooks
call `submit_model(...)` from here so a single template
(`src/run_ritme_model.sh`) plus one base JSON per use case cover every
standard combination; genuinely distinct hyperparameter spaces are preserved
as `<prefix>_<model>_<sampler>_<variant>.json` files alongside the base.

Two execution modes:
- ``mode="slurm"`` (default): submit via ``sbatch`` with per-(usecase, model)
  resource defaults from :data:`SLURM_RESOURCES` and per-model Ray-Tune
  concurrency from :data:`MAX_CONCURRENT_TRIALS`. Cluster-only.
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
# kinds (see src/convert_qiime2_artifacts.py). SLURM resources (cpus /
# mem_per_cpu_mb) and Ray-Tune concurrency live in the SLURM_RESOURCES and
# MAX_CONCURRENT_TRIALS dicts below, keyed by (usecase, model_type).
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
        # CRC classification (Topcuoğlu 2020 / Baxter 2016). Both feature
        # tables ship pre-processed in the SchlossLab repos: ritme consumes
        # the un-rarefied table, the original baseline consumes the
        # subsampled one — mirrors the U1 pattern. `stratify_by="srn"`
        # preserves the ~47% positive-class prevalence in both train and
        # test (each patient is one sample, so no group split needed).
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
        "qza_inputs": [],
    },
    "u3_legacy": {
        # Former U3 (microbial load / Nishijima 2024 / MLP regression).
        # Parked under this key while U3 = CRC classification is evaluated;
        # remove this entry once the new U3 is confirmed as the keeper.
        # Configs are named "u3_galaxy_log_<...>.json" — keep that prefix
        # so the launcher can resolve them.
        "config_prefix": "u3_galaxy_log",
        "use_case_dir": "use_cases/u3_mlp_prediction",
        "data_splits": "use_cases/u3_mlp_prediction/data_splits_u3_galaxy_log",
        "path_md": "data/u3_mlp_nishijima24/md_galaxy.tsv",
        "path_ft": "data/u3_mlp_nishijima24/galaxy_otu_table.tsv",
        "path_tax": "data/u3_mlp_nishijima24/u3_taxonomy.tsv",
        "path_phylo": None,
        "group_by_column": None,
        "stratify_by": None,
        "task": "regression",
        "qza_inputs": [
            (
                "taxonomy",
                "data/u3_mlp_nishijima24/u3_taxonomy.qza",
                "data/u3_mlp_nishijima24/u3_taxonomy.tsv",
            ),
        ],
    },
}

# Per-(usecase, model class) SLURM allocation (cpus + memory per CPU).
# One row per actually-used (usecase, model) pair; submit_model raises a
# KeyError on any other combination. Inline `# bump` / `# tighten` markers
# call out deviations from a baseline shape of 30/3072 (linear), 40/5120
# (rf), 50/5120 (trac), 50/4096 (xgb), 100/4096 (nn_*).
SLURM_RESOURCES: dict[tuple[str, str], dict] = {
    # u1 (~850 OTUs after cleaning, repeated measures per host)
    ("u1", "linreg"): {"cpus": 30, "mem_per_cpu_mb": 3072},
    ("u1", "rf"): {"cpus": 40, "mem_per_cpu_mb": 5120},
    ("u1", "trac"): {"cpus": 50, "mem_per_cpu_mb": 5120},
    ("u1", "xgb"): {"cpus": 50, "mem_per_cpu_mb": 4096},
    ("u1", "nn_reg"): {"cpus": 100, "mem_per_cpu_mb": 4096},
    ("u1", "nn_class"): {"cpus": 100, "mem_per_cpu_mb": 4096},
    ("u1", "nn_corn"): {"cpus": 100, "mem_per_cpu_mb": 4096},
    # u2 (~36k features — trac matrix A + NN input layer need more memory)
    ("u2", "linreg"): {"cpus": 30, "mem_per_cpu_mb": 3072},
    ("u2", "rf"): {"cpus": 40, "mem_per_cpu_mb": 5120},
    ("u2", "trac"): {"cpus": 50, "mem_per_cpu_mb": 6144},  # bump
    ("u2", "xgb"): {"cpus": 50, "mem_per_cpu_mb": 4096},
    ("u2", "nn_reg"): {"cpus": 100, "mem_per_cpu_mb": 6144},  # bump
    ("u2", "nn_class"): {"cpus": 100, "mem_per_cpu_mb": 6144},  # bump
    ("u2", "nn_corn"): {"cpus": 100, "mem_per_cpu_mb": 6144},  # bump
    # u3 (~11k features, CRC binary classification — classifier model set)
    ("u3", "logreg"): {"cpus": 30, "mem_per_cpu_mb": 3072},
    ("u3", "rf_class"): {"cpus": 40, "mem_per_cpu_mb": 5120},
    ("u3", "xgb_class"): {"cpus": 50, "mem_per_cpu_mb": 4096},
    ("u3", "nn_class"): {"cpus": 100, "mem_per_cpu_mb": 4096},
    # u3_legacy (~2k features — narrower data fits in less memory)
    ("u3_legacy", "linreg"): {"cpus": 30, "mem_per_cpu_mb": 2048},  # tighten
    ("u3_legacy", "rf"): {"cpus": 40, "mem_per_cpu_mb": 3072},  # tighten
    ("u3_legacy", "xgb"): {"cpus": 50, "mem_per_cpu_mb": 3072},  # tighten
    ("u3_legacy", "nn_reg"): {"cpus": 100, "mem_per_cpu_mb": 3072},  # tighten
}

# Number of trials ritme runs in parallel within one sbatch job (applies in
# both slurm and local modes — controls ritme's Ray Tune concurrency, not
# the sbatch allocation). Does not currently vary by usecase. The
# (mis-)spelling `max_cuncurrent_trials` matches ritme's config key so the
# dict can be merged into the resolved config directly.
MAX_CONCURRENT_TRIALS: dict[str, int] = {
    "linreg": 80,
    "rf": 80,
    "trac": 80,
    "xgb": 80,
    "nn_reg": 10,
    "nn_class": 10,
    "nn_corn": 10,
    "logreg": 80,
    "rf_class": 80,
    "xgb_class": 80,
}


def _resolve_config_for_run(
    usecase: str,
    model_type: str,
    sampler: str,
    variant: Optional[str],
    logs_dir: Path,
    config_overrides: Optional[dict] = None,
    max_concurrent_trials: Optional[int] = None,
) -> Path:
    """Return the JSON config to hand to ritme for this run.

    Variants are kept as standalone files (`<prefix>_<model>_<sampler>_<variant>.json`)
    and consumed verbatim. Without a variant, the per-usecase base config is
    augmented with `max_cuncurrent_trials` (from ``max_concurrent_trials``
    if provided, otherwise :data:`MAX_CONCURRENT_TRIALS[model_type]`);
    the resolved config is materialized into
    `<logs_dir>/_resolved_configs/` for inspection.

    Synthetic variant ``"no_enrich"`` derives from the base config the same
    way the no-variant path does, then strips
    ``model_hyperparameters.data_enrich_with`` so ritme runs without
    metadata enrichment. The experiment_tag gains a ``_no_enrich`` suffix
    so its outputs land in their own directory and are tracked separately
    from the enriched runs.

    Named variants (anything other than ``"no_enrich"``) are consumed
    verbatim from disk: ``max_cuncurrent_trials`` is only injected when
    ``max_concurrent_trials`` is explicitly passed, so the variant config
    must encode its own concurrency if a non-default is needed.
    """
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
        base_path = config_dir / f"{prefix}_base_{sampler}.json"
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
    if spec["qza_inputs"]:
        env["QZA_INPUTS"] = " ".join(
            f"{kind}:{REPO_ROOT / src}:{REPO_ROOT / dst}"
            for kind, src, dst in spec["qza_inputs"]
        )
    return env


# Available SLURM walltime tiers on the target cluster (seconds). Jobs are
# placed in the smallest tier that fits `time_budget_s + SHAP/bootstrap`,
# so short search budgets queue against the short-walltime partition
# rather than the 120 h one.
SLURM_WALLTIME_TIERS_S: list[int] = [
    4 * 3600,  # 4 h
    24 * 3600,  # 24 h
    120 * 3600,  # 120 h
]
# Conservative floor for the SHAP + bootstrap step that runs after
# `find-best-model-config` exits. Lift it (or pass `slurm_time=` explicitly
# on submit_model) if your SHAP background is unusually large.
_SHAP_BOOTSTRAP_BUFFER_S: int = 3600  # 1 h


def _seconds_to_slurm_time(seconds: int) -> str:
    """Format ``seconds`` as the SLURM ``HH:MM:SS`` walltime string."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _default_slurm_time(time_budget_s: int) -> str:
    """Snap a ritme ``time_budget_s`` (+ SHAP/bootstrap buffer) to the
    smallest entry of :data:`SLURM_WALLTIME_TIERS_S` that fits.

    Raises :class:`ValueError` if even the largest tier is too small —
    reduce ``time_budget_s`` or pass ``slurm_time=`` directly on the
    submit_model call to bypass this rule.
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
    """Submit (or run locally) a single ritme experiment.

    Parameters
    ----------
    usecase : "u1" | "u2" | "u3" | "u3_legacy"
    model_type : ritme model class (e.g. "linreg", "xgb", "rf", "trac",
        "nn_reg", "nn_class", "nn_corn", "logreg", "rf_class", "xgb_class").
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
    slurm_time : SLURM walltime ``HH:MM:SS``. ``None`` (default) auto-derives
        it from the resolved config's ``time_budget_s`` via
        :func:`_default_slurm_time` so short runs get short walltimes and
        queue accordingly. Pass an explicit string to override.
    slurm_account : value passed to sbatch ``--account=...`` (a.k.a. SLURM
        share). When ``None`` (default) the cluster's default account
        applies.
    cpus, mem_per_cpu_mb : override the per-job SLURM allocation. ``None``
        (default) inherits ``SLURM_RESOURCES[(usecase, model_type)]``;
        raises ``KeyError`` in slurm mode if no entry is registered and
        neither override is passed. Ignored when ``mode="local"``.
    max_concurrent_trials : override the per-model Ray-Tune concurrency
        baked into the resolved ritme config. ``None`` (default) inherits
        ``MAX_CONCURRENT_TRIALS[model_type]``; raises ``KeyError`` if no
        entry is registered. Applies in both modes.
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
        max_concurrent_trials=max_concurrent_trials,
    )
    env = _build_env(usecase, config_path, logs_path)

    if mode == "local":
        cmd = ["bash", str(TEMPLATE)]
        return subprocess.run(cmd, env=env, check=True)

    if mode != "slurm":
        raise ValueError(f"Unknown mode: {mode!r}")

    try:
        res = SLURM_RESOURCES[(usecase, model_type)]
    except KeyError as e:
        raise KeyError(
            f"No SLURM allocation registered for {(usecase, model_type)!r}. "
            f"Add an entry to src/launch_models.py:SLURM_RESOURCES or pass "
            f"cpus= and mem_per_cpu_mb= explicitly on submit_model."
        ) from e
    cpus = res["cpus"] if cpus is None else cpus
    mem_per_cpu_mb = res["mem_per_cpu_mb"] if mem_per_cpu_mb is None else mem_per_cpu_mb
    if cpus <= 0 or mem_per_cpu_mb <= 0:
        raise ValueError(
            f"cpus and mem_per_cpu_mb must be positive; got "
            f"cpus={cpus}, mem_per_cpu_mb={mem_per_cpu_mb}."
        )
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
    if slurm_account:
        cmd.insert(1, f"--account={slurm_account}")
    if sbatch_extra:
        cmd[1:1] = list(sbatch_extra)
    cmd.append(str(TEMPLATE))

    print("submitting:", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, env=env, check=True)
