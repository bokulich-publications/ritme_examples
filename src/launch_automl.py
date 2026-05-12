"""Launch the auto-sklearn baseline for a use case (mirror of `launch_models.py`).

Reuses the per-use-case data paths from :data:`launch_models.USECASES` and
reads the prediction target from the corresponding base config so that one
call replaces a whole `n5_generic_automl_<usecase>.sh` SLURM script.

The autoML run is implemented in :mod:`src.generic_automl`, which has its
own `autosklearn` import; this helper just builds the right CLI invocation
and (optionally) wraps it in `sbatch`. The conda environment with
`auto-sklearn` must already be active when calling this — the helper does
not switch envs.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from src.launch_models import REPO_ROOT, USECASES, _default_slurm_time


def _read_target(usecase: str) -> str:
    spec = USECASES[usecase]
    base_prefix = spec.get("base_config_prefix", spec["config_prefix"])
    base = REPO_ROOT / spec["use_case_dir"] / "config" / f"{base_prefix}_base_tpe.json"
    return json.loads(base.read_text())["target"]


def _ensure_qza_converted(usecase: str) -> None:
    """Materialize the .tsv/.nwk versions of any QIIME2 inputs ritme needs."""
    spec = USECASES[usecase]
    for kind, src, dst in spec["qza_inputs"]:
        dst_abs = REPO_ROOT / dst
        if dst_abs.exists():
            continue
        subprocess.run(
            [
                "python",
                "-m",
                "src.convert_qiime2_artifacts",
                kind,
                str(REPO_ROOT / src),
                "-o",
                str(dst_abs),
            ],
            check=True,
        )


_DEFAULT_AUTOML_CPUS: int = 100
_DEFAULT_AUTOML_MEM_PER_CPU_MB: int = 4096


def submit_automl(
    usecase: str,
    *,
    total_time_s: int = 428_400,
    restricted_model: str = "mlp",
    logs_dir: str | os.PathLike = "use_cases/ritme_runs/local_automl",
    mode: str = "slurm",
    sbatch_extra: Optional[Iterable[str]] = None,
    cpus: Optional[int] = None,
    mem_per_cpu_mb: Optional[int] = None,
    slurm_time: Optional[str] = None,
    slurm_account: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Submit (or run locally) the auto-sklearn baseline for a use case.

    Parameters
    ----------
    usecase : "u1" | "u2" | "u3" | "u3_legacy"
    total_time_s : auto-sklearn search budget (seconds). Default 428400
        (~119 h) fills the 120 h SLURM tier; lower it for smoke tests.
    restricted_model : auto-sklearn estimator name. ``mlp``, ``random_forest``
        and ``gradient_boosting`` are valid for both regression and
        classification tasks; ``ard_regression`` is regression-only.
    logs_dir : parent dir for the auto-sklearn output and SLURM log.
    mode : "slurm" (default) submits via sbatch; "local" runs inline.
    cpus, mem_per_cpu_mb : per-job SLURM allocation. ``None`` (default) uses
        ``_DEFAULT_AUTOML_CPUS=100`` / ``_DEFAULT_AUTOML_MEM_PER_CPU_MB=4096``.
        Pass explicit values to match ``launch_models.SLURM_RESOURCES`` for
        an apples-to-apples resource comparison against ritme. Slurm-mode
        only — ignored when ``mode="local"``.
    slurm_time : SLURM walltime ``HH:MM:SS``. ``None`` (default) snaps
        ``total_time_s`` to the smallest cluster tier (see
        ``launch_models._default_slurm_time``).
    slurm_account : value for sbatch ``--account=...`` (a.k.a. SLURM share).
        ``None`` (default) leaves it unset and uses the cluster default.
    """
    if cpus is None:
        cpus = _DEFAULT_AUTOML_CPUS
    if mem_per_cpu_mb is None:
        mem_per_cpu_mb = _DEFAULT_AUTOML_MEM_PER_CPU_MB
    if cpus <= 0 or mem_per_cpu_mb <= 0:
        raise ValueError(
            f"cpus and mem_per_cpu_mb must be positive; got "
            f"cpus={cpus}, mem_per_cpu_mb={mem_per_cpu_mb}."
        )
    if slurm_time is None:
        slurm_time = _default_slurm_time(total_time_s)
    if usecase not in USECASES:
        raise KeyError(f"Unknown usecase: {usecase!r}")
    _ensure_qza_converted(usecase)

    spec = USECASES[usecase]
    target = _read_target(usecase)
    task = spec["task"]

    logs_path = Path(logs_dir)
    if not logs_path.is_absolute():
        logs_path = REPO_ROOT / logs_path
    logs_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "src.generic_automl",
        "--total-time-s",
        str(total_time_s),
        "--usecase",
        usecase,
        "--task",
        task,
        "--data-splits-folder",
        str(REPO_ROOT / spec["data_splits"]),
        "--path-to-features",
        str(REPO_ROOT / spec["path_ft"]),
        "--path-to-md",
        str(REPO_ROOT / spec["path_md"]),
        "--target",
        target,
        "--single-best",
        "--restricted-model",
        restricted_model,
    ]

    if mode == "local":
        return subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    if mode != "slurm":
        raise ValueError(f"Unknown mode: {mode!r}")

    job_name = f"n5_automl_{usecase}_{task}_{restricted_model}"
    out_log = logs_path / "logs" / f"{job_name}_out.txt"
    out_log.parent.mkdir(parents=True, exist_ok=True)

    wrapped = " ".join(shlex.quote(c) for c in cmd)
    sbatch_cmd = [
        "sbatch",
        f"--job-name={job_name}",
        "--ntasks=1",
        f"--cpus-per-task={cpus}",
        f"--mem-per-cpu={mem_per_cpu_mb}",
        f"--time={slurm_time}",
        f"--output={out_log}",
        "--open-mode=append",
        f"--chdir={REPO_ROOT}",
        f"--wrap={wrapped}",
    ]
    if slurm_account:
        sbatch_cmd.insert(1, f"--account={slurm_account}")
    if sbatch_extra:
        sbatch_cmd[1:1] = list(sbatch_extra)
    print("submitting:", " ".join(shlex.quote(c) for c in sbatch_cmd))
    return subprocess.run(sbatch_cmd, check=True)
