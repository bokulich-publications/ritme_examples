"""Launch the TabPFN comparator arm (mirror of `launch_comparators.py`).

The worker runs in the ``tabpfn`` conda environment, so the sbatch payload is
wrapped in ``mamba run -n tabpfn``; submit from any environment with this
package installed. Data paths, prediction target and metadata enrichment come
from the same sources the ritme and AutoML arms use.

TabPFN predicts from pretrained weights, which the compute nodes cannot fetch.
Call `ensure_models` once on a node with internet access; the jobs then read
the checkpoints from `MODEL_DIR`.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from src import cluster_config
from src.launch_automl import (
    _ensure_qza_converted,
    _read_enrich_with,
    _read_target,
)
from src.launch_models import REPO_ROOT, USECASES

TABPFN_ENV = "tabpfn"
MODEL_VERSION = "v2"
# Gitignored next to the other third-party downloads.
MODEL_DIR = REPO_ROOT / "external" / "tabpfn_models"

# Inference on a few hundred samples takes minutes; the allocation covers the
# widest table (u2, 35k features) comfortably.
_DEFAULT_CPUS = 8
_DEFAULT_MEM_PER_CPU_MB = 8192
_DEFAULT_SLURM_TIME = "04:00:00"

# Site-specific; read from .cluster.json / the environment, never hardcoded.
# See src/cluster_config.py.
_UNSET = object()


def method_name(max_features: Optional[int]) -> str:
    """Output file stem of an arm, shared with `src.comparator_tabpfn`."""
    return "tabpfn" if max_features is None else f"tabpfn_top{max_features}"


def metrics_path(
    usecase: str,
    max_features: Optional[int] = None,
    logs_dir: str | os.PathLike = "comparators",
) -> Path:
    """Where `submit_tabpfn(usecase, max_features=...)` writes its metrics."""
    logs_path = Path(logs_dir)
    if not logs_path.is_absolute():
        logs_path = REPO_ROOT / logs_path
    return logs_path / f"{usecase}_{method_name(max_features)}_metrics.csv"


def ensure_models(
    model_dir: str | os.PathLike = MODEL_DIR, model_version: str = MODEL_VERSION
) -> None:
    """Download the TabPFN checkpoints into ``model_dir`` unless already there.

    Needs internet access, so run it where that exists before submitting jobs.
    The worker's output is captured: it names absolute paths, which would
    otherwise end up in a notebook's saved output.
    """
    cmd = [
        "mamba",
        "run",
        "-n",
        TABPFN_ENV,
        "python",
        "-m",
        "src.tabpfn_models",
        str(model_dir),
        "--version",
        model_version,
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Checkpoint download failed:\n{proc.stderr[-2000:]}")
    print(
        f"TabPFN {model_version} checkpoints in {os.path.relpath(model_dir, REPO_ROOT)}"
    )


def _worker_cmd(
    usecase: str,
    max_features: Optional[int],
    seed: int,
    cpus: int,
    logs_path: Path,
    model_dir: Path,
    model_version: str,
) -> list[str]:
    spec = USECASES[usecase]
    cmd = [
        "python",
        "-m",
        "src.comparator_tabpfn",
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
        "--seed",
        str(seed),
        "--n-jobs",
        str(cpus),
        "--out-dir",
        str(logs_path),
        "--model-dir",
        str(model_dir),
        "--model-version",
        model_version,
    ]
    for feat in _read_enrich_with(usecase):
        cmd += ["--enrich-with", feat]
    if max_features is not None:
        cmd += ["--max-features", str(max_features)]
    return cmd


def submit_tabpfn(
    usecase: str,
    *,
    max_features: Optional[int] = None,
    seed: int = 12,
    logs_dir: str | os.PathLike = "comparators",
    mode: str = "slurm",
    sbatch_extra: Optional[Iterable[str]] = None,
    cpus: Optional[int] = None,
    mem_per_cpu_mb: Optional[int] = None,
    slurm_time: str = _DEFAULT_SLURM_TIME,
    slurm_account: Optional[str] = _UNSET,  # type: ignore[assignment]
    node_constraint: Optional[str] = _UNSET,  # type: ignore[assignment]
    model_dir: str | os.PathLike = MODEL_DIR,
    model_version: str = MODEL_VERSION,
) -> subprocess.CompletedProcess | list[str]:
    """Submit (or run) the TabPFN arm for a use case.

    Parameters
    ----------
    max_features : None feeds the full feature table; an int caps the design
        matrix at that many columns (metadata plus the most abundant
        microbial features). Output files are named after it, see
        `method_name`.
    mode : "slurm" (default), "local" (run inline) or "dry-run" (print only).
    """
    if usecase not in USECASES:
        raise KeyError(f"Unknown usecase: {usecase!r}")
    if slurm_account is _UNSET:
        slurm_account = cluster_config.slurm_account()
    if node_constraint is _UNSET:
        node_constraint = cluster_config.node_constraint()

    cpus = _DEFAULT_CPUS if cpus is None else cpus
    mem_per_cpu_mb = (
        _DEFAULT_MEM_PER_CPU_MB if mem_per_cpu_mb is None else mem_per_cpu_mb
    )
    if cpus <= 0 or mem_per_cpu_mb <= 0:
        raise ValueError(
            f"cpus and mem_per_cpu_mb must be positive; got "
            f"cpus={cpus}, mem_per_cpu_mb={mem_per_cpu_mb}."
        )

    _ensure_qza_converted(usecase)

    logs_path = Path(logs_dir)
    if not logs_path.is_absolute():
        logs_path = REPO_ROOT / logs_path
    logs_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_dir)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path

    inner = _worker_cmd(
        usecase, max_features, seed, cpus, logs_path, model_path, model_version
    )
    if mode == "local":
        return subprocess.run(inner, cwd=REPO_ROOT, check=True)
    if mode not in ("slurm", "dry-run"):
        raise ValueError(f"Unknown mode: {mode!r}")

    wrapped = " ".join(
        shlex.quote(c) for c in ["mamba", "run", "-n", TABPFN_ENV, *inner]
    )
    job_name = f"n7_{method_name(max_features)}_{usecase}_{USECASES[usecase]['task']}"
    out_log = logs_path / "logs" / f"{job_name}_out.txt"
    out_log.parent.mkdir(parents=True, exist_ok=True)
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
    if node_constraint:
        sbatch_cmd.insert(1, f"--constraint={node_constraint}")
    if sbatch_extra:
        sbatch_cmd[1:1] = list(sbatch_extra)

    # Absolute paths name the institution's storage, so the echo shows them
    # relative to the repository.
    printable = cluster_config.redact(sbatch_cmd).replace(str(REPO_ROOT), "<repo>")
    if mode == "dry-run":
        print("[dry-run] would submit:", printable)
        return sbatch_cmd

    print("submitting:", printable)
    proc = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip())
    return proc
