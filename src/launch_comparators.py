"""Launch the TPOT and mAML comparator arms (mirror of `launch_automl.py`).

Each worker runs in its own conda environment, so the sbatch payload is
wrapped in `mamba run -n <env>`; submit from the `ritme_usecases` env.
Data paths, prediction target and metadata enrichment all come from the same
sources the ritme and auto-sklearn arms use.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src import cluster_config
from src.launch_automl import (
    _ensure_qza_converted,
    _read_enrich_with,
    _read_target,
)
from src.launch_models import REPO_ROOT, USECASES, _default_slurm_time

METHOD_ENVS = {"tpot": "tpot_bench", "maml": "maml_bench"}

_DEFAULT_CPUS = 50
_DEFAULT_MEM_PER_CPU_MB = 4096

# Site-specific; read from .cluster.json / the environment, never hardcoded.
# See src/cluster_config.py.
_UNSET = object()


def ensure_parquet_splits() -> list[Path]:
    """Write parquet copies of every `data_splits_*` pickle.

    The comparator environments run NumPy 1.x and cannot read pickles written
    under NumPy 2.x. Run this from `ritme_usecases` before submitting jobs.
    """
    written = []
    seen: set[Path] = set()
    for spec in USECASES.values():
        splits_dir = REPO_ROOT / spec["data_splits"]
        if splits_dir in seen or not splits_dir.is_dir():
            continue
        seen.add(splits_dir)
        for name in ("train_val", "test"):
            parquet = splits_dir / f"{name}.parquet"
            pkl = splits_dir / f"{name}.pkl"
            if parquet.exists() or not pkl.exists():
                continue
            pd.read_pickle(pkl).to_parquet(parquet, index=True)
            written.append(parquet)
            print(f"Converted {parquet}")
    if not written:
        print("All splits already have parquet copies.")
    return written


def _worker_cmd(
    usecase: str,
    method: str,
    total_time_s: int,
    logs_dir: Path,
    seed: int,
    cpus: int,
    restricted_model: Optional[str],
    unrestricted: bool,
    max_eval_time_mins: int,
    checkpoint_dir: Optional[str] = None,
) -> list[str]:
    spec = USECASES[usecase]
    common = [
        "python",
        "-m",
        f"src.comparator_{method}",
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
        str(logs_dir),
    ]
    for feat in _read_enrich_with(usecase):
        common += ["--enrich-with", feat]

    if method == "tpot":
        common += [
            "--total-time-s",
            str(total_time_s),
            "--max-eval-time-mins",
            str(max_eval_time_mins),
        ]
        if checkpoint_dir:
            common += ["--checkpoint-dir", str(checkpoint_dir)]
        if unrestricted:
            common.append("--unrestricted")
        elif restricted_model:
            common += ["--restricted-model", restricted_model]
    return common


def submit_comparator(
    usecase: str,
    *,
    method: str = "tpot",
    total_time_s: int = 82_800,
    restricted_model: Optional[str] = None,
    unrestricted: bool = False,
    seed: int = 12,
    logs_dir: str | os.PathLike = "comparators",
    mode: str = "slurm",
    sbatch_extra: Optional[Iterable[str]] = None,
    cpus: Optional[int] = None,
    mem_per_cpu_mb: Optional[int] = None,
    slurm_time: Optional[str] = None,
    slurm_account: Optional[str] = _UNSET,  # type: ignore[assignment]
    node_constraint: Optional[str] = _UNSET,  # type: ignore[assignment]
    max_eval_time_mins: int = 30,
    checkpoint_dir: Optional[str] = None,
) -> subprocess.CompletedProcess | list[str]:
    """Submit (or run) one comparator arm for a use case.

    Parameters
    ----------
    method : "tpot" | "maml". mAML is classification-only.
    mode : "slurm" (default), "local" (run inline) or "dry-run" (print only).
    """
    if usecase not in USECASES:
        raise KeyError(f"Unknown usecase: {usecase!r}")
    if method not in METHOD_ENVS:
        raise KeyError(f"Unknown method: {method!r}. Options: {sorted(METHOD_ENVS)}")
    task = USECASES[usecase]["task"]
    if method == "maml" and task != "classification":
        raise ValueError(
            f"mAML is classification-only, but usecase {usecase!r} is {task!r}."
        )

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
    if slurm_time is None:
        slurm_time = _default_slurm_time(total_time_s)

    _ensure_qza_converted(usecase)

    logs_path = Path(logs_dir)
    if not logs_path.is_absolute():
        logs_path = REPO_ROOT / logs_path
    logs_path.mkdir(parents=True, exist_ok=True)

    inner = _worker_cmd(
        usecase,
        method,
        total_time_s,
        logs_path,
        seed,
        cpus,
        restricted_model,
        unrestricted,
        max_eval_time_mins,
        checkpoint_dir,
    )

    if mode == "local":
        return subprocess.run(inner, cwd=REPO_ROOT, check=True)
    if mode not in ("slurm", "dry-run"):
        raise ValueError(f"Unknown mode: {mode!r}")

    job_name = f"n6_{method}_{usecase}_{task}"
    out_log = logs_path / "logs" / f"{job_name}_out.txt"
    out_log.parent.mkdir(parents=True, exist_ok=True)

    wrapped = " ".join(
        shlex.quote(c) for c in ["mamba", "run", "-n", METHOD_ENVS[method], *inner]
    )
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

    printable = " ".join(shlex.quote(c) for c in sbatch_cmd)
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ensure-parquet",
        action="store_true",
        help="Write parquet copies of every data_splits_* pickle, then exit.",
    )
    args = p.parse_args()
    if args.ensure_parquet:
        ensure_parquet_splits()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
