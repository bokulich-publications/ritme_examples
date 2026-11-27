"""Locate and pre-download the TabPFN checkpoints the comparator arm runs on.

Compute nodes have no internet access, so the weights are fetched once, on a
node that does, into a directory the jobs then read. Only importable from the
``tabpfn`` environment; `src.launch_tabpfn.ensure_models` shells out to the CLI::

    python -m src.tabpfn_models <model_dir> --version v2

TabPFN v2 downloads without an account. Later versions are gated behind the
Prior Labs license: accept it once at https://ux.priorlabs.ai and export
``TABPFN_TOKEN`` and ``HF_TOKEN`` before running this.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tabpfn.constants import ModelVersion
from tabpfn.model_loading import download_model, resolve_model_path

ESTIMATOR_TYPES = ("classifier", "regressor")


def checkpoint_path(model_dir: str | Path, version: str, which: str) -> Path:
    """Path of the default ``version`` checkpoint for ``which`` in ``model_dir``."""
    _, _, names, _ = resolve_model_path(None, which=which, version=version)
    return Path(model_dir) / names[0]


def download_checkpoints(model_dir: str | Path, version: str) -> list[Path]:
    """Download the default classifier and regressor checkpoints that are missing."""
    paths = []
    for which in ESTIMATOR_TYPES:
        path = checkpoint_path(model_dir, version, which)
        if not path.exists():
            result = download_model(path, version=ModelVersion(version), which=which)
            if result != "ok":
                raise result[0]
            print(f"Downloaded {path.name}")
        paths.append(path)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model_dir", type=Path)
    p.add_argument("--version", default="v2", choices=[v.value for v in ModelVersion])
    args = p.parse_args()
    for path in download_checkpoints(args.model_dir, args.version):
        print(path.name)


if __name__ == "__main__":
    main()
