"""Site-specific cluster settings, kept out of version control.

SLURM account names, node constraints and similar values identify a specific
institution's infrastructure and must never be committed. They are read at
runtime from, in order of precedence:

1. the environment (``RITME_SLURM_ACCOUNT``, ``RITME_NODE_CONSTRAINT``);
2. ``.cluster.json`` at the repo root, which is gitignored;
3. nothing -- the launchers then omit the flag entirely and SLURM applies its
   own defaults.

Copy `.cluster.example.json` to `.cluster.json` and fill in your own values.
"""

from __future__ import annotations

import json
import os
import shlex
from typing import Optional

from src.launch_models import REPO_ROOT

CLUSTER_CONFIG_PATH = REPO_ROOT / ".cluster.json"

_cache: Optional[dict] = None


def _load() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CLUSTER_CONFIG_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            _cache = {}
    return _cache


def get(key: str, default: Optional[str] = None) -> Optional[str]:
    """Return a site setting, or ``default`` when it is not configured."""
    env = os.environ.get(f"RITME_{key.upper()}")
    if env:
        return env
    value = _load().get(key)
    return value if value else default


def slurm_account() -> Optional[str]:
    """SLURM account for ``sbatch --account=``. None omits the flag."""
    return get("slurm_account")


def node_constraint() -> Optional[str]:
    """Node type for ``sbatch --constraint=``. None omits the flag."""
    return get("node_constraint")


#: sbatch flags whose values identify the institution's infrastructure.
_SECRET_FLAGS = ("--account=", "--constraint=")


def redact(parts) -> str:
    """Render an sbatch command with the site-specific values masked.

    The launchers echo the command they submit, and a notebook run with that
    output committed would put the values into version control. Printing goes
    through here so nothing has to remember to strip them.
    """
    out = []
    for part in parts:
        for flag in _SECRET_FLAGS:
            if part.startswith(flag):
                part = f"{flag}<from .cluster.json>"
                break
        out.append(shlex.quote(part))
    return " ".join(out)
