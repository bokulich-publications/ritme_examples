"""Run mAML's published CLI with its transformer cache disabled.

`sklearn_pipeline.py` builds `Pipeline(memory=Memory("./mycache"))`. Under the
pinned joblib, `sklearn.pipeline._fit_transform_one` cannot be resolved to a
source line (joblib reports line -1 and warns about a name collision), so
joblib decides the function changed on every check and clears then rewrites the
cache directory. At `-j 50` one worker deletes the directory while another
writes into it:

    FileNotFoundError: './mycache/joblib/sklearn/pipeline/
    _fit_transform_one/func_code.py'

The failure lands inside `select_best_scl_clf`, which upstream wraps in
`@tryExcept`, so the run continues to `hypertune_best_classifier`, fails again
on the missing `best_clf`, and exits having dumped no model.

The cache only memoizes fitted transformers between grid points, so switching
it off changes runtime, never results. Rather than edit the published script,
`joblib.Memory` is replaced before the script is imported -- it binds the name
with `from joblib import Memory` -- and the real `__main__` then runs untouched
through `runpy`. The replacement lives in `maml_nocache` rather than here; that
module's docstring explains why it cannot be defined in this file.
"""

from __future__ import print_function

import os
import runpy
import sys

import joblib

# Sibling module: a script's own directory is always on sys.path.
from maml_nocache import NoCacheMemory


def main():
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: comparator_maml_run.py <path to sklearn_pipeline.py> "
            "[CLI args ...]"
        )
    cli = os.path.abspath(sys.argv[1])
    if not os.path.exists(cli):
        raise SystemExit("mAML CLI not found: %s" % cli)

    joblib.Memory = NoCacheMemory
    # `utils` and `sklearn_pipeline_config` are siblings of the CLI.
    sys.path.insert(0, os.path.dirname(cli))
    sys.argv = [cli] + sys.argv[2:]
    print("running published mAML CLI with the transformer cache disabled")
    print("  ", " ".join(sys.argv))
    runpy.run_path(cli, run_name="__main__")


if __name__ == "__main__":
    main()
