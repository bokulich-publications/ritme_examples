"""A joblib Memory that never persists, importable so pickles can find it.

Used only by `comparator_maml_run.py`, but it cannot live there. Two
constraints pin it to its own module:

- It must be a **class**. scikit-learn 0.21 subclasses `joblib.Memory` at
  import time (`sklearn/utils/__init__.py`), so replacing the name with a
  factory function raises ``TypeError: function() argument 1 must be code``
  before any mAML code runs.
- It must be **importable**. The fitted pipeline keeps its `memory` argument
  and `joblib.dump` pickles it, so a class defined in the runner's `__main__`
  is unfindable once `runpy` swaps `__main__` for the mAML script's namespace:
  ``PicklingError: Can't pickle <class '__main__._NoCacheMemory'>``.

Both the runner and `comparator_maml_score.py` sit next to this file, and
Python puts a script's own directory on `sys.path`, so the reference resolves
when the model is written and again when it is read back.
"""

from __future__ import print_function

from joblib import Memory as _Memory


class NoCacheMemory(_Memory):
    """Accepts upstream's ``Memory("./mycache")`` call and caches nothing.

    Disabling the cache avoids a race: under the pinned joblib,
    `sklearn.pipeline._fit_transform_one` cannot be resolved to a source line,
    so joblib clears and rewrites the cache directory on every check and
    parallel workers collide on it. The cache only memoizes fitted transformers
    between grid points, so switching it off changes runtime, never results.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("location", None)
        kwargs.pop("cachedir", None)
        _Memory.__init__(self, location=None, verbose=kwargs.get("verbose", 0))
