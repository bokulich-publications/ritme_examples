# ritme_examples
This repos contains fully reproducible workflows to reproduce the use cases accompanying the [*ritme* manuscript](https://doi.org/10.64898/2025.12.08.693045).

## Setup
Each use case has different dependencies - you can find the instructions to set up the necessary conda environments at the start of each notebook.

## AutoML comparators

Three generic AutoML frameworks run against the same train/test splits, metadata
enrichment and search budget as *ritme*:

| Notebook | Arms | Use cases |
|---|---|---|
| `use_cases/n5_generic_automl.ipynb` | auto-sklearn | u1, u2, u3 |
| `use_cases/n6_comparator_automl.ipynb` | TPOT 0.12.2, mAML | u1–u3 (TPOT), u3 (mAML) |

Each arm pins its estimator family to *ritme*'s validation-selected winner and
leaves the preprocessing search open. Every arm gets 50 CPUs, 200 GB and an
82 800 s budget. Outputs land in `automl/` and `comparators/` (both gitignored);
`evaluate_all_trials.ipynb` reads them and `src/collect_comparison.py` builds the
comparison table, selecting *ritme*'s model on validation.

The SLURM account and node type are site-specific: copy `.cluster.example.json`
to `.cluster.json` (gitignored), or set `RITME_SLURM_ACCOUNT` /
`RITME_NODE_CONSTRAINT`. Leaving both unset uses the cluster's defaults.

### Caveats

- `fit_result` is excluded from u3's enrichment as a clinical screening readout
  for the predicted outcome, so u3 is only matched against *ritme* runs made
  without it.
- The mAML arm re-implements its *published search space* against the *ritme*
  split; the upstream CLI has no held-out-test path. Do not label it "mAML"
  unqualified. Its grid is exhaustive, so it has no budget knob.
- TPOT is pinned to 0.12.2; `pip install tpot` resolves to 1.x, a different
  engine. `PolynomialFeatures` is dropped (no per-evaluation memory cap) and
  `FeatureAgglomeration` cannot be constructed under the pinned scikit-learn,
  leaving 16 searchable operators rather than 18.
- TPOT's search is wall-clock bounded and not reproducible run to run; a fixed
  `random_state` does not change that.

## Contact

In case of questions or comments feel free to raise an issue in this repository.

## License

This repository  is released under a BSD-3-Clause license. See LICENSE for more details.
