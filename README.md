# ritme_examples
This repos contains fully reproducible workflows to reproduce the use cases accompanying the [*ritme* manuscript](https://doi.org/10.64898/2025.12.08.693045).

## Setup
Each use case has different dependencies - you can find the instructions to set up the necessary conda environments at the start of each notebook.

## AutoML comparators

Three generic AutoML frameworks run against the same train/test splits and
metadata enrichment as *ritme*:

| Notebook | Arms | Use cases |
|---|---|---|
| `use_cases/n5_generic_automl.ipynb` | auto-sklearn | u1–u4 |
| `use_cases/n6_comparator_automl.ipynb` | TPOT 0.12.2, mAML | u1–u4 (TPOT), u3–u4 (mAML) |

auto-sklearn and TPOT pin their estimator family to *ritme*'s
validation-selected winner, leave the preprocessing search open, and share
*ritme*'s budget of 50 CPUs, 200 GB and 82 800 s. mAML runs its published
pipeline unchanged over its own fixed grid, which is exhaustive and so takes no
budget; it is classification-only, hence use cases 3 and 4.

Outputs land in `automl/` and `comparators/` (both gitignored);
`evaluate_all_trials.ipynb` reads them and `src/collect_comparison.py` builds the
comparison table, selecting *ritme*'s model on validation.

The SLURM account and node type are site-specific: copy `.cluster.example.json`
to `.cluster.json` (gitignored), or set `RITME_SLURM_ACCOUNT` /
`RITME_NODE_CONSTRAINT`. Leaving both unset uses the cluster's defaults.

## Contact

In case of questions or comments feel free to raise an issue in this repository.

## License

This repository  is released under a BSD-3-Clause license. See LICENSE for more details.
