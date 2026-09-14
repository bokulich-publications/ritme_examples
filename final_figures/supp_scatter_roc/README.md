# Supplementary figures: train vs. test performance per arm

One 3x2 figure per use case, `supp_<usecase>_train_test.{eps,pdf,png}`. Rows
are the three arms, columns the train and the held-out test split:

| Row | Arm |
|---|---|
| 1 | the published pipeline of the source study |
| 2 | the ritme run that wins on the validation metric |
| 3 | TPOT, the strongest autoML comparator |

u1 and u2 predict a continuous target and get true-vs-predicted scatter plots;
u3 and u4 classify and get ROC curves, binary for u3 and macro one-vs-rest with
the per-class curves behind it for u4.

## Reproducing

Two stages. The first writes per-sample predictions, the second draws from
them.

```bash
# from the repository root
final_figures/supp_scatter_roc/submit_predictions.sh          # all twelve arms
final_figures/supp_scatter_roc/submit_predictions.sh u3        # one use case
final_figures/supp_scatter_roc/submit_predictions.sh u3 ritme  # one arm

conda activate ritme_usecases
python -m final_figures.supp_scatter_roc.make_supp_figures
```

`submit_predictions.sh` reads the SLURM account from the untracked
`.cluster.json` and picks the environment and resources per arm. To run one
arm outside SLURM, activate its environment
(`ritme_usecases` for `original` and `ritme`, `tpot_bench` for `automl`) and
call the extractor directly:

```bash
python -m final_figures.supp_scatter_roc.extract_predictions u3 ritme
```

Predictions land in `predictions/<usecase>_<arm>.csv`:

```
sample_id, split, y_true, y_pred            regression (u1, u2)
sample_id, split, y_true, p_<class>, ...    classification (u3, u4)
```

## What is refit and what is not

Only the ritme arm has a persisted fitted model; it is reloaded and asked for
predictions, so those rows are the models of record. The published baselines
and TPOT persist a recipe but no fitted estimator, so both are refit here --
from the recipe in `n4_original_setup.ipynb` and from TPOT's exported pipeline
respectively, at the seeds the original runs used. Every refit reproduces the
test metric recorded for that arm.

u1's TPOT run crashed before exporting, so its arm is rebuilt from the
pareto-front checkpoints, the same recovery path
`src/comparator_tpot.py --recover-from` took originally.

auto-sklearn is not shown: it writes neither a fitted model nor a pipeline, so
its arm cannot be reconstructed without repeating the 23 h search.

`originals.py` reimplements EMP's caret pipeline for u4 (`nearZeroVar`,
`findCorrelation`, `mtry` tuning by Kappa under the oneSE rule) from
`use_cases/u4_amplicon_emp_classification/n4_original_setup.ipynb`; the tuning
results are read from the cache that notebook wrote.
