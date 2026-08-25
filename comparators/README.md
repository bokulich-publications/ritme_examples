# AutoML comparators

TPOT and mAML runs alongside the existing *auto-sklearn* arm, on the same
train/test splits, metadata enrichment and search budget as *ritme*.

| Method | Use cases | Estimator family | Budget |
|---|---|---|---|
| auto-sklearn | u1, u2, u3 | matches the ritme winner | 82800 s |
| TPOT 0.12.2 | u1, u2, u3 | matches the ritme winner | 82800 s |
| mAML search space | u3 | 11 scalers x 13 classifiers, full grid | none (fixed grid) |

## Environments

Create on a login node — compute nodes have no outbound internet.

```shell
mamba env create -f envs/tpot.yml -p $CONDA_ENVS/tpot_bench
mamba run -n tpot_bench pip install -e .

mamba env create -f envs/maml.yml -p $CONDA_ENVS/maml_bench
mamba run -n maml_bench pip install -e .
```

## Running

From the repo root, in the `ritme_usecases` env. All jobs are SLURM jobs on
`EPYC_7742` nodes, account `es_beere`.

```shell
# once: the comparator envs run NumPy 1.x and cannot read NumPy 2.x pickles
python -m src.launch_comparators --ensure-parquet
```

Then use `use_cases/n6_comparator_automl.ipynb`, or directly:

```python
from src.launch_comparators import submit_comparator

for usecase in ("u1", "u2", "u3"):
    submit_comparator(usecase, method="tpot", total_time_s=82800)
submit_comparator("u3", method="maml", total_time_s=82800)
```

`mode="dry-run"` prints the `sbatch` line without submitting.
`unrestricted=True` searches TPOT's full default configuration instead of one
estimator family.

The auto-sklearn arm keeps its own launcher, `use_cases/n5_generic_automl.ipynb`.

## Outputs

Per run, in `comparators/` (gitignored):

| File | Content |
|---|---|
| `<usecase>_<method>_metrics.csv` | one metric row, same columns as `automl/<usecase>_metrics.csv` |
| `<usecase>_<method>_configs.csv` | one row per evaluated configuration |
| `<usecase>_tpot_best_pipeline.py` | TPOT's exported winning pipeline |
| `<usecase>_<method>_best_{roc,true_vs_pred}.png` | fit of the winning model |

`evaluate_all_trials.ipynb` reads these via `comparator_metrics_dirs` and names
each experiment `<usecase>_<method>`.

## Limits

- **mAML is use case 3 only** and the arm re-implements its *published search
  space* against the ritme split; it is not the upstream CLI, which has no
  held-out-test path. Deviations are listed in `src/comparator_maml.py`.
  Do not label the result "mAML" unqualified.
- **mAML has no budget knob** — a fixed grid — so it is excluded from
  matched-budget comparisons. Report its measured wall-clock instead.
- **TPOT is pinned to 0.12.2**, the genetic-programming release. `pip install
  tpot` resolves to 1.x, a different engine.
- **`PolynomialFeatures` is dropped from TPOT's operator set.** At degree 2 it
  expands p features to p(p+1)/2 — 63.7M columns for u3, 635M for u2. Classic
  TPOT applies no per-evaluation memory cap, so one such individual OOM-kills
  the job. Restore with `--allow-infeasible-operators`.
- **`FeatureAgglomeration` never enters TPOT's search**, though it is listed in
  the config. TPOT 0.12.2 parameterises it with `affinity=`, which scikit-learn
  deprecated in 1.2 and removed in 1.4, so it cannot be constructed under the
  pinned sklearn 1.5. It appears in 0 of ~35 000 evaluated pipelines across all
  runs, and in 0 of 5 even when it is one of only two entries in the config.
  TPOT's effective preprocessing set here is therefore **16 operators, not 18**.
- **`fit_result` is excluded from u3's metadata enrichment**
  (`src/launch_automl.py:ENRICH_EXCLUDE`); it is a clinical screening readout
  for the predicted outcome. Archived ritme u3 results still include it, so the
  u3 row is not matched until ritme is rerun without it.
- Pre-enrichment auto-sklearn results are kept in `automl/archive_pre_enrichment/`.

## Matched allocation

Every arm gets 50 CPUs, 200 GB and an 82 800 s search budget on `EPYC_7742`,
matching the ritme runs. Two protocol differences remain and should be stated
wherever the numbers are reported:

- TPOT caps each evaluation (`--max-eval-time-mins`, 20 min; 120 min for u1,
  where a shorter cap crashed the job — see below). ritme has no per-trial cap.
- mAML has no time budget at all: its grid is exhaustive (781 configurations)
  and it terminates when complete. Report its measured wall-clock instead.

TPOT's `stopit` timeout raises asynchronously and corrupted the heap when it
landed inside XGBoost's C code, killing a 10.7 h u1 job with `double free or
corruption`. u1 therefore runs with a cap generous enough to fire rarely.
