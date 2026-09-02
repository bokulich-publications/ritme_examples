# Batch evaluation of ritme trials (u3 / u4)

Runs `use_cases/evaluate_all_trials.ipynb` headless on a compute node via
papermill, once per usecase. The notebook's `USER INPUTS` cell is tagged
`parameters`, so each sbatch file injects its usecase, sampler, task and
`experiment_tags_to_compare` scope (u3: `^u3_.*_tpe_no_fit$`,
u4: `^u4_.*_tpe$`).

## One-time setup

Same environment as the `n2_run_ritme_model.ipynb` notebooks, **plus
`papermill`** (missing from the env line documented there) and the Jupyter
kernel registration papermill's `-k` flag needs. From the repo root:

```bash
mamba create -n ritme_usecases \
  -c adamova -c conda-forge -c bioconda -c pytorch \
  ritme ipykernel nbconvert papermill -y
conda activate ritme_usecases
pip install -e .
python -m ipykernel install --user --name ritme_usecases
```

The SLURM account is read from the untracked `.cluster.json` at the repo
root (`{"slurm_account": "..."}`); nothing site-specific is hardcoded in
these tracked files.

## Usage

```bash
src/eval_jobs/submit_eval.sh          # submit u3 and u4
src/eval_jobs/submit_eval.sh u4       # one usecase only
```

When today's `use_cases/all_experiments_metrics_<yymmdd>.csv` cache does
not exist yet, the wrapper chains u4 behind u3 (`afterany`) so the two
jobs don't race on writing it; delete that CSV to force a re-merge.

## Outputs

- figures: `use_cases/result_figures/<usecase>/` (gitignored)
- executed notebooks + SLURM logs: `x_scratch/eval_trials/` (gitignored)
- merged metrics cache: `use_cases/all_experiments_metrics_<yymmdd>.csv`
  (gitignored)

## Resources

u3 runs in minutes at 4 x 8G. u4 requests 16 x 28G = 448G because its
macro-OvR ROC cell runs the best model's feature engineering +
`predict_proba` per split of `data_splits_u4`, whose `test.pkl` alone is
~54G in memory (`train_val.pkl` ~4.4G) while the best trial keeps ~296k of
315k features (`variance_topi`) — a 192G allocation was OOM-killed there.
Prediction must see each split whole: `variance_topi` recomputes variances
on the inference batch, so chunked prediction would change the
probabilities. The `node_constraint` from `.cluster.json` is deliberately
not applied — evaluation is not timing-sensitive and schedules faster
without it.
