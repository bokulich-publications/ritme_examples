#!/usr/bin/env bash
#SBATCH --job-name=eval_trials_u4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=28G
#SBATCH --time=04:00:00
# Runs use_cases/evaluate_all_trials.ipynb headless for usecase u4.
# 16 x 28G = 448G (fits a standard 512G node; measured peak of the
# 2026-09-02 run: MaxRSS 443G — do NOT trim this request): the macro-OvR ROC cell runs
# the best model's feature engineering + predict_proba per split, and u4's
# test.pkl alone is ~54G in memory (train_val.pkl ~4.4G) while the best
# trial keeps ~296k of 315k features (variance_topi) — so the engineered
# test matrix is nearly full-width. A 192G run was OOM-killed there even
# though the notebook loads splits one at a time. Chunked prediction is NOT
# a valid fallback: variance_topi recomputes variances on the inference
# batch, so chunking changes the probabilities.
# Submit via submit_eval.sh, which supplies --account and --output and
# exports REPO_ROOT / CONDA_SH / OUT_DIR (kept out of #SBATCH directives so
# no site-specific value is hardcoded in this tracked file).
set -euo pipefail
: "${REPO_ROOT:?submit via submit_eval.sh (exports REPO_ROOT)}"
: "${CONDA_SH:?submit via submit_eval.sh (exports CONDA_SH)}"
OUT_DIR=${OUT_DIR:-$REPO_ROOT/x_scratch/eval_trials}
mkdir -p "$OUT_DIR"
source "$CONDA_SH"
conda activate ritme_usecases
cd "$REPO_ROOT/use_cases"
papermill -k ritme_usecases --log-output \
  -p usecase u4 \
  -p sampler tpe \
  -p task classification \
  -y 'experiment_tags_to_compare: ["^u4_.*_tpe$"]' \
  evaluate_all_trials.ipynb \
  "$OUT_DIR/evaluate_all_trials_u4_executed.ipynb"
