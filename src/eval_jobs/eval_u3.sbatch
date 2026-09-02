#!/usr/bin/env bash
#SBATCH --job-name=eval_trials_u3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
# Runs use_cases/evaluate_all_trials.ipynb headless for usecase u3.
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
  -p usecase u3 \
  -p sampler tpe \
  -p task classification \
  -y 'experiment_tags_to_compare: ["^u3_.*_tpe_no_fit$"]' \
  evaluate_all_trials.ipynb \
  "$OUT_DIR/evaluate_all_trials_u3_executed.ipynb"
