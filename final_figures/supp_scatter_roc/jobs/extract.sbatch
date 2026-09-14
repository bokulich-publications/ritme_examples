#!/usr/bin/env bash
#SBATCH --job-name=supp_preds
#SBATCH --ntasks=1
# Runs one (use case, arm) of extract_predictions.py. Resources, the conda
# environment, --account and --output all come from submit_predictions.sh.
set -euo pipefail
: "${REPO_ROOT:?submit via submit_predictions.sh (exports REPO_ROOT)}"
: "${CONDA_SH:?submit via submit_predictions.sh (exports CONDA_SH)}"
: "${USECASE:?}" "${ARM:?}" "${ENV_NAME:?}"
source "$CONDA_SH"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"
python -m final_figures.supp_scatter_roc.extract_predictions "$USECASE" "$ARM"
