#!/usr/bin/env bash
# Minimal launcher: submit each model script as a separate SLURM job.
# Order is fixed and no dependencies are enforced.
# Usage: ./submit_all.sh

set -euo pipefail

scripts=(
  n2_model_linreg_random.sh
  n2_model_nnclass_random.sh
  n2_model_nncorn_random.sh
  n2_model_nnreg_random.sh
  n2_model_rf_random.sh
  n2_model_trac_random.sh
  n2_model_xgb_random.sh
)

for s in "${scripts[@]}"; do
  echo "Submitting $s"
  out=$(sbatch < "$s") || { echo "Failed: $s" >&2; continue; }
  echo "  -> $out"
  sleep 0.2
done
