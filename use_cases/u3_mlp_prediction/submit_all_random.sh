#!/usr/bin/env bash
# Minimal launcher: submit each model script as a separate SLURM job.
# Order is fixed and no dependencies are enforced.
# Usage: ./submit_all.sh

set -euo pipefail

scripts=(
  n2_model_metacardis_log_linreg_random.sh
  n2_model_metacardis_log_nnreg_random.sh
  n2_model_metacardis_log_rf_random.sh
  n2_model_metacardis_log_xgb_random.sh
)

for s in "${scripts[@]}"; do
  echo "Submitting $s"
  out=$(sbatch < "$s") || { echo "Failed: $s" >&2; continue; }
  echo "  -> $out"
  sleep 0.2
done
