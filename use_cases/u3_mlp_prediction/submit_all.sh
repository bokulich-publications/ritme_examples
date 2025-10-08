#!/usr/bin/env bash
# Minimal launcher: submit each model script as a separate SLURM job.
# Order is fixed and no dependencies are enforced.
# Usage: ./submit_all.sh

set -euo pipefail

scripts=(
  n2_model_metacardis_log_linreg_tpe.sh
  n2_model_metacardis_log_nnreg_tpe.sh
  n2_model_metacardis_log_rf_tpe.sh
  n2_model_metacardis_log_xgb_tpe.sh
  n2_model_metacardis_log_xgb_tpe_w_start.sh
)

for s in "${scripts[@]}"; do
  echo "Submitting $s"
  out=$(sbatch < "$s") || { echo "Failed: $s" >&2; continue; }
  echo "  -> $out"
  sleep 0.2
done
