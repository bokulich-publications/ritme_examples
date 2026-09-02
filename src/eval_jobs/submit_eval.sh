#!/usr/bin/env bash
# Submit the headless evaluate_all_trials.ipynb runs as SLURM batch jobs.
#
# Usage: ./submit_eval.sh [u3|u4|all]     (default: all)
#
# Site-specific values are resolved here at submit time and never live in
# the tracked sbatch files: the SLURM account comes from the untracked
# .cluster.json (same source as the benchmarking launchers), conda's
# profile script from `conda info --base`. The node constraint from
# .cluster.json is intentionally NOT applied: evaluation is not
# timing-sensitive, and skipping it schedules faster.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

ACCOUNT=$(sed -n 's/.*"slurm_account"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
  "$REPO_ROOT/.cluster.json")
if [ -z "$ACCOUNT" ]; then
  echo "error: no slurm_account found in $REPO_ROOT/.cluster.json" >&2
  exit 1
fi

CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"

# Executed notebooks and SLURM logs go to personal scratch (gitignored via
# the x_* rule); figures land in use_cases/result_figures/<usecase>/.
OUT_DIR="$REPO_ROOT/x_scratch/eval_trials"
mkdir -p "$OUT_DIR/logs"

submit() {
  sbatch --account="$ACCOUNT" \
    --output="$OUT_DIR/logs/%x_%j.out" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_SH="$CONDA_SH",OUT_DIR="$OUT_DIR" \
    "$@"
}

target=${1:-all}
case "$target" in u3 | u4 | all) ;; *)
  echo "usage: $0 [u3|u4|all]" >&2
  exit 1
  ;;
esac

dep=()
if [ "$target" = u3 ] || [ "$target" = all ]; then
  u3_line=$(submit "$SCRIPT_DIR/eval_u3.sbatch")
  echo "u3: $u3_line"
  # Both notebooks write use_cases/all_experiments_metrics_<yymmdd>.csv when
  # today's cache is missing. Chain u4 behind u3 in that case so the two
  # jobs don't race on the file; with the cache present they run in
  # parallel (both just read it).
  cache="$REPO_ROOT/use_cases/all_experiments_metrics_$(date +%y%m%d).csv"
  if [ "$target" = all ] && [ ! -f "$cache" ]; then
    u3_id=${u3_line##* }
    dep=(--dependency="afterany:$u3_id")
    echo "   (no metrics cache for today; u4 will wait for u3)"
  fi
fi
if [ "$target" = u4 ] || [ "$target" = all ]; then
  u4_line=$(submit "${dep[@]}" "$SCRIPT_DIR/eval_u4.sbatch")
  echo "u4: $u4_line"
fi
