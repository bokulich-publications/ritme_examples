#!/usr/bin/env bash
# Submit the prediction extractions as SLURM jobs, one per (use case, arm).
#
# Usage: ./submit_predictions.sh [u1|u2|u3|u4|all] [original|ritme|automl|all]
#
# The SLURM account is read from the untracked .cluster.json and conda's
# profile script from `conda info --base`.
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
OUT_DIR="$REPO_ROOT/x_scratch/supp_scatter_roc"
mkdir -p "$OUT_DIR/logs"

# usecase:arm -> cpus mem-per-cpu time env. u4 dominates: its ritme arm peaks
# at ~443G (whole-split prediction over ~296k engineered features) and its
# comparator arms carry the full 21828 x 315098 test matrix in memory.
resources() {
  case "$1" in
    u4:ritme) echo "16 28G 04:00:00 ritme_usecases" ;;
    u4:original) echo "16 16G 12:00:00 ritme_usecases" ;;
    u4:automl) echo "20 25G 12:00:00 tpot_bench" ;;
    *:original) echo "16 4G 04:00:00 ritme_usecases" ;;
    *:ritme) echo "4 8G 02:00:00 ritme_usecases" ;;
    *:automl) echo "8 8G 04:00:00 tpot_bench" ;;
  esac
}

usecases=${1:-all}
arms=${2:-all}
[ "$usecases" = all ] && usecases="u1 u2 u3 u4"
[ "$arms" = all ] && arms="original ritme automl"

for usecase in $usecases; do
  for arm in $arms; do
    read -r cpus mem time env <<<"$(resources "$usecase:$arm")"
    line=$(sbatch --account="$ACCOUNT" \
      --job-name="supp_${usecase}_${arm}" \
      --cpus-per-task="$cpus" \
      --mem-per-cpu="$mem" \
      --time="$time" \
      --output="$OUT_DIR/logs/%x_%j.out" \
      --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_SH="$CONDA_SH",USECASE="$usecase",ARM="$arm",ENV_NAME="$env" \
      "$SCRIPT_DIR/jobs/extract.sbatch")
    echo "$usecase $arm ($env, $cpus x $mem, $time): $line"
  done
done
