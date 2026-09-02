#!/usr/bin/env bash
# Search-only ritme run for benchmarks: train/test split (reused if present)
# + find-best-model-config. Deliberately skips evaluate-tuned-models,
# bootstrap CIs and SHAP so that sacct's MaxRSS/TotalCPU reflect the model
# search alone (benchmarks B1/B2 in plan_benchmark_design.md).
#
# Driven by the same env vars as src/run_ritme_model.sh (CONFIG, PATH_MD,
# PATH_FT, PATH_DATA_SPLITS, LOGS_DIR, optional PATH_TAX/PATH_PHYLO/
# GROUP_BY_COLUMN/STRATIFY_BY_COLUMN/QZA_INPUTS); submit via
# `submit_model(..., template=<this file>)`.

set -euo pipefail

: "${CONFIG?Required env var CONFIG is unset}"
: "${PATH_MD?Required env var PATH_MD is unset}"
: "${PATH_FT?Required env var PATH_FT is unset}"
: "${PATH_DATA_SPLITS?Required env var PATH_DATA_SPLITS is unset}"
: "${LOGS_DIR?Required env var LOGS_DIR is unset}"

ulimit -u 60000
ulimit -n 524288

# Ray prestarts one idle worker per *detected node* CPU (up to 128+), which
# on small allocations dwarfs the search's own footprint and would dominate
# the MaxRSS measured in panel 3 of B2. Disable prestart; workers still
# start on demand for every trial.
export RAY_prestart_worker_first_driver=0

if [[ -n "${QZA_INPUTS:-}" ]]; then
  for triple in $QZA_INPUTS; do
    kind="${triple%%:*}"
    rest="${triple#*:}"
    src="${rest%%:*}"
    dst="${rest#*:}"
    if [[ -f "$dst" ]]; then
      echo "[skip] $dst already exists"
    elif [[ ! -f "$src" ]]; then
      echo "[skip] $src not present; nothing to convert for $kind"
    else
      python -m src.convert_qiime2_artifacts "$kind" "$src" -o "$dst"
    fi
  done
fi

if [[ -f "${PATH_DATA_SPLITS}/train_val.pkl" && -f "${PATH_DATA_SPLITS}/test.pkl" ]]; then
  echo "Reusing existing splits in ${PATH_DATA_SPLITS}"
else
  echo "Running split-train-test"
  mkdir -p "$PATH_DATA_SPLITS"
  group_args=()
  if [[ -n "${GROUP_BY_COLUMN:-}" ]]; then
    group_args=(--group-by-column "$GROUP_BY_COLUMN")
  fi
  stratify_args=()
  if [[ -n "${STRATIFY_BY_COLUMN:-}" ]]; then
    stratify_args=(--stratify-by "$STRATIFY_BY_COLUMN")
  fi
  ritme split-train-test "$PATH_DATA_SPLITS" "$PATH_MD" "$PATH_FT" \
    "${group_args[@]}" "${stratify_args[@]}" \
    --train-size 0.8 --seed 12
fi

tax_args=()
[[ -n "${PATH_TAX:-}" ]] && tax_args=(--path-to-tax "$PATH_TAX")
phylo_args=()
[[ -n "${PATH_PHYLO:-}" ]] && phylo_args=(--path-to-tree-phylo "$PATH_PHYLO")

echo "Running find-best-model-config (search only)"
ritme find-best-model-config "$CONFIG" "${PATH_DATA_SPLITS}/train_val.pkl" \
  "${tax_args[@]}" "${phylo_args[@]}" \
  --path-store-model-logs "$LOGS_DIR"
