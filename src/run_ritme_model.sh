#!/usr/bin/env bash
# Run a single ritme experiment end-to-end (optional QIIME2 conversion +
# train/test split + find-best-model-config + evaluate-tuned-models).
#
# Driven entirely by env vars so the same script is used by every
# (usecase, model_type) combination — see `src/launch_models.py`.
#
# Required env vars:
#   CONFIG            Path to the ritme experiment config JSON.
#   PATH_MD           Path to the metadata TSV.
#   PATH_FT           Path to the feature table TSV.
#   PATH_DATA_SPLITS  Directory holding (or to receive) train_val.pkl/test.pkl.
#   LOGS_DIR          Parent directory for the ritme experiment output.
#
# Optional env vars:
#   PATH_TAX          Taxonomy TSV (forwarded to find-best-model-config).
#   PATH_PHYLO        Phylogeny NWK (forwarded to find-best-model-config).
#   GROUP_BY_COLUMN   Column for grouped train/test split.
#   QZA_INPUTS        Space-separated triples `kind:src.qza:dst.tsv|nwk` to
#                     convert via `python -m src.convert_qiime2_artifacts`
#                     before splitting. Idempotent (skips existing dst).

set -euo pipefail

: "${CONFIG?Required env var CONFIG is unset}"
: "${PATH_MD?Required env var PATH_MD is unset}"
: "${PATH_FT?Required env var PATH_FT is unset}"
: "${PATH_DATA_SPLITS?Required env var PATH_DATA_SPLITS is unset}"
: "${LOGS_DIR?Required env var LOGS_DIR is unset}"

# 1. Convert QIIME2 artifacts to plain files where ritme expects them.
if [[ -n "${QZA_INPUTS:-}" ]]; then
  for triple in $QZA_INPUTS; do
    kind="${triple%%:*}"
    rest="${triple#*:}"
    src="${rest%%:*}"
    dst="${rest#*:}"
    if [[ -f "$dst" ]]; then
      echo "[skip] $dst already exists"
    else
      python -m src.convert_qiime2_artifacts "$kind" "$src" -o "$dst"
    fi
  done
fi

# 2. Train/test split (idempotent).
if [[ -f "${PATH_DATA_SPLITS}/train_val.pkl" && -f "${PATH_DATA_SPLITS}/test.pkl" ]]; then
  echo "Reusing existing splits in ${PATH_DATA_SPLITS}"
else
  echo "Running split-train-test"
  mkdir -p "$PATH_DATA_SPLITS"
  group_args=()
  if [[ -n "${GROUP_BY_COLUMN:-}" ]]; then
    group_args=(--group-by-column "$GROUP_BY_COLUMN")
  fi
  ritme split-train-test "$PATH_DATA_SPLITS" "$PATH_MD" "$PATH_FT" \
    "${group_args[@]}" --train-size 0.8 --seed 12
fi

# 3. Find best model config.
tax_args=()
[[ -n "${PATH_TAX:-}"   ]] && tax_args=(--path-to-tax "$PATH_TAX")
phylo_args=()
[[ -n "${PATH_PHYLO:-}" ]] && phylo_args=(--path-to-tree-phylo "$PATH_PHYLO")

echo "Running find-best-model-config"
ritme find-best-model-config "$CONFIG" "${PATH_DATA_SPLITS}/train_val.pkl" \
  "${tax_args[@]}" "${phylo_args[@]}" \
  --path-store-model-logs "$LOGS_DIR"

# 4. Evaluate tuned models on train + held-out test.
exp_tag=$(python -c "import json,sys; print(json.load(open('$CONFIG'))['experiment_tag'])")
echo "Running evaluate-tuned-models"
ritme evaluate-tuned-models "${LOGS_DIR}/${exp_tag}" \
  "${PATH_DATA_SPLITS}/train_val.pkl" \
  "${PATH_DATA_SPLITS}/test.pkl"
