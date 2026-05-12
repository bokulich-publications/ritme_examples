#!/usr/bin/env bash
# Run a single ritme experiment end-to-end (optional QIIME2 conversion +
# train/test split + find-best-model-config + evaluate-tuned-models +
# bootstrap test-set CIs + explain-features for the best tuned model).
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
#   STRATIFY_BY_COLUMN  Column passed to `ritme split-train-test --stratify-by`
#                     (comma-separated for multi-column stratification, e.g. a
#                     binary classification target).
#   QZA_INPUTS        Space-separated triples `kind:src.qza:dst.tsv|nwk` to
#                     convert via `python -m src.convert_qiime2_artifacts`
#                     before splitting. Idempotent (skips existing dst).
#   SHAP_MAX_BACKGROUND_SAMPLES
#                     Forwarded to `ritme explain-features
#                     --max-background-samples`. Unset means use the full
#                     training set (ritme's default). Useful for local
#                     smoke tests where the full background can OOM.

set -euo pipefail

: "${CONFIG?Required env var CONFIG is unset}"
: "${PATH_MD?Required env var PATH_MD is unset}"
: "${PATH_FT?Required env var PATH_FT is unset}"
: "${PATH_DATA_SPLITS?Required env var PATH_DATA_SPLITS is unset}"
: "${LOGS_DIR?Required env var LOGS_DIR is unset}"

# Raise per-user soft caps so Ray's prestart workers (one per detected CPU
# on the compute node, typically ~128) don't exhaust nproc and leave no
# budget for joblib to fork sklearn workers inside trainables.
ulimit -u 60000
ulimit -n 524288

# 1. Convert QIIME2 artifacts to plain files where ritme expects them.
if [[ -n "${QZA_INPUTS:-}" ]]; then
  for triple in $QZA_INPUTS; do
    kind="${triple%%:*}"
    rest="${triple#*:}"
    src="${rest%%:*}"
    dst="${rest#*:}"
    if [[ -f "$dst" ]]; then
      echo "[skip] $dst already exists"
    elif [[ ! -f "$src" ]]; then
      # Missing source qza is fine when the destination is unused by this
      # run (e.g. when train_val.pkl/test.pkl are pre-staged so the split
      # step below is skipped). A downstream step that actually needs the
      # missing file will error with a clear message there.
      echo "[skip] $src not present; nothing to convert for $kind"
    else
      python -m src.convert_qiime2_artifacts "$kind" "$src" -o "$dst"
    fi
  done
fi

# 2. Train/test split (idempotent on path; does NOT detect config drift —
# delete the splits dir if you change --group-by-column, --stratify-by,
# --train-size, or --seed between runs).
if [[ -f "${PATH_DATA_SPLITS}/train_val.pkl" && -f "${PATH_DATA_SPLITS}/test.pkl" ]]; then
  echo "Reusing existing splits in ${PATH_DATA_SPLITS}"
  echo "  (current launch flags — group=${GROUP_BY_COLUMN:-none}, stratify=${STRATIFY_BY_COLUMN:-none}, time=${TIME_COL:-none}, host=${HOST_COL:-none}, n_prev=${N_PREV:-none}; verify these match the cached split)"
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
  snapshot_args=()
  if [[ -n "${TIME_COL:-}" && -n "${HOST_COL:-}" && -n "${N_PREV:-}" ]]; then
    # `--missing-mode exclude` is passed explicitly to work around a ritme
    # CLI default of None that fails its own validation when the snapshot
    # pipeline is enabled.
    snapshot_args=(--time-col "$TIME_COL" --host-col "$HOST_COL" --n-prev "$N_PREV" --missing-mode exclude)
  fi
  ritme split-train-test "$PATH_DATA_SPLITS" "$PATH_MD" "$PATH_FT" \
    "${group_args[@]}" "${stratify_args[@]}" "${snapshot_args[@]}" \
    --train-size 0.8 --seed 12
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

# `ls_model_types` is single-element by construction (`launch_models.py`
# pins one model class per run); we read the first entry verbatim and
# reuse it for both bootstrap and SHAP.
model_type=$(python -c "import json,sys; print(json.load(open('$CONFIG'))['ls_model_types'][0])")

# 5. Bootstrap 95% CIs on test-set metrics for the tuned model. The CLI
# auto-dispatches by model_type: regression models yield RMSE/R²/Pearson,
# classification models yield AUROC/accuracy/F1/precision/recall. Cheap
# (predict once + 1000 metric resamples) so it fits comfortably in the
# SHAP buffer carved out of the SLURM walltime.
echo "Running bootstrap-test-metrics (${model_type})"
python -m src.bootstrap_metrics \
  "${LOGS_DIR}/${exp_tag}" "$model_type" "${PATH_DATA_SPLITS}"

# 6. Compute SHAP feature importance for the best tuned model.
shap_args=()
[[ -n "${SHAP_MAX_BACKGROUND_SAMPLES:-}" ]] && shap_args=(--max-background-samples "$SHAP_MAX_BACKGROUND_SAMPLES")
echo "Running explain-features (${model_type})"
ritme explain-features "${LOGS_DIR}/${exp_tag}" "$model_type" \
  "${PATH_DATA_SPLITS}/train_val.pkl" \
  "${PATH_DATA_SPLITS}/test.pkl" \
  "${shap_args[@]}"
