#!/usr/bin/env bash
# Run `ritme explain-stability` for one completed experiment: retrain the
# near-optimal band of trials and report how stable the deployed model's top
# features are across them.
#
# Driven entirely by env vars so the same script is used by every
# (usecase, model_type) combination — see `src/launch_models.py`.
#
# Required env vars:
#   PATH_EXP          Experiment directory holding experiment_config.json and
#                     mlflow_logs.csv (i.e. `${LOGS_DIR}/${experiment_tag}`).
#   MODEL_TYPE        Model type whose trials form the band (e.g. xgb), or
#                     "all".
#   PATH_DATA_SPLITS  Directory holding train_val.pkl/test.pkl.
#
# Optional env vars:
#   PATH_TAX          Taxonomy TSV (required when the band holds trials that
#                     aggregate by taxonomy, or trac).
#   PATH_PHYLO        Phylogeny NWK (required when the band holds trac).
#   STABILITY_TOP_N, STABILITY_MAX_TRIALS, STABILITY_BAND_SE_FACTOR,
#   STABILITY_MAX_CONCURRENT_TRIALS
#                     Forwarded to the matching `ritme explain-stability`
#                     option. Unset means ritme's default.
#   SHAP_MAX_BACKGROUND_SAMPLES
#                     Forwarded to `--max-background-samples`.

set -euo pipefail

: "${PATH_EXP?Required env var PATH_EXP is unset}"
: "${MODEL_TYPE?Required env var MODEL_TYPE is unset}"
: "${PATH_DATA_SPLITS?Required env var PATH_DATA_SPLITS is unset}"

# Same per-user soft caps as run_ritme_model.sh: the retrain runs through Ray.
ulimit -u 60000
ulimit -n 524288

opt_args=()
[[ -n "${PATH_TAX:-}" ]] && opt_args+=(--path-to-tax "$PATH_TAX")
[[ -n "${PATH_PHYLO:-}" ]] && opt_args+=(--path-to-tree-phylo "$PATH_PHYLO")
[[ -n "${STABILITY_TOP_N:-}" ]] && opt_args+=(--top-n "$STABILITY_TOP_N")
[[ -n "${STABILITY_MAX_TRIALS:-}" ]] && opt_args+=(--max-trials "$STABILITY_MAX_TRIALS")
[[ -n "${STABILITY_BAND_SE_FACTOR:-}" ]] && opt_args+=(--band-se-factor "$STABILITY_BAND_SE_FACTOR")
[[ -n "${STABILITY_MAX_CONCURRENT_TRIALS:-}" ]] && opt_args+=(--max-concurrent-trials "$STABILITY_MAX_CONCURRENT_TRIALS")
[[ -n "${SHAP_MAX_BACKGROUND_SAMPLES:-}" ]] && opt_args+=(--max-background-samples "$SHAP_MAX_BACKGROUND_SAMPLES")

echo "Running explain-stability (${MODEL_TYPE}) for ${PATH_EXP}"
ritme explain-stability "$PATH_EXP" "$MODEL_TYPE" \
  "${PATH_DATA_SPLITS}/train_val.pkl" \
  "${PATH_DATA_SPLITS}/test.pkl" \
  "${opt_args[@]}"
