# AGENTS.md

Guidelines for AI coding agents working on this repository.

---

## Project overview

This repos displays the power of **ritme** in three example usecases demonstrated in an accompanying mansucript. **ritme** is a Python package for finding the best microbial feature representation and model algorithm for a predictive task on next-generation sequencing data (regression or classification). Microbial features are engineered accounting for their statistical characteristics (compositionality, high dimensionality, hierarchical relationships, sparsity). **Ritme** lives in this open-source repos: https://github.com/adamovanja/ritme.

## The usecases
This repos displays 3 use cases: u1_amplicon_age_prediction (regression), u2_metagenome_ocean (regression) & u3_amplicon_crc_classification (binary classification). For each of these first the data is processed according to the original studies' pipeline in `n1_data.ipynb` (with its own conda env). Then ritme is used to perform a selected prediction task in `n2_run_ritme_model.ipynb` and additionally the original modelling approach is launched in `n4_original_setup.iypnb`. After the respective `n2_` notebooks ran, all trials are evaluated in `evaluate_all_trials.ipynb`.

A legacy regression use case (formerly `u3`: `u3_mlp_prediction`, microbial-load prediction from Nishijima 2024) is parked under the `u3_legacy` USECASES key while the new `u3` (CRC classification) is evaluated head-to-head; it will be removed once that comparison is complete.
Additionally, to the ritme comparisons also autoML comparisons are launched for all three usecases in `n5_generic_automl.ipynb` (regression for u1–u2, binary classification for u3 — task auto-dispatched from `USECASES[usecase]["task"]` in `src/launch_models.py`).

## Ritme's data flow

```
metadata + feature_table (+ phylogeny & taxonomy)
  → split_train_test()          # merge, optional temporal snapshotting, train/test split
  → find_best_model_config()    # search over feature engineering × model combinations
  → evaluate_tuned_models()     # evaluate best models on train + held-out test
```

## Usecases components that largely depend on ritme
Notebooks and associated scripts/modules of the following notebooks rely on ritme:
- `n2_run_ritme_model.ipynb`: for running ritme experiments
- `evaluate_all_trials.ipynb`: for comparing/evaluating all usecases and trials


## Important best practices
When performing and changes or additions to this repos make sure to follow best practices in software development. making sure all added code is clearly structured, only contains comments when really needed.
When testing new functionality always do it in an activated conda environment called, the one defined for this notebook. Never install packages in the base environment!

## Rules

### Before writing code
- Read the relevant source files before proposing changes.

### Pull requests
- One concern per PR. No unrelated changes.
- PR descriptions: what changed and why. Keep it concise.
- PR title: stick to the naming of former PRs with the prefix (FIX, ADD, ENH, MAINT, ...) in this project.

### Testing
- Since this repos is a demonstration - no unit tests need to be implemented.

### Formatting
- Ensure all code is formatted according to the pre-commit hooks in this repos.

### Commits
- Imperative form, matching existing `git log` style.
- AI attribution is mandatory: include `Co-Authored-By: <tool>` in commit trailers.

### Never commit site-specific infrastructure
This repository is public. Values that identify the institution's compute
environment must never enter a tracked file — not in code, notebooks, docs,
comments, or saved cell output:

- SLURM account / share names (e.g. anything matching `es_*`)
- node type constraints, partition names, hostnames
- absolute paths containing a real user, group or lab name — cluster
  (`/cluster/project/...`), Linux (`/home/...`) and macOS (`/Users/...`) alike
- internal hostnames, proxies or IP addresses

Read them at runtime instead, via `src/cluster_config.py`, which resolves
`.cluster.json` (gitignored) or `RITME_*` environment variables and omits the
corresponding sbatch flag when unset. Copy `.cluster.example.json` to
`.cluster.json` for local use.

Before every commit that touches launch code, notebooks or docs, verify:

```shell
git grep -n -I -E "\bes_[a-z]+\b|EPYC|/cluster/(project|home|scratch)/[a-z]|/Users/[a-z]|/home/[a-z]"
```

The `\b` around the account pattern is load-bearing: without it, `es_` matches
inside ordinary identifiers (`min_samples_split`, `classes_arr`, `matches`) and
buries the real hits in dozens of false positives.

Notebook **outputs** leak these too — a stack trace or warning pastes absolute
env paths into the committed JSON. Check output cells, not just source, and
match every platform: a notebook run on a laptop leaks `/Users/<name>/...`,
one run on the cluster leaks `/cluster/project/<lab>/<user>/...`.

A gitignore pattern does **not** protect an already-tracked file: `use_cases/
archive/` is ignored, yet the notebooks committed before that rule still had to
be scrubbed by hand.

### Writing style
- When writing documentation (e.g. in README files) - make sure the text is concise and does not contain unnecessary legacy/context information that is not crucial for the comment being made.
- Do not transplant conversational rationale into the document. If you justified a change in chat (e.g. "this works because tool X reads file Y..."), the file itself should still only state *what* the reader needs to do. If they want the rationale, upstream tool docs are the right place to send them.
