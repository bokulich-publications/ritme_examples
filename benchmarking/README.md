# Benchmarking

Computational-efficiency benchmarks of *ritme*. The deliverables are the
three figures in `results/final/`, addressing the reviewer's request for a
rigorous compute benchmark relative to baselines:

- **B1_final** - best running validation RMSE over wall-clock time, TPE vs.
  random search (U1/xgb, 12 h budget, 3 seeds).
- **B2_final** - configurations explored and best validation RMSE over
  allocated CPU cores (4-128), ritme vs. auto-sklearn vs. TPOT on U1
  (enrichment-matched arms, 2 h budget, 3 seeds).
- **B3_final** - ritme's CPU utilisation over allocated cores, from the same
  sweep (backup figure `b3_final_comparators` adds the autoML arms).

Superseded arms and their outputs (the B2 resource-efficiency figure, the B3
matched-budget comparison incl. its probes, the underutilization study) are
parked in `benchmarking/archive/` (gitignored) -- see `archive/README.md`.
Sections below describing those arms are kept as design documentation.

## Environments

- Launch, collection and plotting run in the `ritme_usecases` conda env
  (setup: see `use_cases/*/n2_run_ritme_model.ipynb`).
- The B2 auto-sklearn jobs run in the `autosklearn` env (setup: see
  `use_cases/n5_generic_automl.ipynb`); the launcher invokes that env's
  interpreter directly, so submit everything from `ritme_usecases`. Two
  reasons, and only the first is general: `mamba run --no-capture-output` is
  broken in mamba 2.0.4 here (`exec: --: invalid option`), and without that
  flag the wrapper relays the child's output through itself, which hung this
  arm when that output was consumed through a pipe. Plain `mamba run` is
  *not* unusable on this cluster -- the comparator arms in
  `src/launch_automl.py` use it successfully under `sbatch --output=`, where
  stdout goes to a file and there is no pipe to fill. Calling the
  interpreter directly simply avoids both the broken flag and the question.

No dependencies beyond those two environments are required. The B2 launcher
converts the U1 splits to parquet once so the auto-sklearn env (NumPy 1.x)
can read splits pickled by `ritme_usecases` (NumPy 2.x); both envs already
ship `pyarrow`.

## Running

All commands from the repo root. Every model run is a SLURM job pinned to a
single node type; only sacct queries, CSV collection and plotting run outside
SLURM.

The SLURM account and node type identify a specific site, so they are not
stored in the repo. Copy `.cluster.example.json` to `.cluster.json` (which is
gitignored) and fill in your own, or set `RITME_SLURM_ACCOUNT` /
`RITME_NODE_CONSTRAINT`; leaving either unset omits the corresponding sbatch
flag. Manifests record run directories relative to the repo root for the same
reason. See `src/cluster_config.py`.

```shell
# B1_final - 2 samplers x 3 seeds on U1/xgb, 12h budget each (+ warm-up job)
python -m benchmarking.launch_b1            # add --smoke to validate first
python -m benchmarking.collect_b1
python -m benchmarking.plot_b1_final

# ritme sweep - {4..128 cores} x 3 seeds, 2h budget each. Supplies B2_final's
# ritme arm and the utilisation data behind B3_final; its pre-enrichment
# auto-sklearn arm is archived (restore archive/runs/b2_automl to
# runs/b2/automl to re-collect it)
python -m benchmarking.launch_b2            # add --smoke to validate first
python -m benchmarking.collect_b2

# ritme_efficient: parked study, see benchmarking/excursion_ritme_efficient/
# (self-contained; not part of the results of record)

# comparator sweep - auto-sklearn + TPOT on the same grid. Tagged b4_* in
# manifests, job names and runs/b4/: a legacy prefix kept for compatibility
# with already-submitted jobs
python -m benchmarking.launch_comparators --usecases u1 --methods automl tpot
python -m benchmarking.collect_comparators
python -m benchmarking.plot_b2_final

# B3_final - utilisation decomposition of the ritme sweep (no jobs)
python -m benchmarking.analyze_utilization
python -m benchmarking.plot_b3_final
```

Launchers must run from the `ritme_usecases` interpreter: ritme jobs inherit
the launcher's environment (`sbatch --export=ALL`) and run `ritme` from PATH,
so `ensure_launcher_env()` puts that interpreter's `bin` first and refuses to
submit from one that cannot import ritme's dependencies (a `ritme` binary
beside the interpreter is not proof -- a stale install elsewhere passes that
test and dies at import inside the job).

To relaunch a failed ritme run, remove its run directory *and* move its log
aside: `submit_model` opens logs with `--open-mode=append`, so a retry would
otherwise write beneath the old traceback. The auto-sklearn and TPOT arms
truncate their logs, one attempt per file.

Launchers write a manifest (`manifests/`) with the SLURM job ids and all
parameters of each submission batch; collectors join the run outputs with
sacct via these manifests. Re-running a launcher skips runs whose outputs
already exist. Run outputs land in `runs/` and derived tables/figures in
`results/` (both gitignored).

## Design notes

- B1/B2 ritme jobs use `run_ritme_search_only.sh` (split +
  `find-best-model-config` only), so sacct's MaxRSS/TotalCPU measure the
  search itself, not SHAP/bootstrap post-processing. Ray's idle-worker
  prestart is disabled in this template for the same reason.
- The B2 auto-sklearn arm runs through `automl_b2.py` rather than
  `src.launch_automl.submit_automl`, which cannot express this benchmark:
  it scores on a holdout split instead of grouped K-fold, refits and
  evaluates on the test set, and reports no count of evaluated
  configurations. `automl_b2.py` searches only, scores with the same
  grouped protocol as ritme, and writes the configuration count panel 1
  needs. The unchanged `submit_automl` path still backs B3.
- B2 lets each method convert allocated cores into work the way it is built
  to, then reports how fully it managed to: ritme runs cores/4 concurrent
  trials with xgb's `nthread` set to the 4 CPUs Ray gives each, while
  auto-sklearn runs `n_jobs = cores` single-threaded workers, each capped at
  one core's memory. Both mappings were measured before being adopted, at 32
  cores with the same seed and budget:

  | auto-sklearn mapping | configurations | CPU used | MEMOUT |
  |---|---|---|---|
  | cores/4 workers x 4 threads | 121 | 25% of allocation | 6.8% |
  | cores workers x 1 thread | 336 | 95% of allocation | 3.5% |

  Matching ritme's 4-CPU granularity looks even-handed but leaves three
  quarters of auto-sklearn's allocation idle, because `gradient_boosting`
  does not thread across 4 cores (one 5-fold fit takes 99 s at 4 threads
  versus 211 s at 1). Single-threaded workers nearly saturate the allocation
  and explore ~2.8x more configurations, so that is the configuration
  reported. The threads must be pinned either way: left at their default
  (the node's 128 cores) the libraries oversubscribe every worker and abort
  the 128-core runs in auto-sklearn's dummy prediction.
- Both arms share the U1 train/test split, and the auto-sklearn arm scores
  with GroupKFold(5) on `host_id`, matching ritme's validation protocol, so
  panel 2 compares absolute validation RMSE. Both search feature
  representation x model: ritme over its feature engineering, auto-sklearn
  over its own preprocessors.
- auto-sklearn enforces a memory cap per configuration (3584 MB, one core's
  share) and records configurations killed by it as MEMOUT in `*_runs.csv`,
  whereas ritme trials are bounded only by the job allocation; report the cap
  in the caption beside peak RSS.
- B2 uses 3584 MB/CPU (not the 4096 of B1/B3) because 128 cores x 4096 MB
  exceeds the memory of every 128-core node type.
- At the bottom of the sweep ritme is memory-bound rather than
  compute-bound: its fixed footprint (dataset plus Ray overhead) does not
  shrink with the allocation, so at 4 cores (14 GB) peak RSS reaches the
  allocation and one of the three seeds was killed by it after exhausting
  its budget. That seed's outputs are complete and kept; `state` in
  `b2_summary.csv` records which runs ended OUT_OF_MEMORY.
- Panels 1 and 2 point in opposite directions and should be read together:
  auto-sklearn evaluates more configurations per allocation than ritme (its
  single-threaded `gradient_boosting` fits are cheaper) yet stays well above
  ritme's validation RMSE at every point. The comparison is therefore about
  what each method searches over, not how fast it searches -- ritme's space
  is microbiome-aware feature engineering plus metadata enrichment, while
  auto-sklearn receives the relative-abundance table and its own generic
  preprocessors, exactly as in the manuscript's existing comparison.
- `--ritme-label <name>` (launch_b2 / collect_b2) and `--compare <name>`
  (plot_b2) rerun and overlay the ritme arm under another ritme build,
  everything else fixed. The launcher refuses to submit unless it runs from
  the conda env of the same name, so the label cannot disagree with the build
  the jobs inherit. Their one use so far, the `improve_efficiency` branch, is
  parked in `benchmarking/excursion_ritme_efficient/` (gitignored) with its
  own code, archive and README.
- Two definitions the ritme arms share, fixed so that a build which prunes
  (the `improve_efficiency` branch runs ASHA with `grace_period=2`) stays
  comparable with one that never did: *configurations explored* counts every
  finished trial with a validation estimate -- an ASHA-pruned trial reports
  the running K-fold mean after its last fold and did steer the search --
  while *best validation score* is taken over full-fold trials only, so it
  remains a 5-fold mean in both builds. `n_configs_full` and
  `n_configs_pruned` are reported alongside. For the baseline, which never
  pruned, the two definitions coincide (verified: B2's summary is unchanged).
- Baseline validity for that comparison: B2's ritme runs used ritme 1.4.2,
  and `v1.4.2 -> main` changes none of `tune_models.py`, `model_space/` or
  `feature_space/`, so they are equivalent to the branch's merge base for
  everything the branch touches. The shared `ritme_usecases` env has since
  been moved to a commit that is not on main (`1.4.3+1.ga9bf5f8`), which the
  B4 smoke ran on; pin and record the version before the full B4 sweep.
- Also worth stating in the B2 caption: at 4 cores the budget buys around as
  many ritme trials as TPE's 75-trial warm-up, so that point still largely
  reflects random sampling.
- B1's budget is sized so TPE's adaptive random warm-up (75 trials for
  U1/xgb, computed by `compute_warmup.py`) is a small fraction of the
  search; at this allocation the production run sustained ~41 trials/h.
- B3 reads accuracy from `use_cases/all_experiments_metrics_*.csv` (the
  newest one; its name is recorded in the output as `accuracy_source`) and
  resources from sacct for the archived job ids pinned and name-verified in
  `collect_b3.py`. One run per cell: point values only.
- Unlike B1/B2, the archived B3 jobs ran post-processing inside the same sacct
  record as the search -- for ritme the full `src/run_ritme_model.sh`
  (evaluation, bootstrap CIs, SHAP), for auto-sklearn prediction and plotting.
  Measured against the trial logs, ritme's tail is 0.15-0.40% of each job
  (2-6 minutes of 23 hours), so the reported CPU-hours and elapsed times are
  effectively search-only. Peak RSS is not safe that way: it is a maximum, so
  a single short SHAP spike can set it.
- `probe_b3.py` therefore reruns the search alone -- both methods, all three
  use cases, same allocation, concurrency and configuration as the archived
  runs -- on a 1 h budget, and `collect_b3.py` reports its peak RSS in
  `search_only_max_rss_gb`. Concurrency and data size set most of that peak,
  which is what makes a short probe informative; but a 23 h search samples
  many more configurations and so has more chances to hit a memory-hungry
  one, so **read the probe values as a lower bound on the 23 h search-only
  peak**, not as an exact substitute. CPU-hours are not transferable from a
  probe at all and stay sourced from the archived jobs. The probes keep the
  archived auto-sklearn settings exactly (`n_jobs=-1`,
  `memory_limit=24000`, holdout resampling, library thread defaults), so
  their memory is comparable to the archived run rather than to B2's
  differently-configured arm.
- Removing post-processing changes what the memory panel says: archived,
  ritme's peak RSS was below auto-sklearn's in u1 (65 vs 75 GB); search-only
  it is above auto-sklearn's in all three use cases (57 vs 40, 29 vs 22,
  27 vs 17 GB). Both arms' archived figures were inflated -- auto-sklearn's
  by its post-fit prediction and plotting over the full feature table, ritme's
  by SHAP -- so the comparison had to be redone for both, not just for ritme.
- Probes that die before their budget elapses never reach a representative
  peak, so `collect_b3.py` ignores any whose elapsed time is under 90% of the
  budget and records each probe's state and elapsed time beside its value.

## B4 - compute scaling on all use cases

B2's figure for U1 only; B4 is the same design (2 h budget, 4-128 cores, 3
seeds, 3584 MB/CPU, one node type) on U1, U2 and U3 as figure columns, with
TPOT (all use cases) and mAML (U3) added as swept arms. ritme is always drawn
in orange and listed first; the comparators take the blues. Design notes:

- **ritme's model per use case is chosen on validation, not test** -- ritme's
  own objective (`rmse_val` min / `roc_auc_val` max). That gives xgb (U1),
  linreg (U2) and xgb_class (U3). U3 is the one case where the test metric
  would pick differently (logreg: val 0.836 / test 0.834 against xgb_class
  val 0.867 / test 0.769); selecting on test is the leak the comparison must
  avoid.
- Comparators are restricted to the closest family in their own catalogue:
  auto-sklearn `gradient_boosting` / `sgd` / `gradient_boosting` (it has no
  ElasticNet, so `sgd` stands in for linreg as in the manuscript), TPOT
  `XGBRegressor` / `ElasticNetCV` / `XGBClassifier`
  (`src/comparator_tpot.py:ESTIMATOR_FOR_USECASE`).
- All arms receive the same metadata enrichment
  (`src/launch_automl._read_enrich_with`), including ritme, whose configs are
  aligned to that list at submit time. For U3 this drops `fit_result`, a
  screening readout of the outcome; the `u3_*_tpe_no_fit` runs on the
  `more_methods` branch exist for the same reason.
- The comparator workers (`src/comparator_*.py`, `src/launch_automl.py`) are
  byte-identical copies from `more_methods`, pinned in
  `benchmarking/.comparator_source_commit` and recorded in each manifest.
- (U1, ritme) is reused from B2 unchanged. B2's auto-sklearn U1 arm is *not*
  reused: it predates the enrichment, so U1 auto-sklearn is re-run.
- mAML is an exhaustive grid with no notion of a budget: 781 configurations
  took 11 h 41 min at 50 cores in the manuscript run, so it cannot finish
  inside 2 h at any swept allocation. `archive/code/maml_sweep.py` (archived) walks the same grid in
  its published order and stops starting new (scaler, classifier) grids once
  the budget is spent; each finished grid is appended to the configuration
  file with its completion time, and the collector counts only configurations
  that completed inside the budget, so a job killed at walltime mid-grid is
  still exact. What this measures is a *prefix* of a fixed enumeration --
  which configurations get evaluated depends on grid order, not on their
  promise -- and the caption should say so. It scales with cores through
  `GridSearchCV(n_jobs)`, but small grids cannot occupy many cores, so expect
  its curve to flatten early.
- TPOT's per-evaluation cap is a `stopit` timeout that corrupts XGBoost's
  heap when it fires (3 of 3 manuscript U1 runs died). It cannot be disabled,
  so it is set 10 min *above* the search budget, where it can never fire
  before the search ends; the walltime buffer absorbs one unbounded pipeline.
  Runs that die anyway are counted from the job log (an upper bound: tqdm
  counts GP re-proposals that `evaluated_individuals_` deduplicates, +2% to
  +26% on the manuscript runs) into `n_configs_upper_bound`, never into
  `n_configs`, and drawn as open markers outside the median line.
- TPOT and auto-sklearn workers are pinned to one BLAS/OpenMP thread each;
  the manuscript TPOT runs at 50 cores were not, which the manifest records.
- Panel 2's metric differs by column (RMSE, lower is better, for U1/U2;
  ROC-AUC, higher is better, for U3) and is computed under the same
  resampling per arm: GroupKFold(5) on `host_id` for U1, shuffled KFold(5) /
  StratifiedKFold(5) seeded by the run for U2 / U3.

## Utilisation diagnostic

`analyze_utilization.py` explains the CPU-hours the other benchmarks report.
It runs no jobs; it re-reads the trial logs and the collected sacct tables and
splits each search's CPU efficiency into the only two factors that can lower
it, which are independent and multiply exactly:

    utilisation = (trials in flight / max_cuncurrent_trials)      # slot fill
                x (cores a trial uses / cpus_per_trial)           # core fill

It covers the manuscript's whole final run set, discovered automatically: the
ritme experiments in the newest `use_cases/all_experiments_metrics_*.csv`
(the same file `collect_b3.py` reads accuracy from), minus the `*_original`
and `*_automl` baselines. Each experiment's producing job is the last SLURM
job that completed under its experiment tag -- a rule validated against the
five job ids pinned in `collect_b3.py`, which it reproduces exactly. Earlier
attempts are split into failed (waste) and superseded (deliberate re-runs).

Headline over the 22 experiments: SLURM charged 34,076 core-hours and 12,820
(38%) were spent executing a trial. Utilisation by model family:

| family | n | utilisation | slot fill | core fill | median trial |
|---|---|---|---|---|---|
| logreg    | 1 | 14.3% | 21.8% | 65.4% |   16 s |
| linreg    | 3 | 21.4% | 63.4% | 38.6% |   33 s |
| trac      | 2 | 29.3% | 98.0% | 31.1% |  751 s |
| rf_class  | 1 | 33.7% | 44.0% | 76.6% |   26 s |
| rf        | 2 | 49.5% | 81.7% | 63.7% |   49 s |
| nn_class  | 1 | 54.2% | 97.6% | 55.6% | 2604 s |
| nn_corn   | 2 | 58.5% | 98.5% | 59.4% | 4703 s |
| nn_reg    | 2 | 60.1% | 98.3% | 61.0% | 3958 s |
| xgb_class | 2 | 65.7% | 67.3% | 97.5% |   28 s |
| xgb       | 4 | 92.0% | 98.2% | 94.0% |  226 s |

`plot_utilization.py` renders the supplementary figure
`archive/results/utilisation/utilisation_decomposition.pdf`, and also writes four
figures to `figures/issue_underutilization/` (tracked, unlike `results/`)
that back the findings intended for issues against ritme: the launch-rate
ceiling, model-blind CPU reservation, and what the campaign was charged for.

- Trials still RUNNING when the budget expires are closed at the last event
  in the log rather than dropped. Dropping them undercounts occupancy badly
  where slots are many and trials long -- at 128 cores they hold 29% of all
  trial-time -- while for the 23 h runs the imputation is under 1.3%. The
  share it contributes is reported as `imputed_trial_frac`.
- Core fill is a residual (`utilisation / slot fill`), so it also absorbs
  driver, raylet and MLflow CPU and can slightly exceed 100% when trials are
  short. Read it as an upper bound on true per-trial saturation.
- Both factors are measured against ritme's *effective* concurrency, not the
  configured value: `tune_models.py:800-805` divides trac's concurrency by
  three for memory, and since `_get_resources` then divides the allocation by
  the reduced number, each trac trial reserves three times more CPU. The
  analysis replicates that rule and cross-checks every derived
  `cpus_per_trial` against the `Using these resources: CPU n` line in the
  run's own log; no run disagrees.
- Core fill is a property of the model, and ritme documents it at
  `tune_models.py:573` ("linreg: not parallelizable"). `_build_linreg` accepts
  `n_jobs` and ignores it, so u2's trials hold 5 reserved CPUs and use 1.5.
  Nothing under `ritme/feature_space/` is parallelised, and
  `process_train_kfold` engineers 2K+1 = 11 design matrices serially before
  any fan-out; for an ElasticNet fit that serial head is the whole trial.
- Slot fill is capped by a fixed cost between trials, not by the model. The
  gap between successive trial starts has a hard floor of ~7.5 s (1st
  percentile 7.1-7.7 s at every allocation from 8 to 128 cores; 0 of 15,818
  archived trials started within 4 s of another) even when most slots are
  empty. Measured cleanest at 1 slot, where nothing hides it: median
  end-of-trial to start-of-next is 7.52-7.59 s over 3 seeds.
- The root cause is in the launch path, not the model. Ray sets
  `max_pending_trials = 1` for any searcher that is not a
  `BasicVariantGenerator`, and ritme passes `OptunaSearch`, so only one trial
  may be created and one actor bootstrapping at a time -- all 18 B2 logs print
  only ever `| 1 PENDING`. `reuse_actors` is left at Ray's `False` default, so
  each trial also spawns a fresh process that re-imports
  `ritme.model_space.static_trainables` (~19 s, ~6 s of it `torchmetrics`
  pulling in `torchvision`'s CV model zoo) even for sklearn-only searches.
- Consequence: throughput saturates at ~375-415 trials/h on *any* allocation.
  Keeping every slot full needs one launch every
  `mean_trial_duration / max_cuncurrent_trials` seconds, so slot fill collapses
  exactly where that falls below the floor -- u1/xgb needs 85.4 s and gets
  97.9%, u3/xgb_class needs 4.4 s and gets 44.3%, u3/logreg needs 2.1 s and
  gets 21.8%.
- The same law explains B2's peak at 32 cores: median trial duration is flat
  at 39-55 s across the sweep, so trials are not slowed by contention, but the
  required launch interval falls to 7.7 s at 32 slots -- the floor -- and slot
  fill drops from 93.2% at 8 slots to 66.7% at 32.
- The floor is flat in time, so it is not TPE's sampler getting slower: gaps
  measured per quarter of the 23 h runs stay at 8.2-9.9 s throughout. It is
  also not sensitive to Ray's idle-worker prestart, which the archived runs
  had enabled and B1/B2 disabled, yet both show the same floor.
- Falling trials/hour is not by itself underutilisation. In B1 TPE completes
  far fewer trials than random (394-1568 vs 1257-1578 over 12 h) yet reaches
  *higher* CPU efficiency (90.8-95.2% vs 86.0-88.3%), because it converges on
  feature-richer configurations: median `nb_features` climbs 73 -> 4207 between
  the first and last two hours for TPE seed 0 while random stays at ~37.
