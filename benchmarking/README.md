# Benchmarking

Computational-efficiency benchmarks of *ritme*. The deliverables are the two
figures in `results/final/`:

- **search_efficiency** - best running validation RMSE over wall-clock time,
  TPE vs. random search (U1/xgb, 12 h budget, 3 seeds).
- **compute_efficiency** - three stacked panels over allocated CPU cores
  (4-128), ritme vs. auto-sklearn vs. TPOT on U1 (enrichment-matched arms,
  2 h budget, 3 seeds): best validation RMSE, configurations explored, and
  CPU utilisation.

Both are written as `.eps`, `.pdf` and `.png` at 600 dpi with embedded
TrueType fonts, matching the manuscript figures in `final_figures/`.

Superseded arms and their outputs are parked in `benchmarking/archive/`
(gitignored) -- see `archive/README.md`.

## Environments

- Launch, collection and plotting run in the `ritme_usecases` conda env
  (setup: see `use_cases/*/n2_run_ritme_model.ipynb`).
- The auto-sklearn jobs run in the `autosklearn` env (setup: see
  `use_cases/n5_generic_automl.ipynb`). The launcher invokes that env's
  interpreter directly, so submit everything from `ritme_usecases`.

No dependencies beyond those two environments are required. `launch_b2`
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
# search_efficiency - 2 samplers x 3 seeds on U1/xgb, 12h budget each
# (+ warm-up job)
python -m benchmarking.launch_b1            # add --smoke to validate first
python -m benchmarking.collect_b1
python -m benchmarking.plot_search_efficiency

# ritme sweep - {4..128 cores} x 3 seeds, 2h budget each. Supplies
# compute_efficiency's ritme arm
python -m benchmarking.launch_b2            # add --smoke to validate first
python -m benchmarking.collect_b2

# comparator sweep - auto-sklearn + TPOT on the same grid, tagged b4_* in
# manifests, job names and runs/
python -m benchmarking.launch_comparators --usecases u1 --methods automl tpot
python -m benchmarking.collect_comparators
python -m benchmarking.plot_compute_efficiency

# optional: split the ritme sweep's CPU efficiency into slot fill x core fill
# (no jobs; the figure's utilisation panel does not need it)
python -m benchmarking.analyze_utilization
```

Launchers must run from the `ritme_usecases` interpreter: ritme jobs inherit
the launcher's environment (`sbatch --export=ALL`) and run `ritme` from PATH,
so `ensure_launcher_env()` puts that interpreter's `bin` first and refuses to
submit from one that cannot import ritme's dependencies.

To relaunch a failed ritme run, remove its run directory *and* move its log
aside: `submit_model` opens logs with `--open-mode=append`, so a retry would
otherwise write beneath the old traceback. The auto-sklearn and TPOT arms
truncate their logs, one attempt per file.

Launchers write a manifest (`manifests/`) with the SLURM job ids and all
parameters of each submission batch; collectors join the run outputs with
sacct via these manifests. Re-running a launcher skips runs whose outputs
already exist. Run outputs land in `runs/` and derived tables/figures in
`results/` (both gitignored).

## Reading the figures

- All arms share the U1 train/test split and score with GroupKFold(5) on
  `host_id`, so the score panel compares absolute validation RMSE.
- The comparison is about what each method searches over, not how fast it
  searches: auto-sklearn evaluates more configurations per allocation than
  ritme yet stays well above its validation RMSE at every point. ritme
  searches microbiome-aware feature engineering plus metadata enrichment,
  auto-sklearn the relative-abundance table with its own generic
  preprocessors.
- Each method converts cores into work its own way, which the utilisation
  panel reflects: ritme runs cores/4 concurrent trials with xgb's `nthread`
  set to the 4 CPUs Ray gives each, auto-sklearn and TPOT run one
  single-threaded worker per core.
- auto-sklearn caps memory per configuration (3584 MB, one core's share) and
  records configurations killed by it as MEMOUT in `*_runs.csv`; ritme trials
  are bounded only by the job allocation. State the cap in the caption beside
  peak RSS.
- At 4 cores ritme is memory-bound rather than compute-bound: its fixed
  footprint does not shrink with the allocation, and one of the three seeds
  was killed after exhausting its budget. That seed's outputs are complete and
  kept; `state` in `b2_summary.csv` records which runs ended OUT_OF_MEMORY.
- At 4 cores the budget also buys around as many ritme trials as TPE's
  75-trial warm-up, so that point still largely reflects random sampling.
- *Configurations explored* counts every finished trial with a validation
  estimate; *best validation score* is taken over full-fold trials only, so it
  remains a 5-fold mean even for a build that prunes. `n_configs_full` and
  `n_configs_pruned` are reported alongside.
- TPOT points recovered from a crashed run's log have no exact configuration
  count, only the log's upper bound: they are drawn as open markers, kept out
  of the median line, and recorded in `n_configs_upper_bound`.
- EPS has no alpha channel, so the min-max bands are flattened onto white for
  that format only; where two bands overlap the EPS shows the one drawn last.
