"""Diagnose how much of its CPU allocation each ritme search actually used.

sacct reports what a job consumed; it does not say why a job consumed less
than it reserved. This script decomposes the shortfall into the only two
factors that can produce it, both measurable from the trial logs:

    utilisation = (trials in flight / max_cuncurrent_trials)
                  x (cores a running trial uses / cpus_per_trial)

The first factor is a *scheduling* loss (reserved slots sitting empty), the
second a *parallelism* loss (a running trial holding CPUs its model cannot
use). They are independent and multiply, so a search can be starved by
either or both, and the fix for one does nothing for the other.

Trials in flight is recovered as sum(trial durations) / wall time, and cores
per running trial as sacct TotalCPU / sum(trial durations). Inputs are the
mlflow trial logs plus sacct, so nothing new has to be run.

Usage: python -m benchmarking.analyze_utilization
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd

from src.launch_models import MAX_CONCURRENT_TRIALS, REPO_ROOT

from benchmarking.common import DATA_DIR, RUNS_DIR, parse_slurm_duration

ARCHIVED_DIR = REPO_ROOT / "use_cases" / "ritme_runs" / "local"
USECASES_DIR = REPO_ROOT / "use_cases"

# The manuscript's final run set is whatever the newest merged metrics table
# contains, minus the non-ritme baselines; `collect_b3.py` reads accuracy from
# the same file. Job ids are not pinned here because they are recoverable:
# ritme's SLURM job name is the experiment tag, and the run that produced the
# archived outputs is the last one that completed under that name (validated
# against the five ids pinned in collect_b3.py).
BASELINE_SUFFIXES = ("_original", "_automl")
SACCT_WINDOW = ("2026-05-01", "2026-08-01")


def final_experiments() -> tuple[list[str], Path]:
    """The manuscript's ritme experiments, from the newest metrics table."""
    tables = sorted(USECASES_DIR.glob("all_experiments_metrics_*.csv"))
    if not tables:
        raise SystemExit("No use_cases/all_experiments_metrics_*.csv found.")
    frame = pd.read_csv(tables[-1])
    names = [e for e in frame["experiment"] if not e.endswith(BASELINE_SUFFIXES)]
    return names, tables[-1]


def sacct_by_name(names: list[str]) -> pd.DataFrame:
    """Every top-level SLURM job carrying one of these experiment tags."""
    fields = "JobID,JobName,State,Start,End,Elapsed,TotalCPU,NCPUS"
    completed = subprocess.run(
        [
            "sacct",
            "-S",
            SACCT_WINDOW[0],
            "-E",
            SACCT_WINDOW[1],
            "--name=" + ",".join(names),
            "--parsable2",
            "--noheader",
            f"--format={fields}",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    frame = pd.DataFrame(
        [line.split("|") for line in completed.strip().splitlines()],
        columns=fields.split(","),
    )
    # Job steps (`<id>.batch`, `<id>.extern`) repeat the parent's accounting.
    frame = frame[~frame["JobID"].str.contains(r"\.")].copy()
    frame["elapsed_s"] = frame["Elapsed"].map(parse_slurm_duration)
    frame["total_cpu_s"] = frame["TotalCPU"].map(parse_slurm_duration)
    frame["ncpus"] = frame["NCPUS"].astype(int)
    return frame


def _trial_spans(log: Path, window: tuple | None = None) -> tuple:
    """Per-trial wall durations, launch gaps, and the imputed-time fraction.

    Trials still RUNNING when the time budget expires have no ``end_time``.
    Dropping them would undercount occupancy badly wherever slots are many
    and trials long -- at 128 cores they hold 29% of all trial-time -- so
    they are closed at the last event observed in the log. That is a lower
    bound on their true end, and the returned fraction says how much of the
    total rests on it.

    ``window`` restricts the log to one job's wall-clock span. Some archived
    logs accumulated several attempts of the same experiment (u3_logreg_tpe
    holds three, spanning 72 h), so without it the retries would be pooled
    into the run that actually produced the results.

    Gaps come from the start times of every trial, RUNNING ones included,
    and measure how fast the driver launches trials.
    """
    frame = pd.read_csv(log)
    start = pd.to_datetime(frame["start_time"], format="mixed")
    end = pd.to_datetime(frame["end_time"], format="mixed")
    if window is not None:
        keep = (start >= window[0]) & (start <= window[1])
        start, end = start[keep], end[keep]
    if start.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0.0
    cutoff = max(end.max(), start.max())
    duration = (end.fillna(cutoff) - start).dt.total_seconds()
    duration = duration[duration >= 0]
    imputed = (cutoff - start[end.isna()]).dt.total_seconds().clip(lower=0).sum()
    gap = start.sort_values().diff().dt.total_seconds().dropna()
    return duration, gap, imputed / duration.sum() if duration.sum() else 0.0


def _decompose(
    duration: pd.Series,
    gap: pd.Series,
    elapsed_s: float,
    total_cpu_s: float,
    ncpus: int,
    max_concurrent: int,
) -> dict:
    trial_s = duration.sum()
    in_flight = trial_s / elapsed_s
    cores_per_trial = total_cpu_s / trial_s
    cpus_per_trial = ncpus // max_concurrent
    return {
        "n_trials": len(duration),
        "ncpus": ncpus,
        "median_trial_s": duration.median(),
        "mean_trial_s": duration.mean(),
        # What the launch path would have to sustain to keep every slot full.
        "required_launch_gap_s": duration.mean() / max_concurrent,
        "median_launch_gap_s": gap.median(),
        "p1_launch_gap_s": gap.quantile(0.01),
        "trials_in_flight": in_flight,
        "max_concurrent": max_concurrent,
        "slot_fill": in_flight / max_concurrent,
        "cores_per_trial": cores_per_trial,
        "cpus_per_trial": cpus_per_trial,
        "core_fill": cores_per_trial / cpus_per_trial,
        # The three core-hour levels the two gaps sit between.
        "ideal_core_h": ncpus * elapsed_s / 3600,
        "reserved_core_h": trial_s * cpus_per_trial / 3600,
        "used_core_h": total_cpu_s / 3600,
        "utilisation": total_cpu_s / (ncpus * elapsed_s),
    }


def effective_concurrency(model: str, max_concurrent: int) -> int:
    """The slot count ritme actually launches with.

    ritme divides trac's concurrency by three to cap memory
    (``tune_models.py:800-805``). Because ``_get_resources`` then divides the
    allocation by the *reduced* number, this also triples what each trac trial
    reserves, so the effective value is what both factors must be measured
    against.
    """
    if model == "trac":
        return max(1, round(max_concurrent / 3))
    return max_concurrent


def _logged_cpus_per_trial(tag: str) -> set[int]:
    """The ``Using these resources: CPU n`` values ritme printed for a tag."""
    log = ARCHIVED_DIR / "logs" / f"{tag}_out.txt"
    if not log.exists():
        return set()
    values = re.findall(
        r"Using these resources: CPU (\d+)", log.read_text(errors="ignore")
    )
    return {int(v) for v in values}


def _model_of(tag: str) -> str:
    """Model family from an experiment tag (`u1_xgb_class_tpe` -> xgb_class)."""
    stem = tag.split("_", 1)[1]
    for suffix in ("_tpe_no_enrich", "_tpe_reduced", "_tpe_no_fit", "_tpe"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.removeprefix("dynamic_")


def archived_table() -> pd.DataFrame:
    """Decomposition of every ritme search in the manuscript's final set.

    For each experiment tag the producing job is the last one that completed
    under that name; earlier attempts are counted separately as retry cost.
    An experiment whose job never completed is reported with its outcome and
    no decomposition, because there is no successful run to decompose.
    """
    names, table_path = final_experiments()
    print(f"Final run set: {len(names)} ritme experiments from {table_path.name}")
    jobs = sacct_by_name(names)
    metrics = pd.read_csv(table_path).set_index("experiment")
    # An experiment yielded a result if the merged table carries any test
    # metric for it; a job can end FAILED in post-processing having already
    # written its outputs, and an all-NaN row means the budget bought nothing.
    scored = metrics.filter(regex=r"_test$").notna().any(axis=1)

    rows = []
    for tag in names:
        candidates = jobs[jobs["JobName"] == tag].sort_values("Start")
        if candidates.empty:
            print(f"[warn] no SLURM job found for {tag}; skipping")
            continue
        done = candidates[candidates["State"] == "COMPLETED"]
        producer = (done if len(done) else candidates).iloc[-1]
        earlier = candidates[candidates["Start"] < producer["Start"]]
        # An earlier attempt that COMPLETED was a deliberate re-run, not a
        # failure; only the ones that never completed are wasted compute.
        failed = earlier[earlier["State"] != "COMPLETED"]
        superseded = earlier[earlier["State"] == "COMPLETED"]

        model = _model_of(tag)
        row = {
            "tag": tag,
            "usecase": tag.split("_", 1)[0],
            "model": model,
            "job_id": producer["JobID"],
            "state": producer["State"],
            "produced_output": bool(scored.get(tag, False)),
            "n_attempts": len(candidates),
            "n_failed_attempts": len(failed),
            "failed_core_h": (failed["elapsed_s"] * failed["ncpus"]).sum() / 3600,
            # Attempts that burned essentially the whole budget before failing.
            "n_full_budget_failures": int((failed["elapsed_s"] >= 20 * 3600).sum()),
            "superseded_core_h": (superseded["elapsed_s"] * superseded["ncpus"]).sum()
            / 3600,
            # Set from sacct rather than from the decomposition, so that runs
            # with no usable trial log still carry what they cost.
            "allocated_core_h": producer["elapsed_s"] * producer["ncpus"] / 3600,
            "configured_concurrency": MAX_CONCURRENT_TRIALS.get(model),
        }
        configured = row["configured_concurrency"]
        row["max_concurrent"] = (
            None if configured is None else effective_concurrency(model, configured)
        )

        log = ARCHIVED_DIR / tag / "mlflow_logs.csv"
        if not log.exists() or row["max_concurrent"] is None:
            rows.append(row)
            continue

        # Cross-check the derived reservation against what ritme printed.
        derived = producer["ncpus"] // row["max_concurrent"]
        logged = _logged_cpus_per_trial(tag)
        if logged and derived not in logged:
            print(
                f"[warn] {tag}: derived cpus_per_trial={derived} but the log "
                f"reports {sorted(logged)}; using the derived value"
            )
        # mlflow stamps trials in UTC while sacct reports node-local time.
        window = (
            pd.Timestamp(producer["Start"])
            .tz_localize("Europe/Zurich")
            .tz_convert("UTC")
            .tz_localize(None),
            pd.Timestamp(producer["End"])
            .tz_localize("Europe/Zurich")
            .tz_convert("UTC")
            .tz_localize(None),
        )
        duration, gap, imputed = _trial_spans(log, window)
        if duration.empty:
            print(f"[warn] {tag}: no trials inside job {producer['JobID']}'s window")
            rows.append(row)
            continue
        row["imputed_trial_frac"] = imputed
        row.update(
            _decompose(
                duration,
                gap,
                producer["elapsed_s"],
                producer["total_cpu_s"],
                producer["ncpus"],
                row["max_concurrent"],
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def b2_table(benchmark: str = "b2", method: str = "ritme") -> pd.DataFrame:
    """Same decomposition across a B2-shaped core sweep (U1/xgb, 2 h, 3 seeds).

    Here ``cpus_per_trial`` is held at 4 and the slot count scales with the
    allocation, which isolates how slot fill behaves as slots are added.
    ``benchmark``/``method`` select the sweep: B2 itself, or a relabelled
    ritme build collected as ``results/data/b2_<label>_summary.csv`` (see
    ``launch_b2 --ritme-label``).
    """
    summary_path = DATA_DIR / f"{benchmark}_summary.csv"
    if not summary_path.exists():
        print(f"[warn] {summary_path} not found; run collect_b2 first")
        return pd.DataFrame()
    summary = pd.read_csv(summary_path)

    rows = []
    for _, job in summary[summary["method"] == method].iterrows():
        log = RUNS_DIR / benchmark / "ritme" / job["job_name"] / "mlflow_logs.csv"
        if not log.exists():
            print(f"[warn] missing trial log, skipping: {log}")
            continue
        duration, gap, imputed = _trial_spans(log)
        slots = job["cores"] // 4
        row = {
            "method": method,
            "cores": job["cores"],
            "seed": job["seed"],
            "slots": slots,
            "imputed_trial_frac": imputed,
            **_decompose(
                duration,
                gap,
                job["elapsed_s"],
                job["total_cpu_s"],
                job["ncpus"],
                slots,
            ),
        }
        # With a single slot the next trial cannot start until the previous
        # one ends, so this latency is the launch cost with nothing else in
        # flight to hide it -- the cleanest read on the serial launch path.
        if slots == 1:
            frame = pd.read_csv(log).sort_values("start_time")
            start = pd.to_datetime(frame["start_time"], format="mixed")
            end = pd.to_datetime(frame["end_time"], format="mixed")
            serial = (start.values[1:] - end.values[:-1]) / pd.Timedelta(seconds=1)
            row["serial_launch_s"] = pd.Series(serial).dropna().median()
        rows.append(row)
    return pd.DataFrame(rows)


def feature_cost_table() -> pd.DataFrame:
    """Trial duration against features kept, for the two linreg searches.

    An ElasticNet fit is sub-second at this data scale, so anything that
    makes a trial take a minute must be the feature-engineering prologue.
    Binning by ``nb_features`` separates the two: if fitting drove the cost,
    duration would rise with the feature count. It falls instead, because
    the expensive step is deciding which of ~35k features to keep.
    """
    rows = []
    for tag in ("u2_linreg_tpe", "u1_linreg_tpe"):
        log = ARCHIVED_DIR / tag / "mlflow_logs.csv"
        if not log.exists():
            continue
        frame = pd.read_csv(log)
        start = pd.to_datetime(frame["start_time"], format="mixed")
        end = pd.to_datetime(frame["end_time"], format="mixed")
        duration = (end - start).dt.total_seconds()
        features = frame["metrics.nb_features"]
        keep = duration.notna() & features.notna() & (duration >= 0)
        duration, features = duration[keep], features[keep]
        spearman = features.corr(duration, method="spearman")
        edges = [0, 500, 2000, 10000, 40000]
        for lo, hi in zip(edges[:-1], edges[1:]):
            band = (features >= lo) & (features < hi)
            if not band.any():
                continue
            rows.append(
                {
                    "tag": tag,
                    "features_lo": lo,
                    "features_hi": hi,
                    "n_trials": int(band.sum()),
                    "median_trial_s": duration[band].median(),
                    "q25_trial_s": duration[band].quantile(0.25),
                    "q75_trial_s": duration[band].quantile(0.75),
                    "spearman_features_vs_duration": spearman,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    archived = archived_table()
    if not archived.empty:
        ran = archived[archived["utilisation"].notna()].copy()
        print("\n=== Final manuscript searches: utilisation decomposition ===")
        shown = ran.copy()
        for column in ("slot_fill", "core_fill", "utilisation"):
            shown[column] = (100 * shown[column]).map("{:.1f}%".format)
        print(
            shown[
                [
                    "tag",
                    "model",
                    "ncpus",
                    "max_concurrent",
                    "cpus_per_trial",
                    "n_trials",
                    "median_trial_s",
                    "median_launch_gap_s",
                    "slot_fill",
                    "core_fill",
                    "utilisation",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:.2f}")
        )
        wasted = (ran["ideal_core_h"] - ran["used_core_h"]).sum()
        print(
            f"\n{len(ran)} searches: {ran['ideal_core_h'].sum():.0f} core-h allocated, "
            f"{ran['used_core_h'].sum():.0f} used, {wasted:.0f} wasted "
            f"({100 * wasted / ran['ideal_core_h'].sum():.0f}%)"
        )

        lost = archived[~archived["produced_output"]]
        if not lost.empty:
            print("\n=== Searches whose budget bought no result at all ===")
            print(
                lost[
                    [
                        "tag",
                        "model",
                        "state",
                        "job_id",
                        "n_attempts",
                        "allocated_core_h",
                    ]
                ].to_string(index=False, float_format=lambda v: f"{v:.0f}")
            )

        failed = archived[archived["n_failed_attempts"] > 0]
        if not failed.empty:
            print("\n=== Attempts that failed and had to be relaunched ===")
            print(
                failed[
                    [
                        "tag",
                        "n_failed_attempts",
                        "n_full_budget_failures",
                        "failed_core_h",
                    ]
                ].to_string(index=False, float_format=lambda v: f"{v:.0f}")
            )

        # Total cost of everything that yielded nothing: failed attempts, plus
        # the producing job itself where it too came back empty.
        dead = archived["failed_core_h"].sum() + lost["allocated_core_h"].sum()
        gross = (
            archived["allocated_core_h"].sum()
            + archived["failed_core_h"].sum()
            + archived["superseded_core_h"].sum()
        )
        print(
            f"\nAcross the final set: {gross:.0f} core-h charged in total; "
            f"{dead:.0f} core-h returned no result "
            f"({int(archived['n_full_budget_failures'].sum())} attempts failed "
            f"only after burning a full budget); "
            f"{archived['superseded_core_h'].sum():.0f} core-h were earlier "
            f"runs later re-run deliberately."
        )
        archived.to_csv(out_dir / "archived_utilisation.csv", index=False)

    b2 = b2_table()
    efficient = b2_table("b2_ritme_efficient", "ritme_efficient")
    if not b2.empty:
        print("\n=== B2: slot fill as slots are added (U1/xgb, mean over 3 seeds) ===")
        grouped = b2.groupby(["cores", "slots"]).mean(numeric_only=True).reset_index()
        grouped["slot_fill"] = (100 * grouped.slot_fill).map("{:.1f}%".format)
        grouped["core_fill"] = (100 * grouped.core_fill).map("{:.1f}%".format)
        grouped["utilisation"] = (100 * grouped.utilisation).map("{:.1f}%".format)
        print(
            grouped[
                [
                    "cores",
                    "slots",
                    "n_trials",
                    "median_trial_s",
                    "median_launch_gap_s",
                    "p1_launch_gap_s",
                    "imputed_trial_frac",
                    "slot_fill",
                    "core_fill",
                    "utilisation",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:.2f}")
        )
        b2.to_csv(out_dir / "b2_utilisation.csv", index=False)

    if not efficient.empty:
        efficient.to_csv(out_dir / "b2_ritme_efficient_utilisation.csv", index=False)
        both = pd.concat([b2, efficient], ignore_index=True)
        print(
            "\n=== ritme vs ritme_efficient: utilisation composition per core count ==="
        )
        comp = (
            both.groupby(["cores", "method"])
            .agg(
                n_trials=("n_trials", "mean"),
                median_trial_s=("median_trial_s", "mean"),
                p1_launch_gap_s=("p1_launch_gap_s", "mean"),
                median_launch_gap_s=("median_launch_gap_s", "mean"),
                slot_fill=("slot_fill", "mean"),
                core_fill=("core_fill", "mean"),
                utilisation=("utilisation", "mean"),
            )
            .reset_index()
        )
        shown = comp.copy()
        for column in ("slot_fill", "core_fill", "utilisation"):
            shown[column] = (100 * shown[column]).map("{:.1f}%".format)
        print(shown.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        comp.to_csv(out_dir / "b2_ritme_vs_efficient.csv", index=False)
        # Per-core-count gain of the new build over the old, on the two factors
        # and their product; >1 means the new build does better.
        wide = comp.pivot(index="cores", columns="method")
        if "ritme_efficient" in wide["utilisation"] and "ritme" in wide["utilisation"]:
            gain = pd.DataFrame(
                {
                    k: wide[k]["ritme_efficient"] / wide[k]["ritme"]
                    for k in ("n_trials", "slot_fill", "core_fill", "utilisation")
                }
            )
            print("\n=== gain: ritme_efficient / ritme ===")
            print(gain.to_string(float_format=lambda v: f"{v:.2f}x"))

        serial = b2["serial_launch_s"].dropna() if "serial_launch_s" in b2 else []
        if len(serial):
            print(
                f"\nSerial launch cost at 1 slot (end of a trial -> start of the "
                f"next): median {serial.median():.2f} s over {len(serial)} run(s)"
            )

    feature_cost = feature_cost_table()
    if not feature_cost.empty:
        print("\n=== linreg trials: duration falls as more features are kept ===")
        print(feature_cost.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        feature_cost.to_csv(out_dir / "feature_cost.csv", index=False)

    print(f"\nWrote tables to {out_dir}")


if __name__ == "__main__":
    main()
