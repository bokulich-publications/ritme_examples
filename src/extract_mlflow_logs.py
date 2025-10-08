import argparse
import ast
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import mlflow
import pandas as pd
from tqdm import tqdm


def post_process_data_transform(all_trials):
    """
    function copied from newest ritme version - ritme v1.2.4 used for these
    experiments did not have this fixed yet
    """
    proc_trials = all_trials.copy()
    # none, shannon, metadata_only shannon_and_metadata

    #  get number of metadata fields
    proc_trials.loc[proc_trials["params.data_enrich"] == "None", "nb_md_fts"] = 0
    proc_trials.loc[proc_trials["params.data_enrich"] == "shannon", "nb_md_fts"] = 1

    if "params.data_enrich_with" in proc_trials.columns:
        nb_enrich_fts = (
            proc_trials["params.data_enrich_with"]
            .where(
                proc_trials["params.data_enrich_with"].notna()
                & (proc_trials["params.data_enrich_with"] != "None"),
                "[]",
            )
            .map(ast.literal_eval)
            .str.len()
        )
        proc_trials.loc[
            proc_trials["params.data_enrich"] == "metadata_only", "nb_md_fts"
        ] = nb_enrich_fts
        proc_trials.loc[
            proc_trials["params.data_enrich"] == "shannon_and_metadata", "nb_md_fts"
        ] = nb_enrich_fts + 1

    # from this retrieve # microbiome features
    proc_trials["nb_microbiome_fts"] = (
        proc_trials["metrics.nb_features"] - proc_trials["nb_md_fts"]
    )

    # wherever nb_microbiome_fts is 1, set data_transform to None
    # this was performed in ritme - but MLflow UI did not update this automatically
    proc_trials.loc[proc_trials["nb_microbiome_fts"] == 1, "params.data_transform"] = (
        "None"
    )
    # drop helper columns
    proc_trials = proc_trials.drop(columns=["nb_md_fts", "nb_microbiome_fts"])
    return proc_trials


def _fetch_runs_for_directory(log_folder_location: str) -> pd.DataFrame:
    """Fetch and post-process MLflow runs for a single tracking directory.

    Parameters
    ----------
    log_folder_location : str
        Base path containing an `mlruns` directory (or itself is the tracking
        dir).

    Returns
    -------
    pd.DataFrame
        Post-processed runs DataFrame.
    """
    print(f"post-processing {log_folder_location}")
    mlflow_folder_location = os.path.join(log_folder_location, "mlruns")
    # Allow user to pass either the parent or the mlruns folder itself.
    if os.path.isdir(mlflow_folder_location):
        tracking_uri = mlflow_folder_location
    else:  # fall back to provided path
        tracking_uri = log_folder_location
    mlflow.set_tracking_uri(tracking_uri)
    trials = mlflow.search_runs(
        order_by=["metrics.rmse_val ASC"], search_all_experiments=True
    )
    trials = post_process_data_transform(trials)
    return trials


def extract_mlflow_logs(
    ls_folder_locations: List[str],
    output_filename: str,
    workers: int | None = None,
    show_progress: bool = True,
) -> str:
    """Extract and merge MLflow runs from multiple tracking directories.

    Parameters
    ----------
    ls_folder_locations : list[str]
        MLflow tracking URI directories (paths containing an mlruns folder).
    output_filename : str
        Base filename (without extension). The .csv extension will be appended.

    Returns
    -------
    str
        The path to the written CSV file.
    """
    # Determine worker count (cap by number of dirs and CPU count)
    if workers is None or workers <= 0:
        cpu_count = os.cpu_count() or 1
        workers = min(len(ls_folder_locations), cpu_count)

    all_trials: list[pd.DataFrame] = []
    total = len(ls_folder_locations)
    use_tqdm = bool(show_progress and tqdm)
    if show_progress and not use_tqdm:
        # Simple textual header only if tqdm not available
        print(
            f"Starting extraction from {total} tracking directories "
            f"using {workers} worker(s)...",
            file=sys.stderr,
        )
    bar = None
    if use_tqdm:
        bar = tqdm(total=total, desc="Extracting runs", unit="dir")
    if workers == 1:
        for idx, d in enumerate(ls_folder_locations, start=1):
            df = _fetch_runs_for_directory(d)
            all_trials.append(df)
            if use_tqdm and bar:
                bar.update(1)
                bar.set_postfix(runs=len(df))
            elif show_progress:
                print(
                    f"[{idx}/{total}] Collected {len(df)} runs from: {d}",
                    file=sys.stderr,
                )
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(_fetch_runs_for_directory, d): d
                for d in ls_folder_locations
            }
            for future in as_completed(future_map):
                df = future.result()
                all_trials.append(df)
                completed += 1
                src = future_map[future]
                if use_tqdm and bar:
                    bar.update(1)
                    bar.set_postfix(runs=len(df))
                elif show_progress:
                    print(
                        f"[{completed}/{total}] Collected {len(df)} runs from: {src}",
                        file=sys.stderr,
                    )

    # merge all dataframes in all_trials list
    all_trials_df = pd.concat(all_trials, ignore_index=True)

    output_path = Path(output_filename).with_suffix(".csv")
    all_trials_df.to_csv(output_path, index=False)
    if bar:
        bar.close()
    print(
        f"Extracted {len(all_trials_df)} trials in {output_path} "
        f"from {len(ls_folder_locations)} folders."
    )
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and merge MLflow runs from one or more tracking directories "
            "into a single CSV (sorted by rmse_val)."
        ),
        epilog=(
            "Example: python extract_mlflow_logs.py --dirs ./mlruns_a ./mlruns_b "
            "-o merged_runs"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dirs",
        metavar="DIR",
        nargs="+",
        required=True,
        help="One or more MLflow tracking directories (paths containing mlruns).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="all_trials",
        help="Output CSV filename stem (without .csv).",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel worker processes. Defaults to min(CPU count, "
            "number of directories). Use 1 to disable parallelism."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress reporting to stderr.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    extract_mlflow_logs(
        args.dirs,
        args.output,
        args.workers,
        show_progress=not args.no_progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
