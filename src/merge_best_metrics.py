#!/usr/bin/env python3
"""Recursively collect all `best_metrics.csv` files under a root directory,
annotate each row with its experiment folder name plus configuration
(num_trials, max_cuncurrent_trials) and elapsed_time from logs, and emit
a single merged CSV, filling missing results with NaNs. Optionally,
append an external CSV of original model runs into the final output."""

import argparse
import datetime
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, List

import pandas as pd
from tqdm import tqdm


def merge_metrics(root: Path) -> pd.DataFrame:
    """
    Walk the immediate subdirectories of `root` (except `logs`), read each
    `best_metrics.csv` (or produce a NaN row), read `experiment_config.json`
    for num_trials/max_cuncurrent_trials and its creation date as launch_date,
    parse elapsed_time from logs/<exp>_out.txt (with utf-8 replace) looking for
    a D-HH:MM:SS timestamp, "DUE TO TIME LIMIT" or "CANCELLED AT" in the last 5
    lines, and return a DataFrame with columns [
      experiment, model, num_trials, max_cuncurrent_trials, <metrics…>,
      elapsed_time, launch_date
    ]. elapsed_time is HH:MM:SS, "error_time_limit", "cancelled" or "error";
    launch_date is YYYY-MM-DD or "error".
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Input path is not a directory: {root}")

    exp_dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name != "logs"]
    if not exp_dirs:
        raise FileNotFoundError(f"No subdirectories found under: {root}")

    records: List[pd.DataFrame] = []
    for exp_dir in tqdm(exp_dirs, desc="Experiments", file=sys.stdout):
        csv_file = exp_dir / "best_metrics.csv"
        df = (
            pd.read_csv(csv_file, index_col=0)
            if csv_file.exists()
            else pd.DataFrame(index=[pd.NA])
        )

        cfg_file = exp_dir / "experiment_config.json"
        if cfg_file.exists():
            text = cfg_file.read_text(encoding="utf-8", errors="replace")
            cfg: Any = json.loads(text)
            num_trials = cfg.get("num_trials", pd.NA)
            max_conc = cfg.get("max_cuncurrent_trials", pd.NA)
            launch_date = (
                datetime.datetime.fromtimestamp(cfg_file.stat().st_ctime)
                .date()
                .isoformat()
            )
        else:
            num_trials, max_conc = pd.NA, pd.NA
            launch_date = "error"

        log_file = root / "logs" / f"{exp_dir.name}_out.txt"
        if log_file.exists():
            text = log_file.read_text(encoding="utf-8", errors="replace")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if lines:
                raw = lines[-1]
                m = re.match(r"(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})", raw)
                if m:
                    days = int(m.group(1) or 0)
                    h, mi, s = map(int, m.group(2, 3, 4))
                    total_h = days * 24 + h
                    elapsed_time = f"{total_h:02d}:{mi:02d}:{s:02d}"
                elif any("DUE TO TIME LIMIT" in ln for ln in lines[-5:]):
                    elapsed_time = "error_time_limit"
                elif any("CANCELLED AT" in ln for ln in lines[-5:]):
                    elapsed_time = "cancelled"
                else:
                    elapsed_time = "error"
            else:
                elapsed_time = "error"
        else:
            elapsed_time = "error"

        df["experiment"] = exp_dir.name
        df["num_trials"] = num_trials
        df["max_cuncurrent_trials"] = max_conc
        df["elapsed_time"] = elapsed_time
        df["launch_date"] = launch_date
        records.append(df)

    merged = pd.concat(records, axis=0).reset_index().rename(columns={"index": "model"})

    drop_cols = {
        "experiment",
        "model",
        "num_trials",
        "max_cuncurrent_trials",
        "elapsed_time",
        "launch_date",
    }
    metric_cols = [c for c in merged.columns if c not in drop_cols]
    other_cols = [c for c in metric_cols if "pvalue" not in c.lower()]
    merged[other_cols] = merged[other_cols].round(3)

    front = ["experiment", "model", "num_trials", "max_cuncurrent_trials"]
    rest = [
        c for c in merged.columns if c not in front + ["elapsed_time", "launch_date"]
    ]
    merged = merged[front + rest + ["elapsed_time", "launch_date"]]

    return merged


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge best_metrics.csv + configs and logs; optionally append an extra CSV."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Root folder containing experiment subdirectories.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("merged_metrics.csv"),
        help="Output CSV file.",
    )
    parser.add_argument(
        "-e",
        "--extra-csv",
        type=Path,
        help="Optional CSV file with the same schema to append to the merged results.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    merged_df = merge_metrics(args.input)
    if args.extra_csv:
        original = pd.read_csv(args.extra_csv)
        merged_df = pd.concat([merged_df, original], ignore_index=True)
        logging.info(f"Appended {len(original)} rows from {args.extra_csv!r}")

    merged_df.to_csv(args.output, index=False)
    logging.info(f"Merged {len(merged_df)} total rows into {args.output!r}")


if __name__ == "__main__":
    main()
