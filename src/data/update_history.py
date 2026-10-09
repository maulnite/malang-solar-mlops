
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


HISTORY_PATH = Path("data/raw/ecmwf_ifs_history.csv")
FORECAST_DIR = Path("data/raw/forecast")

KEY_COLUMNS = [
    "source_model",
    "run_time_utc",
    "valid_time_utc",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append new ECMWF runs to DVC history."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Specific forecast CSV (default: latest snapshot).",
    )
    args = parser.parse_args()

    if args.input:
        snapshot_path = args.input
    else:
        snapshots = sorted(FORECAST_DIR.glob("ecmwf_ifs_*.csv"))
        if not snapshots:
            raise FileNotFoundError("No forecast snapshots found.")
        snapshot_path = snapshots[-1]

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"History missing: {HISTORY_PATH}")

    history = pd.read_csv(HISTORY_PATH)
    incoming = pd.read_csv(snapshot_path)

    if set(history.columns) != set(incoming.columns):
        raise ValueError("History and snapshot schemas differ.")

    incoming = incoming[history.columns]

    for name, df in [("history", history), ("incoming", incoming)]:
        if df[KEY_COLUMNS].isna().any().any():
            raise ValueError(f"Missing primary key in {name}")
        if df.duplicated(subset=KEY_COLUMNS).any():
            raise ValueError(f"Duplicate primary key in {name}")

    existing_keys = pd.MultiIndex.from_frame(
        history[KEY_COLUMNS]
    )
    incoming_keys = pd.MultiIndex.from_frame(
        incoming[KEY_COLUMNS]
    )

    new_rows = incoming.loc[
        ~incoming_keys.isin(existing_keys)
    ].copy()

    print("=== DATASET HISTORY UPDATE ===")
    print(f"Snapshot         : {snapshot_path}")
    print(f"Rows before      : {len(history)}")
    print(f"New rows         : {len(new_rows)}")

    if new_rows.empty:
        print("No new records. History unchanged.")
        return

    updated = (
        pd.concat([history, new_rows], ignore_index=True)
        .sort_values(["run_time_utc", "valid_time_utc"])
        .reset_index(drop=True)
    )

    # Atomic replacement protects the previous DVC cache version.
    temp_path = HISTORY_PATH.with_name(
        HISTORY_PATH.name + ".tmp"
    )

    try:
        updated.to_csv(temp_path, index=False)
        os.replace(temp_path, HISTORY_PATH)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(f"Rows after       : {len(updated)}")
    print(f"Unique runs      : {updated['run_time_utc'].nunique()}")
    print(
        "Duplicates       : "
        f"{updated.duplicated(subset=KEY_COLUMNS).sum()}"
    )
    print("HISTORY UPDATE SUCCESS")


if __name__ == "__main__":
    main()