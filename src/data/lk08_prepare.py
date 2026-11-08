"""Restore LK-05 DVC V2 history & ERA5 fixture inside a clean CI runner.

Does not call external APIs, does not require DagsHub/DVC remote, and refuses
history content whose MD5 disagrees with the Git-tracked .dvc pointer.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'data/fixtures/lk08_ecmwf_history_v2.csv'
DVC_META = ROOT / 'data/raw/ecmwf_ifs_history.csv.dvc'
HISTORY = ROOT / 'data/raw/ecmwf_ifs_history.csv'
ERA5_FIXTURE_DIR = ROOT / 'data/fixtures/lk08_era5'
ERA5_RAW_DIR = ROOT / 'data/raw/era5'
LABELED = ROOT / 'data/processed/training/ecmwf_ifs_labeled.parquet'


def compare_dvc_fixture(fixture: Path, pointer: Path) -> str:
    found = re.search(r'\bmd5:\s*([0-9a-fA-F]{32})\b', pointer.read_text(encoding='utf-8'))
    if not found:
        raise ValueError('DVC pointer has no MD5 field')
    expected = found.group(1).lower()
    actual = hashlib.md5(fixture.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'Fixture integrity failure: DVC MD5={expected} but fixture MD5={actual}')
    return actual


def main() -> None:
    md5 = compare_dvc_fixture(FIXTURE, DVC_META)
    forecast = pd.read_csv(FIXTURE)
    keys = ['source_model', 'run_time_utc', 'valid_time_utc']
    if forecast.empty or not set(keys).issubset(forecast) or forecast.duplicated(keys).any():
        raise ValueError('Invalid or duplicated forecast fixture records')
    times = pd.to_datetime(forecast['valid_time_utc'], utc=True, errors='raise')
    days = sorted(times.dt.strftime('%Y%m%d').unique())
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    if HISTORY.exists():
        if hashlib.md5(HISTORY.read_bytes()).hexdigest() != md5:
            raise RuntimeError('Local DVC history differs; stop rather than overwrite it')
    else:
        shutil.copyfile(FIXTURE, HISTORY)

    ERA5_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for day in days:
        source = ERA5_FIXTURE_DIR / f'era5_{day}.csv'
        destination = ERA5_RAW_DIR / f'era5_{day}.csv'
        if not source.is_file():
            raise FileNotFoundError(f'Missing ERA5 fixture: {source}')
        if not destination.is_file():
            shutil.copyfile(source, destination)
        else:
            # Avoid altering original source data on a developer's laptop.
            existing = pd.read_csv(destination)
            reference = pd.read_csv(source)
            lookup = dict(zip(pd.to_datetime(existing['valid_time_utc'], utc=True), existing['actual_ghi']))
            for _, row in reference.iterrows():
                t = pd.to_datetime(row['valid_time_utc'], utc=True)
                if t not in lookup or float(lookup[t]) != float(row['actual_ghi']):
                    raise RuntimeError(f'ERA5 local/fixture conflict for {t}; do not overwrite')
    print(f'DVC V2 MD5: {md5}; history={len(forecast)} rows, ERA5 days={days}', flush=True)
    subprocess.run([sys.executable, 'src/data/build_training_dataset.py'], cwd=ROOT, check=True)
    labeled = pd.read_parquet(LABELED)
    if labeled.empty or len(labeled) != len(forecast) or labeled['residual'].isna().any():
        raise ValueError('Training dataset integrity failure')
    print(f'LK08 DATA BOOTSTRAP: PASS ({len(labeled)} labeled rows)', flush=True)


if __name__ == '__main__':
    main()
