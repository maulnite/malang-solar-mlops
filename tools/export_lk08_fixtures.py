"""Export a *frozen* LK-05 forecast history + delayed ERA5 reference fixture.

Run once on the original local machine that retains the DVC V2 dataset and
ERA5 raw files. The generated fixture is for LK-08 CI demonstration, not
future data ingestion or a replacement for DVC remote storage.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FORECAST_HISTORY = ROOT / 'data/raw/ecmwf_ifs_history.csv'
DVC_POINTER = ROOT / 'data/raw/ecmwf_ifs_history.csv.dvc'
RAW_ERA5 = ROOT / 'data/raw/era5'
FIXTURES = ROOT / 'data/fixtures'
KEYS = ['source_model', 'run_time_utc', 'valid_time_utc']


def expected_md5(metadata: str) -> str:
    found = re.search(r'\bmd5:\s*([0-9a-fA-F]{32})\b', metadata)
    if found is None:
        raise ValueError('DVC metadata missing md5 field')
    return found.group(1).lower()


def validate_history(source: Path, pointer: Path) -> pd.DataFrame:
    data = source.read_bytes()
    digest = hashlib.md5(data).hexdigest()
    expected = expected_md5(pointer.read_text(encoding='utf-8'))
    if digest != expected:
        raise RuntimeError(f'DVC forecast mismatch: expected {expected}, actual {digest}')
    df = pd.read_csv(source)
    if df.empty or any(k not in df.columns for k in KEYS) or df.duplicated(KEYS).any():
        raise RuntimeError('Invalid or duplicate history keys')
    return df


def main() -> None:
    forecast = validate_history(FORECAST_HISTORY, DVC_POINTER)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / 'lk08_ecmwf_history_v2.csv').write_bytes(FORECAST_HISTORY.read_bytes())

    valid = pd.to_datetime(forecast['valid_time_utc'], utc=True, errors='raise')
    keys = pd.DataFrame({'valid_time_utc': valid.dt.strftime('%Y-%m-%dT%H:%M:%SZ')})
    dates = sorted(valid.dt.strftime('%Y%m%d').unique())
    reference_dir = FIXTURES / 'lk08_era5'
    reference_dir.mkdir(exist_ok=True)
    for day in dates:
        source = RAW_ERA5 / f'era5_{day}.csv'
        if not source.is_file():
            raise FileNotFoundError(f'Missing original ERA5 file: {source}')
        ref = pd.read_csv(source)
        required = {'source_model', 'valid_time_utc', 'actual_ghi'}
        if not required.issubset(ref):
            raise ValueError(f'Invalid columns in {source}')
        ref = ref[['source_model', 'valid_time_utc', 'actual_ghi']].copy()
        ref['valid_time_utc'] = pd.to_datetime(ref['valid_time_utc'], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        ref = ref.loc[ref['valid_time_utc'].isin(keys['valid_time_utc'])].drop_duplicates('valid_time_utc')
        if ref.empty or ref['actual_ghi'].isna().any() or (ref['actual_ghi'] < 0).any():
            raise RuntimeError(f'Invalid ERA5 reference subset: {source}')
        output = reference_dir / f'era5_{day}.csv'
        ref.sort_values('valid_time_utc').to_csv(output, index=False)
        print(f'{output.relative_to(ROOT)}: {len(ref)} rows')

    refs = pd.concat((pd.read_csv(p) for p in reference_dir.glob('era5_*.csv')), ignore_index=True)
    if not keys['valid_time_utc'].isin(refs['valid_time_utc']).all():
        raise RuntimeError('At least one forecast valid_time has no frozen ERA5 label')
    print(f'LK08 EXPORT: PASS — {len(forecast)} forecast rows, {len(dates)} ERA5 dates')
    print('These fixtures are immutable demonstration snapshots; not real-time data.')


if __name__ == '__main__':
    main()
