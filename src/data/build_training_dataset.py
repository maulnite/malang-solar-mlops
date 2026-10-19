"""LK-06: Build labeled regression data from LK-05 DVC forecast history + ERA5.

Run from repository root:
    python src/data/build_training_dataset.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from preprocess import preprocess_era5, preprocess_forecast


HISTORY_PATH = Path('data/raw/ecmwf_ifs_history.csv')
RAW_ERA5_DIR = Path('data/raw/era5')
OUTPUT_PATH = Path('data/processed/training/ecmwf_ifs_labeled.parquet')
KEY_COLUMNS = ['source_model', 'run_time_utc', 'valid_time_utc']


def main() -> None:
    if not HISTORY_PATH.is_file():
        raise FileNotFoundError(
            f'{HISTORY_PATH} not found. LK-05 DVC history must be restored locally.'
        )

    # Reuse the exact LK-04 validation + time/wind feature engineering.
    features_path = preprocess_forecast(HISTORY_PATH)
    forecast = pd.read_parquet(features_path)
    forecast['valid_time_utc'] = pd.to_datetime(forecast['valid_time_utc'], utc=True)

    if forecast.duplicated(KEY_COLUMNS).any():
        raise ValueError('Duplicate forecast key in LK-05 history.')

    forecast_dates = sorted(forecast['valid_time_utc'].dt.date.unique())
    era5_frames = []
    for forecast_date in forecast_dates:
        raw_file = RAW_ERA5_DIR / f'era5_{forecast_date:%Y%m%d}.csv'
        if not raw_file.is_file():
            raise FileNotFoundError(
                f'ERA5 reference is missing: {raw_file}\n'
                f'Run: python src/data/ingest_data.py --source era5 --date {forecast_date:%Y-%m-%d}'
            )
        clean_path = preprocess_era5(raw_file)
        era5_frames.append(pd.read_parquet(clean_path)[['valid_time_utc', 'actual_ghi']])

    era5 = pd.concat(era5_frames, ignore_index=True)
    era5['valid_time_utc'] = pd.to_datetime(era5['valid_time_utc'], utc=True)
    if era5.duplicated(['valid_time_utc']).any():
        raise ValueError('Duplicate valid_time_utc in ERA5 reference files.')

    labeled = forecast.merge(
        era5,
        on='valid_time_utc',
        how='left',
        validate='many_to_one',
    )
    if labeled['actual_ghi'].isna().any():
        examples = labeled.loc[labeled['actual_ghi'].isna(), 'valid_time_utc'].head(3).tolist()
        raise ValueError(f'ERA5 label missing for forecast timestamps: {examples}')

    labeled['residual'] = labeled['actual_ghi'] - labeled['raw_ghi_forecast']
    labeled = labeled.sort_values(['run_time_utc', 'valid_time_utc']).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(OUTPUT_PATH, index=False)

    print('\n=== LK-06 LABEL DATASET ===')
    print(f'DVC history        : {HISTORY_PATH}')
    print(f'Forecast rows      : {len(forecast)}')
    print(f'ERA5 matched rows  : {labeled["actual_ghi"].notna().sum()}')
    print(f'Unique runs        : {labeled["run_time_utc"].nunique()}')
    print(f'Daylight rows      : {(labeled["is_day"] == 1).sum()}')
    print(f'Missing target     : {labeled["residual"].isna().sum()}')
    print(f'Output             : {OUTPUT_PATH}')
    print('SUCCESS')


if __name__ == '__main__':
    main()