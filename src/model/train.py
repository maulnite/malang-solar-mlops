"""LK-06: Track three residual-regression experiments with MLflow and register best.

Run from repository root:
    python src/model/train.py
    mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
"""
from __future__ import annotations

import json
import re
import subprocess
from math import ceil, sqrt
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = Path('data/processed/training/ecmwf_ifs_labeled.parquet')
DVC_META_PATH = Path('data/raw/ecmwf_ifs_history.csv.dvc')
OUTPUT_DIR = Path('reports')
TRACKING_URI = 'sqlite:///mlflow.db'
EXPERIMENT_NAME = 'LK06-Malang-Solar-Residual-Correction'
REGISTERED_MODEL_NAME = 'MalangSolarResidualCorrector'
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    'temperature_2m', 'relative_humidity_2m', 'precipitation',
    'cloud_cover', 'pressure_msl', 'wind_speed_10m',
    'raw_ghi_forecast', 'lead_time_hour', 'hour_sin', 'hour_cos',
    'day_of_year_sin', 'day_of_year_cos', 'wind_dir_sin', 'wind_dir_cos',
]

# At least three distinct hyperparameter experiments, same temporal validation split.
CONFIGS = [
    {'name': 'RF-50-depth3', 'n_estimators': 50, 'max_depth': 3, 'min_samples_leaf': 1},
    {'name': 'RF-100-depth6', 'n_estimators': 100, 'max_depth': 6, 'min_samples_leaf': 1},
    {'name': 'RF-200-leaf2', 'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 2},
]


def provenance() -> dict[str, str]:
    raw = DVC_META_PATH.read_text(encoding='utf-8')
    match = re.search(r'\bmd5:\s*([0-9a-f]{32})\b', raw)
    if not match:
        raise ValueError(f'No MD5 hash in {DVC_META_PATH}')
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    return {'dvc_forecast_md5': match.group(1), 'git_commit': commit}


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f'{DATA_PATH} not found. Run: python src/data/build_training_dataset.py'
        )
    df = pd.read_parquet(DATA_PATH)
    needed = FEATURE_COLUMNS + [
        'source_model', 'run_time_utc', 'valid_time_utc',
        'raw_ghi_forecast', 'actual_ghi', 'residual', 'is_day',
    ]
    missing = sorted(set(needed) - set(df.columns))
    if missing:
        raise ValueError(f'Dataset missing columns: {missing}')

    df['run_time_utc'] = pd.to_datetime(df['run_time_utc'], utc=True)
    df['valid_time_utc'] = pd.to_datetime(df['valid_time_utc'], utc=True)
    df = df.loc[df['is_day'] == 1].copy()
    if df[needed].isna().any().any():
        raise ValueError('Missing values found in supervised training columns.')

    runs = sorted(df['run_time_utc'].unique())
    if len(runs) < 2:
        raise ValueError('Need at least 2 forecast runs for time-based train/validation split.')
    n_val_runs = max(1, ceil(len(runs) * 0.2))
    train = df.loc[~df['run_time_utc'].isin(runs[-n_val_runs:])].copy()
    valid = df.loc[df['run_time_utc'].isin(runs[-n_val_runs:])].copy()
    if len(train) < 3 or len(valid) < 2:
        raise ValueError(
            f'Insufficient daylight observations: train={len(train)}, valid={len(valid)}. '
            'Collect more distinct forecast runs to obtain a meaningful experiment.'
        )
    if train['valid_time_utc'].max() >= valid['valid_time_utc'].min():
        raise ValueError('Temporal overlap between training and validation sets.')
    return train, valid


def main() -> None:
    train, valid = load_and_split()
    X_train = train[FEATURE_COLUMNS].astype(float)
    y_train = train['residual'].astype(float)
    X_valid = valid[FEATURE_COLUMNS].astype(float)
    actual = valid['actual_ghi'].to_numpy(dtype=float)
    raw = valid['raw_ghi_forecast'].to_numpy(dtype=float)

    baseline_mae = float(mean_absolute_error(actual, raw))
    source_tags = provenance()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print('=== LK-06 MLFLOW EXPERIMENTS ===')
    print(f'Train: {len(train)} daylight rows; validation: {len(valid)} daylight rows')
    print(f'Forecast runs train={train["run_time_utc"].nunique()}, validation={valid["run_time_utc"].nunique()}')
    print(f'Raw baseline validation MAE: {baseline_mae:.3f} W/m²')
    if len(train) < 100:
        print('WARNING: very small training dataset; results are MLflow demonstrations, NOT final model benchmarks.')

    rows = []
    for config in CONFIGS:
        name = config['name']
        params = {k: v for k, v in config.items() if k != 'name'}
        model = RandomForestRegressor(
            **params, random_state=RANDOM_STATE, n_jobs=-1
        )
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({**params, 'random_state': RANDOM_STATE})
            mlflow.set_tags({
                **source_tags,
                'task': 'ghi_residual_regression',
                'source_dataset': 'LK05 DVC ECMWF history + ERA5',
                'evaluation': 'temporal_validation_by_forecast_run',
                'scope': 'LK06 demonstration (not production benchmark)',
            })
            model.fit(X_train, y_train)
            residual_prediction = model.predict(X_valid)
            corrected = np.clip(raw + residual_prediction, a_min=0.0, a_max=None)
            mae = float(mean_absolute_error(actual, corrected))
            rmse = float(sqrt(mean_squared_error(actual, corrected)))
            bias = float(np.mean(corrected - actual))
            improvement = 100 * (baseline_mae - mae) / baseline_mae if baseline_mae > 0 else 0.0
            mlflow.log_metrics({
                'val_mae': mae,
                'val_rmse': rmse,
                'val_bias': bias,
                'baseline_raw_mae': baseline_mae,
                'improvement_percent': improvement,
            })
            info = mlflow.sklearn.log_model(
                sk_model=model,
                name='residual_model',
                input_example=X_train.head(2),
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ],
            )
            rows.append({
                'run_name': name,
                'run_id': run.info.run_id,
                'model_uri': info.model_uri,
                'n_estimators': params['n_estimators'],
                'max_depth': params['max_depth'],
                'min_samples_leaf': params['min_samples_leaf'],
                'val_mae': mae,
                'val_rmse': rmse,
                'val_bias': bias,
                'raw_mae': baseline_mae,
                'improvement_percent': improvement,
            })
            print(f'{name}: MAE={mae:.3f} RMSE={rmse:.3f} Bias={bias:.3f}')

    ranking = pd.DataFrame(rows).sort_values('val_mae').reset_index(drop=True)
    best = ranking.iloc[0].to_dict()
    registered = mlflow.register_model(
        model_uri=best['model_uri'],
        name=REGISTERED_MODEL_NAME,
        await_registration_for=30,
    )
    MlflowClient().set_registered_model_alias(
        REGISTERED_MODEL_NAME, 'lk06-candidate', registered.version
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUTPUT_DIR / 'lk06_experiment_summary.csv', index=False)
    metadata = {
        'registered_model': REGISTERED_MODEL_NAME,
        'registered_version': registered.version,
        'registered_alias': 'lk06-candidate',
        'selection_criterion': 'lowest temporal validation MAE',
        'selected_run': best['run_name'],
        'run_id': best['run_id'],
        'model_uri': best['model_uri'],
        'best_validation_mae': best['val_mae'],
        'best_validation_rmse': best['val_rmse'],
        'best_validation_bias': best['val_bias'],
        'raw_validation_mae': best['raw_mae'],
        'train_rows': len(train),
        'validation_rows': len(valid),
        'dvc_forecast_md5': source_tags['dvc_forecast_md5'],
        'git_commit': source_tags['git_commit'],
        'note': 'Selected from LK-06 validation only; do not treat as deployed champion.',
    }
    (OUTPUT_DIR / 'lk06_best_model.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print('\n=== EXPERIMENT RANKING ===')
    print(ranking[['run_name', 'val_mae', 'val_rmse', 'val_bias', 'improvement_percent']].to_string(index=False))
    print(f'\nRegistered candidate: {REGISTERED_MODEL_NAME} v{registered.version}')
    print('Metadata saved: reports/lk06_best_model.json')
    print('Start MLflow UI: mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000')


if __name__ == '__main__':
    main()