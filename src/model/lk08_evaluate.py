"""LK-08 independent model-quality gate, intentionally BEFORE registry writing.

The 5% operational gate is a coursework demonstration. LK-01's 10% target
remains a separate, NOT YET MET project success KPI on the tiny LK-06 sample.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error

REPORT = Path('reports/lk08/evaluation.json')
TRAINING = Path('data/processed/training/ecmwf_ifs_labeled.parquet')
METADATA = Path('reports/lk08/best_model.json')


def compute_gate(
    candidate_mae: float,
    candidate_rmse: float,
    candidate_bias: float,
    baseline_mae: float,
    baseline_rmse: float,
    min_improvement_pct: float = 5.0,
    max_abs_bias: float = 20.0,
) -> dict:
    values = (candidate_mae, candidate_rmse, candidate_bias, baseline_mae,
              baseline_rmse, min_improvement_pct, max_abs_bias)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Non-finite evaluation metrics are not allowed.')
    if baseline_mae <= 0 or baseline_rmse <= 0 or candidate_mae < 0 or candidate_rmse < 0:
        raise ValueError('Invalid baseline/candidate metrics.')
    if min_improvement_pct < 0 or max_abs_bias < 0:
        raise ValueError('Invalid evaluation thresholds.')
    improvement = 100 * (baseline_mae - candidate_mae) / baseline_mae
    checks = {
        'mae_improvement_ge_pilot_threshold': improvement >= min_improvement_pct,
        'candidate_rmse_lower_than_raw': candidate_rmse < baseline_rmse,
        'abs_candidate_bias_le_threshold': abs(candidate_bias) <= max_abs_bias,
    }
    return {
        'passed': all(checks.values()),
        'checks': checks,
        'min_improvement_pct': min_improvement_pct,
        'max_abs_bias_w_m2': max_abs_bias,
        'relative_improvement_pct': improvement,
        'project_target_10pct_met': improvement >= 10.0,
        'target_note': '10% is the LK-01 project success KPI; 5% is a provisional LK-08 demo gate, not production acceptance.',
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-improvement', type=float, default=5.0)
    parser.add_argument('--max-abs-bias', type=float, default=20.0)
    args = parser.parse_args()
    meta = json.loads(METADATA.read_text(encoding='utf-8'))
    df = pd.read_parquet(TRAINING)
    df['run_time_utc'] = pd.to_datetime(df['run_time_utc'], utc=True)
    df = df.loc[df['is_day'] == 1].copy()
    runs = sorted(df['run_time_utc'].unique())
    n_val_runs = max(1, ceil(len(runs) * 0.2))
    valid = df.loc[df['run_time_utc'].isin(runs[-n_val_runs:])].copy()
    if len(valid) != int(meta['validation_rows']):
        raise ValueError('Validation count mismatch between training report and dataset.')
    baseline_rmse = float(math.sqrt(mean_squared_error(valid['actual_ghi'], valid['raw_ghi_forecast'])))
    result = compute_gate(
        candidate_mae=float(meta['best_validation_mae']),
        candidate_rmse=float(meta['best_validation_rmse']),
        candidate_bias=float(meta['best_validation_bias']),
        baseline_mae=float(meta['raw_validation_mae']),
        baseline_rmse=baseline_rmse,
        min_improvement_pct=args.min_improvement,
        max_abs_bias=args.max_abs_bias,
    )
    result.update({
        'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
        'candidate_run': meta['selected_run'],
        'candidate_run_id': meta['run_id'],
        'candidate_model_uri': meta['model_uri'],
        'candidate_mae_w_m2': meta['best_validation_mae'],
        'candidate_rmse_w_m2': meta['best_validation_rmse'],
        'candidate_bias_w_m2': meta['best_validation_bias'],
        'raw_mae_w_m2': meta['raw_validation_mae'],
        'raw_rmse_w_m2': baseline_rmse,
        'train_rows': meta['train_rows'],
        'validation_rows': meta['validation_rows'],
        'dvc_forecast_md5': meta['dvc_forecast_md5'],
        'scope': 'LK08 CI demonstration on small historical LK05 snapshot; not production KPI verification',
    })
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('=== LK08 QUALITY GATE ===', flush=True)
    print(json.dumps(result, indent=2), flush=True)
    if not result['passed']:
        raise SystemExit('LK08 QUALITY GATE: FAIL. Registry step must not run.')
    print('LK08 QUALITY GATE: PASS', flush=True)


if __name__ == '__main__':
    main()
