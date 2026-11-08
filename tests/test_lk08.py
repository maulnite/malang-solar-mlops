"""Tests run without API access, DVC cache, model artifacts, or MLflow server."""
from pathlib import Path

import pandas as pd
import pytest

from src.model.lk08_evaluate import compute_gate


def test_lk04_sample_forecast_integrity():
    path = Path('data/raw/sample/ecmwf_ifs_sample.csv')
    df = pd.read_csv(path)
    assert len(df) == 6
    required = {'source_model', 'run_time_utc', 'valid_time_utc', 'lead_time_hour',
                'raw_ghi_forecast', 'temperature_2m', 'cloud_cover', 'is_day'}
    assert required.issubset(df.columns)
    assert not df.duplicated(['source_model', 'run_time_utc', 'valid_time_utc']).any()
    assert df['lead_time_hour'].tolist() == [1, 2, 3, 4, 5, 6]
    assert (df['raw_ghi_forecast'] >= 0).all()


def test_gate_passes_small_lk06_demo():
    result = compute_gate(69.75, 86.699, -1.51, 73.5, 95.0)
    assert result['passed'] is True
    assert result['project_target_10pct_met'] is False


def test_gate_blocks_regression():
    result = compute_gate(79.0, 99.0, 25.0, 73.5, 95.0)
    assert result['passed'] is False
    assert result['checks']['mae_improvement_ge_pilot_threshold'] is False


def test_gate_rejects_nan():
    with pytest.raises(ValueError, match='Non-finite'):
        compute_gate(float('nan'), 86, 0, 73, 100)


def test_gate_can_enforce_stricter_ten_percent_goal():
    result = compute_gate(69.75, 86.699, -1.51, 73.5, 95.0, min_improvement_pct=10.0)
    assert result['passed'] is False


def test_frozen_dvc_v2_matches_pointer():
    from src.data.lk08_prepare import compare_dvc_fixture
    digest = compare_dvc_fixture(
        Path('data/fixtures/lk08_ecmwf_history_v2.csv'),
        Path('data/raw/ecmwf_ifs_history.csv.dvc'),
    )
    assert digest == '5ed231ffe84c086adc1ee80957693395'


def test_dvc_integrity_blocks_modified_fixture(tmp_path):
    from src.data.lk08_prepare import compare_dvc_fixture
    fixture = tmp_path / 'history.csv'
    pointer = tmp_path / 'history.csv.dvc'
    fixture.write_bytes(b'modified-history')
    pointer.write_text('outs:\n- md5: 5ed231ffe84c086adc1ee80957693395\n')
    with pytest.raises(ValueError, match='Fixture integrity failure'):
        compare_dvc_fixture(fixture, pointer)
