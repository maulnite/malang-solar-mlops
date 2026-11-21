"""Standalone feature sanity tests; no Docker/MLflow server needed."""
import math
from datetime import datetime, timezone

import pytest

from src.serving.lk09_features import make_feature_row


def sample():
    return {
        'valid_time_utc': '2026-10-09T07:00:00Z',
        'lead_time_hour': 1, 'temperature_2m': 25,
        'relative_humidity_2m': 70, 'precipitation': 0,
        'cloud_cover': 40, 'pressure_msl': 1010,
        'wind_speed_10m': 6, 'wind_direction_10m': 90,
        'raw_ghi_forecast': 660,
    }


def test_features_match_wib_and_wind_cycle():
    row = make_feature_row(sample())
    assert math.isclose(row['hour_sin'], -0.5, abs_tol=1e-12)  # 14 WIB
    assert math.isclose(row['wind_dir_sin'], 1.0, abs_tol=1e-12)
    assert math.isclose(row['wind_dir_cos'], 0.0, abs_tol=1e-12)
    assert len(row) == 15  # 9 raw inputs + six derived time/wind features


def test_reject_naive_datetime():
    data = sample(); data['valid_time_utc'] = '2026-10-09T07:00:00'
    with pytest.raises(ValueError, match='explicit timezone'):
        make_feature_row(data)


def test_reject_nan_feature():
    data = sample(); data['raw_ghi_forecast'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        make_feature_row(data)
