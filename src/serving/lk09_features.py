"""Feature engineering compatible with LK-04 preprocess.py (WIB time zone)."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

INPUTS = (
    'temperature_2m', 'relative_humidity_2m', 'precipitation',
    'cloud_cover', 'pressure_msl', 'wind_speed_10m',
    'wind_direction_10m', 'raw_ghi_forecast', 'lead_time_hour',
)


def make_feature_row(data: dict) -> dict[str, float]:
    time = data['valid_time_utc']
    if isinstance(time, str):
        time = datetime.fromisoformat(time.replace('Z', '+00:00'))
    if not isinstance(time, datetime) or time.tzinfo is None or time.utcoffset() is None:
        raise ValueError('valid_time_utc must have an explicit timezone, e.g. 2026-10-09T07:00:00Z')
    local = time.astimezone(ZoneInfo('Asia/Jakarta'))
    hour = local.hour + local.minute / 60
    day = local.timetuple().tm_yday
    wind_rad = math.radians(float(data['wind_direction_10m']))
    row = {key: float(data[key]) for key in INPUTS}
    if not all(math.isfinite(value) for value in row.values()):
        raise ValueError('Features must be finite')
    row.update({
        'hour_sin': math.sin(2 * math.pi * hour / 24),
        'hour_cos': math.cos(2 * math.pi * hour / 24),
        'day_of_year_sin': math.sin(2 * math.pi * day / 365.25),
        'day_of_year_cos': math.cos(2 * math.pi * day / 365.25),
        'wind_dir_sin': math.sin(wind_rad),
        'wind_dir_cos': math.cos(wind_rad),
    })
    return row
