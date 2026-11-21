"""Run from repo root: python tools/lk09_smoke_test.py (stdlib only)."""
from __future__ import annotations

import csv
import json
import math
import os
import urllib.request
from pathlib import Path

URL = os.environ.get('LK09_API_URL', 'http://127.0.0.1:8000').rstrip('/')


def get(path):
    with urllib.request.urlopen(URL + path, timeout=20) as response:
        return json.load(response)


def post(path, payload):
    request = urllib.request.Request(URL + path,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def main():
    fixture = Path('data/fixtures/lk08_ecmwf_history_v2.csv')
    with fixture.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    row = next(item for item in rows if float(item['raw_ghi_forecast']) > 0)
    numeric = (
        'lead_time_hour', 'temperature_2m', 'relative_humidity_2m',
        'precipitation', 'cloud_cover', 'pressure_msl', 'wind_speed_10m',
        'wind_direction_10m', 'raw_ghi_forecast'
    )
    payload = {'valid_time_utc': row['valid_time_utc']}
    for key in numeric:
        payload[key] = int(row[key]) if key == 'lead_time_hour' else float(row[key])
    health, info = get('/health'), get('/model-info')
    result = post('/predict', payload)
    assert health['status'] == 'ok' and health['model_loaded']
    assert info['tracking_uri'] == 'http://mlflow:5000'
    assert info['alias'] == 'staging' and int(info['version']) >= 1
    assert info['source_dvc_md5'] == '5ed231ffe84c086adc1ee80957693395'
    assert math.isfinite(result['predicted_residual_w_m2'])
    assert abs(result['corrected_ghi_w_m2'] - max(0, result['raw_ghi_forecast_w_m2'] + result['predicted_residual_w_m2'])) < 1e-9
    print('=== LK09 HEALTH ===')
    print(json.dumps(health, indent=2))
    print('=== LK09 CROSS-CONTAINER MLFLOW LOOKUP ===')
    print(json.dumps(info, indent=2))
    print('=== LK09 INFERENCE ===')
    print(json.dumps({'input':payload,'output':result}, indent=2))
    print('LK09 END-TO-END SMOKE TEST: PASS')


if __name__ == '__main__':
    main()
