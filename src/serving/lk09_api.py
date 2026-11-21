"""LK-09 API: resolve a registered MLflow Staging model through Compose DNS."""
from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from datetime import datetime

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from src.model.train import FEATURE_COLUMNS
from src.serving.lk09_features import make_feature_row

TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MODEL_NAME = os.getenv('MLOPS_REGISTERED_MODEL_NAME', 'MalangSolarResidualCorrectorLK08')
MODEL_URI = f'models:/{MODEL_NAME}@staging'


class ForecastRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra='forbid')
    valid_time_utc: datetime
    lead_time_hour: int = Field(ge=1, le=6)
    temperature_2m: float
    relative_humidity_2m: float = Field(ge=0, le=100)
    precipitation: float = Field(ge=0)
    cloud_cover: float = Field(ge=0, le=100)
    pressure_msl: float = Field(gt=0)
    wind_speed_10m: float = Field(ge=0)
    wind_direction_10m: float = Field(ge=0, le=360)
    raw_ghi_forecast: float = Field(ge=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    version = client.get_model_version_by_alias(MODEL_NAME, 'staging')
    model = mlflow.pyfunc.load_model(MODEL_URI)
    app.state.model = model
    app.state.version = str(version.version)
    app.state.dvc_md5 = version.tags.get('source_dvc_md5', '')
    print(f'LK09 API READY: {MODEL_URI} -> v{version.version}; DVC={app.state.dvc_md5}', flush=True)
    yield


app = FastAPI(
    title='LK-09 Malang Solar Residual Correction API',
    description='Academic demonstration. ERA5 reference, 12 historic forecast rows. Not production-certified.',
    version='0.1.0', lifespan=lifespan,
)


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': True, 'model_uri': MODEL_URI}


@app.get('/model-info')
def model_info():
    # Explicitly query the registry over http://mlflow:5000; this proves
    # service-to-service networking rather than just reusing local files.
    client = MlflowClient(tracking_uri=TRACKING_URI)
    version = client.get_model_version_by_alias(MODEL_NAME, 'staging')
    return {
        'tracking_uri': TRACKING_URI,
        'model_name': MODEL_NAME,
        'alias': 'staging',
        'version': str(version.version),
        'source_dvc_md5': version.tags.get('source_dvc_md5', ''),
        'loaded_model_uri': MODEL_URI,
        'note': 'LK09 demo candidate, not actual production model',
    }


@app.post('/predict')
def predict(payload: ForecastRequest):
    try:
        features = make_feature_row(payload.model_dump())
        X = pd.DataFrame([{key: features[key] for key in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)
        prediction = app.state.model.predict(X)
        residual = float(prediction[0])
        if not math.isfinite(residual):
            raise ValueError('Model returned a non-finite residual')
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raw = float(payload.raw_ghi_forecast)
    return {
        'model_uri': MODEL_URI,
        'registered_version': app.state.version,
        'raw_ghi_forecast_w_m2': raw,
        'predicted_residual_w_m2': residual,
        'corrected_ghi_w_m2': max(0.0, raw + residual),
        'scope': 'LK09 academic service demo; do not use as operational solar forecast',
    }
