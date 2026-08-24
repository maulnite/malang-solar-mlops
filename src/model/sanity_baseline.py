import os
from math import sqrt

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = (
    "data/raw/"
    "openmeteo_malang_solar_feasibility.csv"
)

TRAIN_END = "2025-12-31 23:00:00"
TEST_START = "2026-01-01 00:00:00"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan:\n{DATA_PATH}\n\n"
            "Jalankan test_openmeteo_solar.py terlebih dahulu."
        )

    print(
        "=== SOLAR FORECAST CORRECTION "
        "SANITY TEST ===\n"
    )

    df = pd.read_csv(
        DATA_PATH
    )

    df["time"] = pd.to_datetime(
        df["time"]
    )

    df = (
        df
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    print(
        f"Total rows : {len(df)}"
    )

    print(
        f"Start      : {df['time'].min()}"
    )

    print(
        f"End        : {df['time'].max()}"
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):
    df = df.copy()

    # --------------------------------------------------------
    # TIME FEATURES
    # --------------------------------------------------------

    df["hour"] = (
        df["time"].dt.hour
    )

    df["day_of_year"] = (
        df["time"].dt.dayofyear
    )

    # Cyclical hour encoding
    df["hour_sin"] = np.sin(
        2 * np.pi
        * df["hour"]
        / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi
        * df["hour"]
        / 24
    )

    # Cyclical yearly season encoding
    df["doy_sin"] = np.sin(
        2 * np.pi
        * df["day_of_year"]
        / 365.25
    )

    df["doy_cos"] = np.cos(
        2 * np.pi
        * df["day_of_year"]
        / 365.25
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    # Yang ingin dipelajari model bukan GHI dari nol,
    # tetapi ERROR / RESIDUAL forecast.
    #
    # residual =
    # reference ERA5 - raw forecast
    #
    # corrected forecast =
    # raw forecast + predicted residual

    df["forecast_error"] = (
        df["actual_ghi"]
        - df[
            "forecast_shortwave_radiation"
        ]
    )

    return df


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "forecast_temperature_2m",
    "forecast_relative_humidity_2m",
    "forecast_precipitation",
    "forecast_cloud_cover",
    "forecast_pressure_msl",
    "forecast_wind_speed_10m",
    "forecast_wind_direction_10m",
    "forecast_shortwave_radiation",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def temporal_split(df):
    train_end = pd.Timestamp(
        TRAIN_END
    )

    test_start = pd.Timestamp(
        TEST_START
    )

    train_df = df[
        df["time"] <= train_end
    ].copy()

    test_df = df[
        df["time"] >= test_start
    ].copy()

    # Kita fokus pada daylight.
    #
    # Malam terlalu mudah karena GHI ~= 0
    # dan bisa membuat metric terlihat bagus
    # secara palsu.
    train_df = train_df[
        train_df["forecast_is_day"] == 1
    ].copy()

    test_df = test_df[
        test_df["forecast_is_day"] == 1
    ].copy()

    print(
        "\n=== TEMPORAL SPLIT ==="
    )

    print(
        f"Train period : "
        f"{train_df['time'].min()} "
        f"→ {train_df['time'].max()}"
    )

    print(
        f"Train rows   : "
        f"{len(train_df)}"
    )

    print()

    print(
        f"Test period  : "
        f"{test_df['time'].min()} "
        f"→ {test_df['time'].max()}"
    )

    print(
        f"Test rows    : "
        f"{len(test_df)}"
    )

    return train_df, test_df


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    prediction,
):
    mae = mean_absolute_error(
        actual,
        prediction
    )

    rmse = sqrt(
        mean_squared_error(
            actual,
            prediction
        )
    )

    error = (
        prediction - actual
    )

    bias = np.mean(
        error
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
    }


def evaluate_prediction(
    name,
    actual,
    prediction,
    baseline_mae,
):
    metrics = calculate_metrics(
        actual,
        prediction
    )

    improvement = (
        (
            baseline_mae
            - metrics["MAE"]
        )
        / baseline_mae
        * 100
    )

    print(
        f"\n{name}"
    )

    print(
        "-" * 60
    )

    print(
        f"MAE         : "
        f"{metrics['MAE']:.2f} W/m²"
    )

    print(
        f"RMSE        : "
        f"{metrics['RMSE']:.2f} W/m²"
    )

    print(
        f"Bias        : "
        f"{metrics['Bias']:.2f} W/m²"
    )

    print(
        f"Improvement : "
        f"{improvement:+.2f}% "
        f"vs raw forecast"
    )

    return {
        "model": name,
        **metrics,
        "improvement_percent":
            improvement,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_data()

    df = create_features(
        df
    )

    # --------------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = (
        FEATURE_COLUMNS
        + [
            "forecast_error",
            "actual_ghi",
            "forecast_is_day",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Kolom berikut tidak ditemukan:\n"
            + "\n".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # DROP INVALID VALUES
    # --------------------------------------------------------

    df = df.dropna(
        subset=required_columns
    ).copy()

    # --------------------------------------------------------
    # TEMPORAL SPLIT
    # --------------------------------------------------------

    train_df, test_df = (
        temporal_split(df)
    )

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        "forecast_error"
    ]

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    actual_test = test_df[
        "actual_ghi"
    ].to_numpy()

    raw_forecast_test = test_df[
        "forecast_shortwave_radiation"
    ].to_numpy()

    # --------------------------------------------------------
    # RAW OPEN-METEO BASELINE
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BASELINE: RAW OPEN-METEO FORECAST"
    )

    print(
        "=" * 70
    )

    baseline_metrics = (
        calculate_metrics(
            actual_test,
            raw_forecast_test,
        )
    )

    baseline_mae = (
        baseline_metrics["MAE"]
    )

    print(
        f"MAE  : "
        f"{baseline_metrics['MAE']:.2f} W/m²"
    )

    print(
        f"RMSE : "
        f"{baseline_metrics['RMSE']:.2f} W/m²"
    )

    print(
        f"Bias : "
        f"{baseline_metrics['Bias']:.2f} W/m²"
    )

    results = [
        {
            "model":
                "Raw Open-Meteo Forecast",
            **baseline_metrics,
            "improvement_percent": 0.0,
        }
    ]

    # ========================================================
    # MODEL 1: LINEAR RESIDUAL CORRECTION
    # ========================================================

    linear_model = (
        LinearRegression()
    )

    linear_model.fit(
        X_train,
        y_train,
    )

    linear_residual = (
        linear_model.predict(
            X_test
        )
    )

    linear_corrected = (
        raw_forecast_test
        + linear_residual
    )

    # Irradiance secara fisik tidak boleh negatif.
    linear_corrected = np.clip(
        linear_corrected,
        a_min=0,
        a_max=None,
    )

    results.append(
        evaluate_prediction(
            name=(
                "Linear Regression "
                "Residual Correction"
            ),
            actual=actual_test,
            prediction=linear_corrected,
            baseline_mae=baseline_mae,
        )
    )

    # ========================================================
    # MODEL 2: RANDOM FOREST
    # ========================================================

    rf_model = (
        RandomForestRegressor(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )

    print(
        "\nTraining Random Forest..."
    )

    rf_model.fit(
        X_train,
        y_train,
    )

    rf_residual = (
        rf_model.predict(
            X_test
        )
    )

    rf_corrected = (
        raw_forecast_test
        + rf_residual
    )

    rf_corrected = np.clip(
        rf_corrected,
        a_min=0,
        a_max=None,
    )

    results.append(
        evaluate_prediction(
            name=(
                "Random Forest "
                "Residual Correction"
            ),
            actual=actual_test,
            prediction=rf_corrected,
            baseline_mae=baseline_mae,
        )
    )

    # ========================================================
    # MODEL 3: HIST GRADIENT BOOSTING
    # ========================================================

    hgb_model = (
        HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
    )

    print(
        "\nTraining HistGradientBoosting..."
    )

    hgb_model.fit(
        X_train,
        y_train,
    )

    hgb_residual = (
        hgb_model.predict(
            X_test
        )
    )

    hgb_corrected = (
        raw_forecast_test
        + hgb_residual
    )

    hgb_corrected = np.clip(
        hgb_corrected,
        a_min=0,
        a_max=None,
    )

    results.append(
        evaluate_prediction(
            name=(
                "HistGradientBoosting "
                "Residual Correction"
            ),
            actual=actual_test,
            prediction=hgb_corrected,
            baseline_mae=baseline_mae,
        )
    )

    # ========================================================
    # RANKING
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    result_df = (
        result_df
        .sort_values(
            "MAE",
            ascending=True
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODEL RANKING"
    )

    print(
        "=" * 70
    )

    print(
        result_df.to_string(
            index=False,
            formatters={
                "MAE":
                    lambda x:
                    f"{x:.2f}",
                "RMSE":
                    lambda x:
                    f"{x:.2f}",
                "Bias":
                    lambda x:
                    f"{x:.2f}",
                "improvement_percent":
                    lambda x:
                    f"{x:+.2f}%",
            }
        )
    )

    # ========================================================
    # FINAL VERDICT
    # ========================================================

    best_model = (
        result_df.iloc[0]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SANITY TEST VERDICT"
    )

    print(
        "=" * 70
    )

    print(
        f"Best model : "
        f"{best_model['model']}"
    )

    print(
        f"Best MAE   : "
        f"{best_model['MAE']:.2f} W/m²"
    )

    improvement = (
        best_model[
            "improvement_percent"
        ]
    )

    print(
        f"Improvement: "
        f"{improvement:.2f}%"
    )

    if improvement >= 10:
        print(
            "\n✅ STRONG SIGNAL"
        )

        print(
            "ML correction memberikan "
            "peningkatan yang meaningful."
        )

        print(
            "Problem formulation layak "
            "digunakan untuk proyek."
        )

    elif improvement > 0:
        print(
            "\n🟡 WEAK/MODERATE SIGNAL"
        )

        print(
            "ML berhasil memperbaiki forecast, "
            "tetapi improvement masih kecil."
        )

    else:
        print(
            "\n❌ NO CORRECTION SIGNAL"
        )

        print(
            "Model belum dapat mengalahkan "
            "raw forecast."
        )

    # ========================================================
    # SAVE RESULT
    # ========================================================

    output_path = (
        "data/raw/"
        "solar_sanity_model_results.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nResult tersimpan:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()