import os
from datetime import date, timedelta

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

# Koordinat pusat Kota Malang
LATITUDE = -7.9666
LONGITUDE = 112.6326

TIMEZONE = "Asia/Jakarta"

# Historical Forecast tersedia sejak sekitar 2021/2022.
# Untuk feasibility kita gunakan data yang cukup panjang
# tetapi masih relevan untuk kondisi modern.
START_DATE = "2023-01-01"

# ERA5 memiliki delay beberapa hari.
# Supaya aman, gunakan data sampai 7 hari sebelum hari ini.
END_DATE = (
    date.today() - timedelta(days=7)
).isoformat()


HISTORICAL_FORECAST_URL = (
    "https://historical-forecast-api.open-meteo.com/v1/forecast"
)

HISTORICAL_WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)


OUTPUT_FORECAST = (
    "data/raw/openmeteo_malang_historical_forecast.csv"
)

OUTPUT_ACTUAL = (
    "data/raw/openmeteo_malang_era5_actual.csv"
)

OUTPUT_MERGED = (
    "data/raw/openmeteo_malang_solar_feasibility.csv"
)


# ============================================================
# VARIABLES
# ============================================================

FORECAST_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "is_day",
]


ACTUAL_VARIABLES = [
    "shortwave_radiation",
]


# ============================================================
# HTTP
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "User-Agent": "malang-solar-mlops-feasibility-test"
    }
)


def request_json(url, params):
    """
    Generic API request helper.
    """

    try:
        response = session.get(
            url,
            params=params,
            timeout=120,
        )

    except requests.RequestException as error:
        raise RuntimeError(
            f"Request gagal: {error}"
        ) from None

    if not response.ok:
        print("\nAPI ERROR:")
        print(response.text)

        raise RuntimeError(
            f"HTTP {response.status_code}"
        )

    try:
        return response.json()

    except ValueError:
        raise RuntimeError(
            "Response API bukan JSON valid."
        ) from None


# ============================================================
# OPEN-METEO RESPONSE PARSER
# ============================================================

def hourly_payload_to_dataframe(payload):
    """
    Mengubah response Open-Meteo menjadi DataFrame.
    """

    hourly = payload.get("hourly")

    if not hourly:
        raise RuntimeError(
            "Field 'hourly' tidak ditemukan."
        )

    df = pd.DataFrame(hourly)

    if "time" not in df.columns:
        raise RuntimeError(
            "Field 'time' tidak ditemukan."
        )

    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce",
    )

    df = (
        df
        .dropna(subset=["time"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# FETCH HISTORICAL FORECAST
# ============================================================

def fetch_historical_forecast():
    print(
        "\n[1] Mengambil Historical Forecast API..."
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(
            FORECAST_VARIABLES
        ),
        "timezone": TIMEZONE,
    }

    payload = request_json(
        HISTORICAL_FORECAST_URL,
        params,
    )

    print(
        f"Latitude API  : {payload.get('latitude')}"
    )

    print(
        f"Longitude API : {payload.get('longitude')}"
    )

    print(
        f"Elevation     : {payload.get('elevation')} m"
    )

    print(
        f"Timezone      : {payload.get('timezone')}"
    )

    df = hourly_payload_to_dataframe(
        payload
    )

    rename_map = {}

    for column in df.columns:
        if column == "time":
            continue

        rename_map[column] = (
            f"forecast_{column}"
        )

    df = df.rename(
        columns=rename_map
    )

    print(
        f"Forecast rows : {len(df)}"
    )

    return df


# ============================================================
# FETCH ERA5 REANALYSIS TARGET
# ============================================================

def fetch_historical_actual():
    print(
        "\n[2] Mengambil ERA5 Historical Weather..."
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(
            ACTUAL_VARIABLES
        ),
        "timezone": TIMEZONE,

        # Stable reanalysis target
        "models": "era5",
    }

    payload = request_json(
        HISTORICAL_WEATHER_URL,
        params,
    )

    print(
        f"Latitude API  : {payload.get('latitude')}"
    )

    print(
        f"Longitude API : {payload.get('longitude')}"
    )

    print(
        f"Elevation     : {payload.get('elevation')} m"
    )

    df = hourly_payload_to_dataframe(
        payload
    )

    df = df.rename(
        columns={
            "shortwave_radiation":
                "actual_ghi"
        }
    )

    print(
        f"ERA5 rows     : {len(df)}"
    )

    return df


# ============================================================
# DATA QUALITY ANALYSIS
# ============================================================

def audit_dataframe(
    df,
    name,
):
    print(
        "\n"
        + "=" * 70
    )

    print(
        f"DATA QUALITY: {name}"
    )

    print(
        "=" * 70
    )

    print(
        f"Rows       : {len(df)}"
    )

    print(
        f"Columns    : {len(df.columns)}"
    )

    print(
        f"Start      : {df['time'].min()}"
    )

    print(
        f"End        : {df['time'].max()}"
    )

    duplicate_count = (
        df["time"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicates : {duplicate_count}"
    )

    expected_index = pd.date_range(
        start=df["time"].min(),
        end=df["time"].max(),
        freq="h",
    )

    actual_index = pd.DatetimeIndex(
        df["time"]
    )

    available_hours = len(
        actual_index.intersection(
            expected_index
        )
    )

    expected_hours = len(
        expected_index
    )

    missing_hours = (
        expected_hours
        - available_hours
    )

    completeness = (
        available_hours
        / expected_hours
        * 100
    )

    print(
        f"Expected hours : {expected_hours}"
    )

    print(
        f"Available      : {available_hours}"
    )

    print(
        f"Missing hours  : {missing_hours}"
    )

    print(
        f"Completeness   : {completeness:.4f}%"
    )

    print(
        "\nMissing values per column:"
    )

    missing = (
        df
        .isna()
        .sum()
    )

    for column, count in missing.items():
        percentage = (
            count / len(df) * 100
        )

        print(
            f"{column:<40} "
            f"{count:>6} "
            f"({percentage:.4f}%)"
        )


# ============================================================
# MERGE
# ============================================================

def merge_datasets(
    forecast_df,
    actual_df,
):
    print(
        "\n[3] Merge forecast + actual..."
    )

    df = forecast_df.merge(
        actual_df,
        on="time",
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Merged rows : {len(df)}"
    )

    return df


# ============================================================
# SOLAR ANALYSIS
# ============================================================

def analyze_solar(df):
    print(
        "\n"
        + "=" * 70
    )

    print(
        "SOLAR IRRADIANCE ANALYSIS"
    )

    print(
        "=" * 70
    )

    required = [
        "forecast_shortwave_radiation",
        "actual_ghi",
    ]

    for column in required:
        if column not in df.columns:
            raise RuntimeError(
                f"Kolom {column} tidak ditemukan."
            )

    forecast_ghi = pd.to_numeric(
        df["forecast_shortwave_radiation"],
        errors="coerce",
    )

    actual_ghi = pd.to_numeric(
        df["actual_ghi"],
        errors="coerce",
    )

    print(
        "\nForecast GHI:"
    )

    print(
        f"Min    : {forecast_ghi.min():.2f}"
    )

    print(
        f"Median : {forecast_ghi.median():.2f}"
    )

    print(
        f"Mean   : {forecast_ghi.mean():.2f}"
    )

    print(
        f"Max    : {forecast_ghi.max():.2f}"
    )

    print(
        "\nActual ERA5 GHI:"
    )

    print(
        f"Min    : {actual_ghi.min():.2f}"
    )

    print(
        f"Median : {actual_ghi.median():.2f}"
    )

    print(
        f"Mean   : {actual_ghi.mean():.2f}"
    )

    print(
        f"Max    : {actual_ghi.max():.2f}"
    )

    # --------------------------------------------------------
    # DAY / NIGHT
    # --------------------------------------------------------

    if "forecast_is_day" in df.columns:
        day_df = df[
            df["forecast_is_day"] == 1
        ]

        night_df = df[
            df["forecast_is_day"] == 0
        ]

        print(
            "\nDay/Night:"
        )

        print(
            f"Day rows   : {len(day_df)}"
        )

        print(
            f"Night rows : {len(night_df)}"
        )

    # --------------------------------------------------------
    # BASIC FORECAST ERROR
    # --------------------------------------------------------

    valid = df[
        [
            "forecast_shortwave_radiation",
            "actual_ghi",
        ]
    ].dropna()

    valid["error"] = (
        valid[
            "forecast_shortwave_radiation"
        ]
        - valid["actual_ghi"]
    )

    valid["absolute_error"] = (
        valid["error"]
        .abs()
    )

    mae = (
        valid[
            "absolute_error"
        ]
        .mean()
    )

    bias = (
        valid["error"]
        .mean()
    )

    print(
        "\nRaw Open-Meteo forecast vs ERA5:"
    )

    print(
        f"MAE  : {mae:.2f} W/m²"
    )

    print(
        f"Bias : {bias:.2f} W/m²"
    )

    # Daylight-only error lebih bermakna
    if "forecast_is_day" in df.columns:
        daylight = df[
            df["forecast_is_day"] == 1
        ][
            [
                "forecast_shortwave_radiation",
                "actual_ghi",
            ]
        ].dropna()

        daylight["absolute_error"] = (
            daylight[
                "forecast_shortwave_radiation"
            ]
            .sub(
                daylight["actual_ghi"]
            )
            .abs()
        )

        daylight_mae = (
            daylight[
                "absolute_error"
            ]
            .mean()
        )

        print(
            f"Daylight MAE : "
            f"{daylight_mae:.2f} W/m²"
        )


# ============================================================
# SAVE
# ============================================================

def save_outputs(
    forecast_df,
    actual_df,
    merged_df,
):
    os.makedirs(
        "data/raw",
        exist_ok=True,
    )

    forecast_df.to_csv(
        OUTPUT_FORECAST,
        index=False,
    )

    actual_df.to_csv(
        OUTPUT_ACTUAL,
        index=False,
    )

    merged_df.to_csv(
        OUTPUT_MERGED,
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 70
    )

    print(
        OUTPUT_FORECAST
    )

    print(
        OUTPUT_ACTUAL
    )

    print(
        OUTPUT_MERGED
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "=== MALANG SOLAR MLOPS FEASIBILITY TEST ==="
    )

    print(
        f"\nCoordinate : "
        f"{LATITUDE}, {LONGITUDE}"
    )

    print(
        f"Timezone   : {TIMEZONE}"
    )

    print(
        f"Period     : "
        f"{START_DATE} sampai {END_DATE}"
    )

    forecast_df = (
        fetch_historical_forecast()
    )

    actual_df = (
        fetch_historical_actual()
    )

    audit_dataframe(
        forecast_df,
        "Historical Forecast",
    )

    audit_dataframe(
        actual_df,
        "ERA5 Actual / Reanalysis",
    )

    merged_df = (
        merge_datasets(
            forecast_df,
            actual_df,
        )
    )

    audit_dataframe(
        merged_df,
        "Merged Dataset",
    )

    analyze_solar(
        merged_df
    )

    save_outputs(
        forecast_df,
        actual_df,
        merged_df,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FEASIBILITY TEST SELESAI"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()