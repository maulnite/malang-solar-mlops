from zoneinfo import ZoneInfo

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

LATITUDE = -7.9666
LONGITUDE = 112.6326

# Single Runs API menggunakan waktu inisialisasi dalam UTC.
# Kita sengaja pakai archived run agar pasti sudah tersedia.
RUN = "2026-08-24T00:00"

MODEL = "ecmwf_ifs"

URL = (
    "https://single-runs-api.open-meteo.com/v1/forecast"
)

HOURLY_VARIABLES = [
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


# ============================================================
# FETCH
# ============================================================

def fetch_single_run():
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        # Model yang kita lock
        "models": MODEL,

        # Exact model initialization time
        "run": RUN,

        "hourly": ",".join(
            HOURLY_VARIABLES
        ),

        # Ambil +0h sampai sekitar +6h.
        # Nanti +0h kita buang.
        "forecast_hours": 7,

        # Sengaja UTC agar lead time mudah dihitung.
        "timezone": "GMT",
    }

    print(
        "=== ECMWF IFS HRES SINGLE RUN TEST ===\n"
    )

    print(
        f"Requested coordinate : "
        f"{LATITUDE}, {LONGITUDE}"
    )

    print(
        f"Requested model      : {MODEL}"
    )

    print(
        f"Requested run        : {RUN} UTC"
    )

    print(
        "\nFetching Single Runs API..."
    )

    try:
        response = requests.get(
            URL,
            params=params,
            timeout=120,
        )

    except requests.RequestException as error:
        raise RuntimeError(
            f"Request gagal: {error}"
        ) from None

    print(
        f"HTTP status          : "
        f"{response.status_code}"
    )

    if not response.ok:
        print(
            "\nAPI RESPONSE:"
        )

        print(
            response.text
        )

        raise RuntimeError(
            "Single Runs API request gagal."
        )

    return response.json()


# ============================================================
# PARSE
# ============================================================

def parse_response(payload):
    print(
        "\n=== RESOLVED LOCATION ==="
    )

    print(
        f"Latitude  : "
        f"{payload.get('latitude')}"
    )

    print(
        f"Longitude : "
        f"{payload.get('longitude')}"
    )

    print(
        f"Elevation : "
        f"{payload.get('elevation')} m"
    )

    print(
        f"Timezone  : "
        f"{payload.get('timezone')}"
    )

    hourly = payload.get(
        "hourly"
    )

    if not hourly:
        raise RuntimeError(
            "Field 'hourly' tidak ditemukan."
        )

    df = pd.DataFrame(
        hourly
    )

    if df.empty:
        raise RuntimeError(
            "API berhasil, tetapi data hourly kosong."
        )

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
    )

    run_time = pd.Timestamp(
        RUN,
        tz="UTC",
    )

    # Hitung lead time dari exact model run.
    df["lead_time_hour"] = (
        (
            df["time"] - run_time
        )
        .dt.total_seconds()
        / 3600
    )

    df["lead_time_hour"] = (
        df["lead_time_hour"]
        .astype(int)
    )

    # Kita hanya membutuhkan +1 sampai +6 jam.
    df = df[
        df["lead_time_hour"]
        .between(1, 6)
    ].copy()

    if df.empty:
        raise RuntimeError(
            "Tidak ditemukan forecast "
            "lead +1 sampai +6 jam."
        )

    # Rename agar maknanya eksplisit.
    df = df.rename(
        columns={
            "time": "valid_time_utc",
            "shortwave_radiation":
                "raw_ghi_forecast",
        }
    )

    # --------------------------------------------------------
    # WIB VERSION
    # --------------------------------------------------------

    jakarta_tz = ZoneInfo(
        "Asia/Jakarta"
    )

    df["valid_time_wib"] = (
        df["valid_time_utc"]
        .dt.tz_convert(
            jakarta_tz
        )
    )

    run_wib = (
        run_time
        .tz_convert(
            jakarta_tz
        )
    )

    df.insert(
        0,
        "run_time_utc",
        run_time,
    )

    df.insert(
        1,
        "run_time_wib",
        run_wib,
    )

    return df


# ============================================================
# VALIDATE
# ============================================================

def validate(df):
    print(
        "\n=== VALIDATION ==="
    )

    expected_leads = {
        1, 2, 3, 4, 5, 6
    }

    actual_leads = set(
        df["lead_time_hour"]
        .tolist()
    )

    print(
        f"Expected lead times : "
        f"{sorted(expected_leads)}"
    )

    print(
        f"Received lead times : "
        f"{sorted(actual_leads)}"
    )

    missing_leads = (
        expected_leads
        - actual_leads
    )

    print(
        f"Missing lead times  : "
        f"{sorted(missing_leads)}"
    )

    missing_values = (
        df
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing values      : "
        f"{missing_values}"
    )

    duplicates = (
        df["valid_time_utc"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate valid time: "
        f"{duplicates}"
    )

    if not missing_leads:
        print(
            "\n✅ Lead +1 sampai +6 "
            "tersedia lengkap."
        )

    if (
        missing_values == 0
        and duplicates == 0
    ):
        print(
            "✅ Data valid untuk "
            "proof-of-concept."
        )


# ============================================================
# DISPLAY
# ============================================================

def display_result(df):
    columns = [
        "run_time_utc",
        "valid_time_utc",
        "valid_time_wib",
        "lead_time_hour",
        "temperature_2m",
        "relative_humidity_2m",
        "cloud_cover",
        "precipitation",
        "wind_speed_10m",
        "raw_ghi_forecast",
        "is_day",
    ]

    print(
        "\n"
        + "=" * 110
    )

    print(
        "FORECAST +1h SAMPAI +6h"
    )

    print(
        "=" * 110
    )

    print(
        df[columns]
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

def save_result(df):
    output = (
        "data/raw/"
        "ecmwf_ifs_hres_single_run_poc.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    print(
        f"\nSaved to: {output}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    payload = (
        fetch_single_run()
    )

    df = parse_response(
        payload
    )

    validate(
        df
    )

    display_result(
        df
    )

    save_result(
        df
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SINGLE RUN POC SUCCESS ✅"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()