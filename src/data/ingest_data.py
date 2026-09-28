from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

LATITUDE = -7.9666
LONGITUDE = 112.6326

IFS_MODEL = "ecmwf_ifs"

SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"

RAW_FORECAST_DIR = Path("data/raw/forecast")
RAW_ERA5_DIR = Path("data/raw/era5")

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
# HTTP HELPER
# ============================================================

def request_json(
    url: str,
    params: dict,
    retries: int = 3,
    timeout: int = 30,
) -> dict:
    """Request JSON dengan retry sederhana untuk error jaringan."""

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)

            if response.status_code == 200:
                return response.json()

            raise RuntimeError(
                f"HTTP {response.status_code}: {response.text[:300]}"
            )

        except (requests.RequestException, RuntimeError) as exc:
            if attempt == retries:
                raise

            wait_seconds = 2 ** (attempt - 1)

            print(
                f"Request gagal ({exc}). "
                f"Retry dalam {wait_seconds} detik..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Request gagal.")


# ============================================================
# ECMWF IFS
# ============================================================

def candidate_ifs_runs(max_runs: int = 12) -> list[datetime]:
    """
    Buat kandidat run ECMWF:
    00, 06, 12, 18 UTC.

    Dicek mundur dari cycle terbaru.
    """

    now = datetime.now(timezone.utc)

    cycle_hour = (now.hour // 6) * 6

    latest_cycle = now.replace(
        hour=cycle_hour,
        minute=0,
        second=0,
        microsecond=0,
    )

    return [
        latest_cycle - timedelta(hours=6 * i)
        for i in range(max_runs)
    ]


def format_run(run_time: datetime) -> str:
    return run_time.strftime("%Y-%m-%dT%H:%M")


def is_run_available(run_time: datetime) -> bool:
    """
    Probe ringan untuk mengecek apakah run sudah tersedia.
    """

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "models": IFS_MODEL,
        "run": format_run(run_time),
        "hourly": "shortwave_radiation",
        "forecast_hours": 2,
        "timezone": "GMT",
    }

    try:
        response = requests.get(
            SINGLE_RUNS_URL,
            params=params,
            timeout=20,
        )
    except requests.RequestException:
        return False

    if response.status_code != 200:
        return False

    payload = response.json()

    hourly = payload.get("hourly", {})

    return bool(hourly.get("time"))


def find_latest_available_run() -> datetime:
    print("Checking latest available ECMWF IFS run...")

    for run_time in candidate_ifs_runs():
        print(
            f"  checking {format_run(run_time)} UTC...",
            end=" ",
        )

        if is_run_available(run_time):
            print("AVAILABLE")
            return run_time

        print("not available")

    raise RuntimeError(
        "Tidak menemukan ECMWF IFS run yang tersedia "
        "dalam window pencarian."
    )


def fetch_forecast(run_time: datetime) -> Path:
    RAW_FORECAST_DIR.mkdir(parents=True, exist_ok=True)

    filename = (
        f"ecmwf_ifs_"
        f"{run_time.strftime('%Y%m%dT%H%MZ')}.csv"
    )

    output_path = RAW_FORECAST_DIR / filename

    # Idempotency
    if output_path.exists():
        print()
        print(f"Run {format_run(run_time)} UTC sudah pernah disimpan.")
        print(f"Existing file: {output_path}")
        print("Skipping duplicate ingestion.")

        return output_path

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "models": IFS_MODEL,
        "run": format_run(run_time),
        "hourly": ",".join(HOURLY_VARIABLES),

        # Ambil lebih panjang, kemudian filter lead +1 sampai +6.
        "forecast_hours": 12,

        "timezone": "GMT",
    }

    print()
    print("Fetching complete forecast variables...")

    payload = request_json(
        SINGLE_RUNS_URL,
        params=params,
    )

    hourly = payload.get("hourly")

    if not hourly:
        raise RuntimeError(
            "API tidak mengembalikan hourly data."
        )

    missing_api_variables = [
        column
        for column in HOURLY_VARIABLES
        if column not in hourly
    ]

    if missing_api_variables:
        raise RuntimeError(
            "Variable tidak ditemukan pada response: "
            + ", ".join(missing_api_variables)
        )

    df = pd.DataFrame(
        {
            "valid_time_utc": hourly["time"],
            "temperature_2m": hourly["temperature_2m"],
            "relative_humidity_2m": hourly["relative_humidity_2m"],
            "precipitation": hourly["precipitation"],
            "cloud_cover": hourly["cloud_cover"],
            "pressure_msl": hourly["pressure_msl"],
            "wind_speed_10m": hourly["wind_speed_10m"],
            "wind_direction_10m": hourly["wind_direction_10m"],
            "raw_ghi_forecast": hourly["shortwave_radiation"],
            "is_day": hourly["is_day"],
        }
    )

    # --------------------------------------------------------
    # TIME METADATA
    # --------------------------------------------------------

    df["valid_time_utc"] = pd.to_datetime(
        df["valid_time_utc"],
        utc=True,
    )

    run_timestamp = pd.Timestamp(run_time)

    df.insert(
        0,
        "run_time_utc",
        run_timestamp,
    )

    df.insert(
        2,
        "lead_time_hour",
        (
            (
                df["valid_time_utc"]
                - run_timestamp
            )
            .dt.total_seconds()
            / 3600
        ).astype(int),
    )

    # Hanya lead +1 sampai +6
    df = df[
        df["lead_time_hour"].between(1, 6)
    ].copy()

    if len(df) != 6:
        raise RuntimeError(
            f"Expected 6 lead times, received {len(df)}."
        )

    # WIB hanya untuk kemudahan inspeksi/reporting.
    df.insert(
        2,
        "valid_time_wib",
        df["valid_time_utc"]
        .dt.tz_convert("Asia/Jakarta"),
    )

    # --------------------------------------------------------
    # SOURCE METADATA
    # --------------------------------------------------------

    df.insert(0, "source_model", IFS_MODEL)

    df["latitude"] = payload.get("latitude")
    df["longitude"] = payload.get("longitude")
    df["elevation_m"] = payload.get("elevation")

    df["ingested_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    # --------------------------------------------------------
    # BASIC CHECK
    # --------------------------------------------------------

    duplicate_count = df.duplicated(
        subset=[
            "source_model",
            "run_time_utc",
            "valid_time_utc",
        ]
    ).sum()

    missing_count = df.isna().sum().sum()

    # Save timestamp as ISO string
    df["run_time_utc"] = (
        df["run_time_utc"]
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    df["valid_time_utc"] = (
        df["valid_time_utc"]
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    df["valid_time_wib"] = (
        df["valid_time_wib"]
        .dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=== FORECAST INGESTION RESULT ===")
    print(f"Run time          : {format_run(run_time)} UTC")
    print(f"Rows saved        : {len(df)}")
    print(f"Lead times        : {df['lead_time_hour'].tolist()}")
    print(f"Missing values    : {missing_count}")
    print(f"Duplicate records : {duplicate_count}")
    print(f"Saved to          : {output_path}")
    print()
    print("FORECAST INGESTION SUCCESS ✅")

    return output_path


# ============================================================
# ERA5
# ============================================================

def fetch_era5(target_date: date) -> Path:
    RAW_ERA5_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RAW_ERA5_DIR
        / f"era5_{target_date.strftime('%Y%m%d')}.csv"
    )

    if output_path.exists():
        print(
            f"ERA5 {target_date} sudah tersedia: "
            f"{output_path}"
        )

        print("Skipping duplicate ingestion.")

        return output_path

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),

        "hourly": "shortwave_radiation",

        "models": "era5",

        "timezone": "GMT",
    }

    print(
        f"Fetching ERA5 reference for "
        f"{target_date}..."
    )

    payload = request_json(
        ERA5_URL,
        params=params,
    )

    hourly = payload.get("hourly")

    if not hourly:
        raise RuntimeError(
            "ERA5 API tidak mengembalikan hourly data."
        )

    if "shortwave_radiation" not in hourly:
        raise RuntimeError(
            "shortwave_radiation tidak tersedia."
        )

    df = pd.DataFrame(
        {
            "valid_time_utc": hourly["time"],
            "actual_ghi": hourly["shortwave_radiation"],
        }
    )

    df["valid_time_utc"] = pd.to_datetime(
        df["valid_time_utc"],
        utc=True,
    )

    df.insert(
        0,
        "source_model",
        "era5",
    )

    df["latitude"] = payload.get("latitude")
    df["longitude"] = payload.get("longitude")
    df["elevation_m"] = payload.get("elevation")

    df["ingested_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    df["valid_time_utc"] = (
        df["valid_time_utc"]
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=== ERA5 INGESTION RESULT ===")
    print(f"Reference date : {target_date}")
    print(f"Rows saved     : {len(df)}")
    print(
        f"Missing GHI    : "
        f"{df['actual_ghi'].isna().sum()}"
    )
    print(f"Saved to       : {output_path}")
    print()
    print("ERA5 INGESTION SUCCESS ✅")

    return output_path


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dynamic weather data ingestion "
            "for Malang Solar MLOps."
        )
    )

    parser.add_argument(
        "--source",
        choices=[
            "forecast",
            "era5",
        ],
        default="forecast",
        help="Datasource yang ingin diambil.",
    )

    parser.add_argument(
        "--date",
        type=str,
        help=(
            "Tanggal ERA5 YYYY-MM-DD. "
            "Jika kosong, menggunakan UTC today - 6 days."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.source == "forecast":
        print(
            "=== ECMWF IFS DYNAMIC DATA INGESTION ==="
        )

        print(
            f"Coordinate : "
            f"{LATITUDE}, {LONGITUDE}"
        )

        print(
            f"Model      : {IFS_MODEL}"
        )

        print()

        latest_run = find_latest_available_run()

        print()
        print(
            f"Latest available run: "
            f"{format_run(latest_run)} UTC"
        )

        fetch_forecast(latest_run)

        return

    # ERA5 ---------------------------------------------------

    print(
        "=== ERA5 DAILY LABEL REFRESH ==="
    )

    if args.date:
        target_date = date.fromisoformat(
            args.date
        )

    else:
        # ERA5 ~5 hari delay.
        # -6 hari digunakan sebagai default konservatif.
        target_date = (
            datetime.now(timezone.utc).date()
            - timedelta(days=6)
        )

    fetch_era5(target_date)


if __name__ == "__main__":
    main()