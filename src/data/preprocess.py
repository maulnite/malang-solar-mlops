from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RAW_FORECAST_DIR = Path("data/raw/forecast")
RAW_ERA5_DIR = Path("data/raw/era5")

PROCESSED_FEATURE_DIR = Path(
    "data/processed/features"
)

PROCESSED_ERA5_DIR = Path(
    "data/processed/era5"
)


FORECAST_REQUIRED_COLUMNS = [
    "source_model",
    "run_time_utc",
    "valid_time_utc",
    "lead_time_hour",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "raw_ghi_forecast",
    "is_day",
]


# ============================================================
# HELPER
# ============================================================

def latest_csv(directory: Path) -> Path:
    files = sorted(directory.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"Tidak ada CSV pada {directory}"
        )

    return files[-1]


def validate_columns(
    df: pd.DataFrame,
    required: list[str],
) -> None:
    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns missing: "
            + ", ".join(missing)
        )


# ============================================================
# FORECAST PREPROCESSING
# ============================================================

def preprocess_forecast(
    input_path: Path,
) -> Path:

    print(
        "=== FORECAST PREPROCESSING ==="
    )

    print(f"Input : {input_path}")

    df = pd.read_csv(input_path)

    input_rows = len(df)

    # --------------------------------------------------------
    # SCHEMA
    # --------------------------------------------------------

    validate_columns(
        df,
        FORECAST_REQUIRED_COLUMNS,
    )

    print("Schema validation : PASS")

    # --------------------------------------------------------
    # DATETIME NORMALIZATION
    # --------------------------------------------------------

    df["run_time_utc"] = pd.to_datetime(
        df["run_time_utc"],
        utc=True,
        errors="raise",
    )

    df["valid_time_utc"] = pd.to_datetime(
        df["valid_time_utc"],
        utc=True,
        errors="raise",
    )

    # --------------------------------------------------------
    # NUMERIC DATA TYPE
    # --------------------------------------------------------

    numeric_columns = [
        "lead_time_hour",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "cloud_cover",
        "pressure_msl",
        "wind_speed_10m",
        "wind_direction_10m",
        "raw_ghi_forecast",
        "is_day",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # MISSING VALUE CHECK
    # --------------------------------------------------------

    critical_missing = (
        df[FORECAST_REQUIRED_COLUMNS]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Critical missing   : "
        f"{critical_missing}"
    )

    if critical_missing > 0:
        raise ValueError(
            "Critical missing values ditemukan. "
            "Data dihentikan."
        )

    # --------------------------------------------------------
    # RANGE VALIDATION
    # --------------------------------------------------------

    invalid = pd.Series(
        False,
        index=df.index,
    )

    invalid |= ~df[
        "relative_humidity_2m"
    ].between(0, 100)

    invalid |= ~df[
        "cloud_cover"
    ].between(0, 100)

    invalid |= (
        df["raw_ghi_forecast"] < 0
    )

    invalid |= ~df[
        "lead_time_hour"
    ].between(1, 6)

    invalid |= ~df[
        "wind_direction_10m"
    ].between(0, 360)

    invalid |= (
        df["wind_speed_10m"] < 0
    )

    invalid_count = int(
        invalid.sum()
    )

    print(
        f"Invalid ranges     : "
        f"{invalid_count}"
    )

    if invalid_count > 0:
        raise ValueError(
            "Range validation failed."
        )

    # --------------------------------------------------------
    # TIME CONSISTENCY
    # --------------------------------------------------------

    calculated_lead = (
        (
            df["valid_time_utc"]
            - df["run_time_utc"]
        )
        .dt.total_seconds()
        / 3600
    )

    inconsistent_time = (
        calculated_lead
        != df["lead_time_hour"]
    )

    inconsistent_count = int(
        inconsistent_time.sum()
    )

    print(
        f"Time inconsistencies: "
        f"{inconsistent_count}"
    )

    if inconsistent_count > 0:
        raise ValueError(
            "run_time / valid_time / lead_time "
            "tidak konsisten."
        )

    # --------------------------------------------------------
    # DUPLICATE
    # --------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=[
            "source_model",
            "run_time_utc",
            "valid_time_utc",
        ],
        keep="first",
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    print(
        f"Duplicates         : "
        f"{duplicate_count}"
    )

    # Exact duplicate aman dibersihkan.
    if duplicate_count > 0:
        df = df.loc[
            ~duplicate_mask
        ].copy()

    # --------------------------------------------------------
    # SORTING
    # --------------------------------------------------------

    df = df.sort_values(
        "valid_time_utc"
    ).reset_index(drop=True)

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    print()
    print(
        "Generating time and wind features..."
    )

    # Gunakan waktu lokal Malang (WIB) untuk pola diurnal.
    local_time = (
        df["valid_time_utc"]
        .dt.tz_convert("Asia/Jakarta")
    )

    df["hour"] = (
        local_time.dt.hour
        + local_time.dt.minute / 60
    )

    df["day_of_year"] = (
        local_time.dt.dayofyear
    )

    # Cyclical hour
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

    # Cyclical day of year
    df["day_of_year_sin"] = np.sin(
        2 * np.pi
        * df["day_of_year"]
        / 365.25
    )

    df["day_of_year_cos"] = np.cos(
        2 * np.pi
        * df["day_of_year"]
        / 365.25
    )

    # Cyclical wind direction
    wind_radians = np.deg2rad(
        df["wind_direction_10m"]
    )

    df["wind_dir_sin"] = np.sin(
        wind_radians
    )

    df["wind_dir_cos"] = np.cos(
        wind_radians
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    PROCESSED_FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PROCESSED_FEATURE_DIR
        / (
            input_path.stem
            + "_features.parquet"
        )
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    print()
    print(
        "=== PREPROCESSING RESULT ==="
    )

    print(
        f"Input rows         : {input_rows}"
    )

    print(
        f"Output rows        : {len(df)}"
    )

    print(
        "Generated features :"
    )

    print("  - hour")
    print("  - day_of_year")
    print("  - hour_sin")
    print("  - hour_cos")
    print("  - day_of_year_sin")
    print("  - day_of_year_cos")
    print("  - wind_dir_sin")
    print("  - wind_dir_cos")

    print(
        f"Saved to           : "
        f"{output_path}"
    )

    print()
    print(
        "PREPROCESSING SUCCESS ✅"
    )

    return output_path


# ============================================================
# ERA5 PREPROCESSING
# ============================================================

def preprocess_era5(
    input_path: Path,
) -> Path:

    print(
        "=== ERA5 PREPROCESSING ==="
    )

    print(f"Input : {input_path}")

    df = pd.read_csv(
        input_path
    )

    validate_columns(
        df,
        [
            "source_model",
            "valid_time_utc",
            "actual_ghi",
        ],
    )

    df["valid_time_utc"] = (
        pd.to_datetime(
            df["valid_time_utc"],
            utc=True,
            errors="raise",
        )
    )

    df["actual_ghi"] = pd.to_numeric(
        df["actual_ghi"],
        errors="coerce",
    )

    if df["actual_ghi"].isna().any():
        raise ValueError(
            "Missing actual_ghi ditemukan."
        )

    if (df["actual_ghi"] < 0).any():
        raise ValueError(
            "actual_ghi memiliki nilai negatif."
        )

    df = (
        df.drop_duplicates(
            subset=[
                "source_model",
                "valid_time_utc",
            ]
        )
        .sort_values(
            "valid_time_utc"
        )
        .reset_index(drop=True)
    )

    PROCESSED_ERA5_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PROCESSED_ERA5_DIR
        / (
            input_path.stem
            + "_clean.parquet"
        )
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"Rows      : {len(df)}"
    )

    print(
        f"Saved to  : {output_path}"
    )

    print()
    print(
        "ERA5 PREPROCESSING SUCCESS ✅"
    )

    return output_path


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Preprocessing pipeline "
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
    )

    parser.add_argument(
        "--input",
        type=str,
        help=(
            "Optional path ke raw CSV. "
            "Jika kosong, otomatis pakai file terbaru."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.source == "forecast":

        input_path = (
            Path(args.input)
            if args.input
            else latest_csv(
                RAW_FORECAST_DIR
            )
        )

        preprocess_forecast(
            input_path
        )

        return

    input_path = (
        Path(args.input)
        if args.input
        else latest_csv(
            RAW_ERA5_DIR
        )
    )

    preprocess_era5(
        input_path
    )


if __name__ == "__main__":
    main()