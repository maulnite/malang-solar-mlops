import requests


LATITUDE = -7.9666
LONGITUDE = 112.6326

MODEL = "ecmwf_ifs"

URL = "https://single-runs-api.open-meteo.com/v1/forecast"


# Audit ringan:
# - satu run sebelum awal archive
# - beberapa run pada tanggal awal archive
# - satu archived run terbaru
RUNS_TO_CHECK = [
    "2024-03-13T18:00",
    "2024-03-14T00:00",
    "2024-03-14T06:00",
    "2024-03-14T12:00",
    "2024-03-14T18:00",
    "2026-09-18T00:00",
]


def check_run(run_time):
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "models": MODEL,
        "run": run_time,
        "hourly": "shortwave_radiation",
        "forecast_hours": 2,
        "timezone": "GMT",
    }

    try:
        response = requests.get(
            URL,
            params=params,
            timeout=60,
        )

    except requests.RequestException as error:
        return {
            "run": run_time,
            "status": "REQUEST ERROR",
            "http": None,
            "rows": 0,
            "detail": str(error),
        }

    if not response.ok:
        detail = response.text

        if len(detail) > 120:
            detail = detail[:120] + "..."

        return {
            "run": run_time,
            "status": "NOT AVAILABLE",
            "http": response.status_code,
            "rows": 0,
            "detail": detail,
        }

    payload = response.json()

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    ghi = hourly.get("shortwave_radiation", [])

    valid_rows = sum(
        value is not None
        for value in ghi
    )

    if valid_rows == 0:
        status = "NOT AVAILABLE"
    else:
        status = "AVAILABLE"

    return {
        "run": run_time,
        "status": status,
        "http": response.status_code,
        "rows": valid_rows,
        "detail": (
            f"{times[0]} -> {times[-1]}"
            if times
            else "No hourly timestamps"
        ),
    }


def main():
    print(
        "=== ECMWF IFS HRES SINGLE RUNS HISTORY AUDIT ==="
    )

    print(
        f"Coordinate : {LATITUDE}, {LONGITUDE}"
    )

    print(
        f"Model      : {MODEL}"
    )

    print(
        "\nChecking archive boundary and recent availability...\n"
    )

    results = []

    for run_time in RUNS_TO_CHECK:
        result = check_run(run_time)
        results.append(result)

        print(
            f"{result['run']} | "
            f"HTTP {result['http']} | "
            f"{result['status']} | "
            f"rows={result['rows']}"
        )

    print(
        "\n"
        + "=" * 72
    )

    print("SUMMARY")

    print(
        "=" * 72
    )

    available = [
        result
        for result in results
        if result["status"] == "AVAILABLE"
    ]

    unavailable = [
        result
        for result in results
        if result["status"] != "AVAILABLE"
    ]

    print(
        f"Available checks   : {len(available)}"
    )

    print(
        f"Unavailable checks : {len(unavailable)}"
    )

    if available:
        print(
            "Earliest available among tested runs : "
            f"{available[0]['run']}"
        )

        print(
            "Latest available among tested runs   : "
            f"{available[-1]['run']}"
        )

    print(
        "\nNOTE:"
    )

    print(
        "This is a boundary verification, "
        "not a full historical download."
    )


if __name__ == "__main__":
    main()