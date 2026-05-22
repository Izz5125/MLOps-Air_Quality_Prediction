import requests
import pandas as pd
from datetime import datetime
import os
import argparse
import sys
import time


# CONFIG
API_KEY = "02ad25e3733cd2a0b515f0d3a3ef7e9d57f675b1793c3a130193334707871efb"
LOCATION_ID = 6144741

BASE_LOCATION_URL = f"https://api.openaq.org/v3/locations/{LOCATION_ID}"

SAVE_DIR = "data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)


# sensor yang ingin dipakai
TARGET_PARAMETERS = [
    "pm1",
    "pm25",
    "relativehumidity",
    "temperature",
    "um003"
]


# GET SENSOR LIST
def get_sensor_list():
    headers = {"X-API-Key": API_KEY}

    response = requests.get(BASE_LOCATION_URL, headers=headers)
    response.raise_for_status()

    sensors = response.json()["results"][0]["sensors"]

    sensor_map = {}

    for sensor in sensors:
        param_name = sensor["parameter"]["name"]

        if param_name in TARGET_PARAMETERS:
            sensor_map[param_name] = sensor["id"]

    return sensor_map


# FETCH MEASUREMENTS
def fetch_sensor_data(sensor_id, parameter_name, since_utc=None, max_pages=None):
    headers = {"X-API-Key": API_KEY}
    url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements"

    all_data = []
    page = 1
    limit = 50
    max_retries = 5
    backoff_base = 1.0

    stop_fetch = False
    while True:
        if max_pages is not None and page > max_pages:
            break
        params = {
            "limit": limit,
            "page": page
        }
        if since_utc is not None:
            params["datetime_from"] = since_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        # retry loop for transient errors / rate limiting
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, headers=headers, params=params)
            except requests.RequestException:
                if attempt == max_retries:
                    raise
                sleep_time = backoff_base * (2 ** (attempt - 1))
                time.sleep(sleep_time)
                continue

            if response.status_code == 429 or response.status_code >= 500:
                if attempt == max_retries:
                    response.raise_for_status()
                sleep_time = backoff_base * (2 ** (attempt - 1))
                time.sleep(sleep_time)
                continue

            # successful-ish response
            break

        response.raise_for_status()

        results = response.json().get("results", [])

        if not results:
            break

        if since_utc is not None:
            filtered = []
            for r in results:
                try:
                    r_dt = pd.to_datetime(r["period"]["datetimeFrom"]["utc"], utc=True)
                except Exception:
                    filtered.append(r)
                    continue

                if r_dt > since_utc:
                    filtered.append(r)

            all_data.extend(filtered)
        else:
            all_data.extend(results)

        if len(results) < limit:
            break

        page += 1

        # polite pause between pages to avoid rate limits
        time.sleep(0.2)

    print(f"{parameter_name}: {len(all_data)} records fetched")

    return all_data


# TRANSFORM EACH SENSOR
def sensor_to_df(data, column_name):
    # handle empty input
    if not data:
        return pd.DataFrame(columns=["datetime", column_name])

    df = pd.DataFrame({
        "datetime": [
            d["period"]["datetimeFrom"]["utc"]
            for d in data
        ],
        column_name: [
            d["value"]
            for d in data
        ]
    })

    # parse as UTC-aware timestamps then convert to local tz
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["datetime"] = df["datetime"].dt.tz_convert("Asia/Jakarta")

    return df


# MERGE ALL SENSOR DATA
def merge_all_sensors(sensor_map, max_pages=None, full_refresh=False):
    # Determine last timestamp from existing file (UTC)
    last_ts = None if full_refresh else get_latest_timestamp_from_existing()

    merged_df = None

    for parameter, sensor_id in sensor_map.items():
        raw_data = fetch_sensor_data(sensor_id, parameter, since_utc=last_ts, max_pages=max_pages)
        sensor_df = sensor_to_df(raw_data, parameter)

        if merged_df is None:
            merged_df = sensor_df
        else:
            merged_df = pd.merge(
                merged_df,
                sensor_df,
                on="datetime",
                how="outer"
            )

    merged_df = merged_df.sort_values("datetime")

    return merged_df


def get_latest_timestamp_from_existing():
    """Return latest timestamp from existing CSV as a UTC tz-aware datetime, or None."""
    filename = f"{SAVE_DIR}/air_quality.csv"
    if not os.path.exists(filename):
        return None

    try:
        existing = pd.read_csv(filename, parse_dates=["datetime"]) 
    except Exception:
        return None

    if existing.empty:
        return None

    # ensure datetime is timezone-aware; try to detect and normalize to UTC
    try:
        if existing["datetime"].dt.tz is None:
            # assume stored in Asia/Jakarta (local timezone used in this app)
            existing["datetime"] = existing["datetime"].dt.tz_localize("Asia/Jakarta")
    except Exception:
        # fallback: parse with utc
        existing["datetime"] = pd.to_datetime(existing["datetime"], utc=True)

    # convert to UTC for comparison
    existing["datetime"] = existing["datetime"].dt.tz_convert("UTC")

    return existing["datetime"].max()


# SAVE DATA
def save_data(df):
    filename = f"{SAVE_DIR}/air_quality.csv"

    df.to_csv(filename, index=False)

    print(f"Data saved to {filename}")
    

# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest air quality data")
    parser.add_argument("--full-refresh", action="store_true", help="Fetch all data (ignore existing file)")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum pages to fetch per sensor (useful to limit load)")
    args = parser.parse_args()

    print("Getting sensor list...")
    sensor_map = get_sensor_list()

    print("Sensors found:")
    print(sensor_map)

    print("Fetching sensor data...")
    final_df = merge_all_sensors(sensor_map, max_pages=args.max_pages, full_refresh=args.full_refresh)

    if final_df is None or (hasattr(final_df, 'empty') and final_df.empty):
        print("No new data to save. Exiting.")
        sys.exit(0)

    print("Preview:")
    print(final_df.tail())

    print("Saving data...")
    # If existing file exists, merge to avoid duplicates
    filename = f"{SAVE_DIR}/air_quality.csv"
    if os.path.exists(filename) and not args.full_refresh:
        try:
            existing = pd.read_csv(filename, parse_dates=["datetime"]) 
            # ensure timezone like earlier
            try:
                if existing["datetime"].dt.tz is None:
                    existing["datetime"] = existing["datetime"].dt.tz_localize("Asia/Jakarta")
            except Exception:
                existing["datetime"] = pd.to_datetime(existing["datetime"], utc=True)

            existing.set_index("datetime", inplace=True)
            final_df.set_index("datetime", inplace=True)

            combined = pd.concat([existing, final_df])
            # keep the last occurrence (prefer newly-fetched rows)
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            combined.reset_index(inplace=True)

            save_data(combined)
        except Exception:
            # fallback: save what we have
            save_data(final_df.reset_index() if hasattr(final_df, 'index') else final_df)
    else:
        save_data(final_df)

    print("Done.")