import requests
import pandas as pd
from datetime import datetime
import os


# CONFIG
API_KEY = "38607f57a206d335ff516624811bd8a1be3891748fd9c4723d3ed55cea62c9c9"
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
def fetch_sensor_data(sensor_id, parameter_name):
    headers = {"X-API-Key": API_KEY}
    url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements"

    all_data = []
    page = 1
    limit = 100

    while True:
        params = {
            "limit": limit,
            "page": page
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        results = response.json().get("results", [])

        if not results:
            break

        all_data.extend(results)

        if len(results) < limit:
            break

        page += 1

    print(f"{parameter_name}: {len(all_data)} records fetched")

    return all_data


# TRANSFORM EACH SENSOR
def sensor_to_df(data, column_name):
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

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["datetime"] = df["datetime"].dt.tz_convert("Asia/Jakarta")

    return df


# MERGE ALL SENSOR DATA
def merge_all_sensors(sensor_map):
    merged_df = None

    for parameter, sensor_id in sensor_map.items():
        raw_data = fetch_sensor_data(sensor_id, parameter)
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


# SAVE DATA
def save_data(df):
    filename = f"{SAVE_DIR}/air_quality.csv"

    df.to_csv(filename, index=False)

    print(f"Data saved to {filename}")
    

# MAIN
if __name__ == "__main__":
    print("Getting sensor list...")
    sensor_map = get_sensor_list()

    print("Sensors found:")
    print(sensor_map)

    print("Fetching all sensor data...")
    final_df = merge_all_sensors(sensor_map)

    print("Preview:")
    print(final_df.tail())

    print("Saving data...")
    save_data(final_df)

    print("Done.")