import requests
import pandas as pd
from datetime import datetime, timezone
import os

# === CONFIG ===
API_KEY = "38607f57a206d335ff516624811bd8a1be3891748fd9c4723d3ed55cea62c9c9"
SENSOR_ID = 14739443
BASE_URL = f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/measurements"

SAVE_DIR = "data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

def fetch_data():
    headers = {"X-API-Key": API_KEY}
    
    all_data = []
    page = 1
    limit = 100

    while True:
        params = {"limit": limit, "page": page}
        response = requests.get(BASE_URL, headers=headers, params=params)
        results = response.json().get("results", [])

        if not results:
            break

        all_data.extend(results)

        if len(results) < limit:
            break

        page += 1

    return all_data


def transform_to_df(data):
    df = pd.DataFrame({
        "datetime": [d["period"]["datetimeFrom"]["utc"] for d in data],
        "pm25": [d["value"] for d in data],
        "unit": [d["parameter"]["units"] for d in data]
    })

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    return df


def save_data(df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SAVE_DIR}/air_quality_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


if __name__ == "__main__":
    print("Fetching data...")
    raw_data = fetch_data()

    print("Transforming data...")
    df = transform_to_df(raw_data)

    print("Saving data...")
    save_data(df)

    print("Done.")