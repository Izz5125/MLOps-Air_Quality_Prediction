import requests
import pandas as pd
from datetime import datetime
import os
import argparse
import sys
import time
from sqlalchemy import create_engine, text


# ============================================
# CONFIG
# ============================================
API_KEY = "02ad25e3733cd2a0b515f0d3a3ef7e9d57f675b1793c3a130193334707871efb"
LOCATION_ID = 6144741

BASE_LOCATION_URL = f"https://api.openaq.org/v3/locations/{LOCATION_ID}"

SAVE_DIR = "data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

# POSTGRESQL CONFIG
DB_USER = os.getenv('POSTGRES_USER', 'mlflow')
DB_PASS = os.getenv('POSTGRES_PASSWORD', 'mlflow_password')
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'mlflowdb')

# Koneksi PostgreSQL
try:
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    # Test koneksi
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("✅ PostgreSQL connection established")
    USE_DB = True
except Exception as e:
    print(f"⚠️  PostgreSQL not available: {e}")
    print("⚠️  Will use CSV only")
    USE_DB = False

# sensor yang ingin dipakai
TARGET_PARAMETERS = [
    "pm1",
    "pm25",
    "relativehumidity",
    "temperature",
    "um003"
]


# ============================================
# GET SENSOR LIST
# ============================================
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


# ============================================
# FETCH MEASUREMENTS
# ============================================
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


# ============================================
# TRANSFORM EACH SENSOR
# ============================================
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


# ============================================
# MERGE ALL SENSOR DATA
# ============================================
def merge_all_sensors(sensor_map, max_pages=None, full_refresh=False):
    # Determine last timestamp from existing data (PostgreSQL or CSV)
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

    if merged_df is not None:
        merged_df = merged_df.sort_values("datetime")

    return merged_df


# ============================================
# GET LATEST TIMESTAMP
# ============================================
def get_latest_timestamp_from_existing():
    """Return latest timestamp from PostgreSQL or CSV, as UTC tz-aware datetime."""
    
    # 1. Coba dari PostgreSQL dulu
    if USE_DB:
        try:
            result = engine.connect().execute(text("SELECT MAX(datetime) FROM raw_data"))
            last_date = result.fetchone()[0]
            if last_date:
                # Pastikan timezone-aware (UTC)
                if last_date.tzinfo is None:
                    import pytz
                    last_date = last_date.replace(tzinfo=pytz.UTC)
                print(f"✅ Latest data in PostgreSQL: {last_date}")
                return last_date
        except Exception as e:
            print(f"⚠️  PostgreSQL check failed: {e}")
    
    # 2. Fallback ke CSV
    filename = f"{SAVE_DIR}/air_quality.csv"
    if not os.path.exists(filename):
        print("⚠️  No existing data found. Will fetch all available data.")
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

    print(f"✅ Latest data in CSV: {existing['datetime'].max()}")
    return existing["datetime"].max()


# ============================================
# SAVE DATA (PostgreSQL + CSV)
# ============================================
def save_data(df):
    if df is None or df.empty:
        print("No data to save.")
        return
    
    # ============================================
    # 1. SIMPAN KE CSV (BACKUP)
    # ============================================
    filename = f"{SAVE_DIR}/air_quality.csv"
    
    if os.path.exists(filename):
        try:
            existing = pd.read_csv(filename, parse_dates=["datetime"])
            # ensure timezone like earlier
            try:
                if existing["datetime"].dt.tz is None:
                    existing["datetime"] = existing["datetime"].dt.tz_localize("Asia/Jakarta")
            except Exception:
                existing["datetime"] = pd.to_datetime(existing["datetime"], utc=True)

            existing.set_index("datetime", inplace=True)
            df_copy = df.copy()
            df_copy.set_index("datetime", inplace=True)

            combined = pd.concat([existing, df_copy])
            # keep the last occurrence (prefer newly-fetched rows)
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            combined.reset_index(inplace=True)

            combined.to_csv(filename, index=False)
            print(f"✅ CSV updated: {filename} ({len(combined)} rows)")
        except Exception as e:
            print(f"⚠️  CSV merge failed: {e}")
            df.to_csv(filename, index=False)
            print(f"✅ CSV saved as new file: {filename} ({len(df)} rows)")
    else:
        df.to_csv(filename, index=False)
        print(f"✅ CSV saved: {filename} ({len(df)} rows)")
    
    # ============================================
    # 2. SIMPAN KE POSTGRESQL
    # ============================================
    if USE_DB:
        try:
            # Siapkan DataFrame untuk PostgreSQL
            df_db = df.copy()
            
            # Konversi datetime ke timezone-naive UTC untuk PostgreSQL
            df_db['datetime'] = pd.to_datetime(df_db['datetime'])
            if df_db['datetime'].dt.tz is not None:
                df_db['datetime'] = df_db['datetime'].dt.tz_convert('UTC').dt.tz_localize(None)
            
            # Insert data baru satu per satu (hindari batch error karena duplicate)
            inserted = 0
            skipped = 0
            
            for _, row in df_db.iterrows():
                try:
                    row_df = pd.DataFrame([row])
                    row_df.to_sql('raw_data', engine, if_exists='append', index=False, method='multi')
                    inserted += 1
                except Exception as e:
                    if 'duplicate key' in str(e).lower() or 'unique constraint' in str(e).lower():
                        skipped += 1
                    else:
                        print(f"⚠️  Insert error for row: {e}")
            
            # Tampilkan statistik
            result = engine.connect().execute(text("SELECT COUNT(*) FROM raw_data"))
            total = result.fetchone()[0]
            print(f"✅ PostgreSQL: {inserted} inserted, {skipped} skipped (duplicates)")
            print(f"📊 Total records in database: {total}")
            
        except Exception as e:
            print(f"⚠️  PostgreSQL save failed: {e}")
            import traceback
            traceback.print_exc()


# ============================================
# CREATE TABLE IF NOT EXISTS
# ============================================
def init_database():
    """Create tables if they don't exist"""
    if not USE_DB:
        return
    
    try:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS raw_data (
            id SERIAL PRIMARY KEY,
            datetime TIMESTAMP NOT NULL,
            pm1 FLOAT,
            pm25 FLOAT,
            relativehumidity FLOAT,
            temperature FLOAT,
            um003 FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(datetime)
        );
        """
        engine.connect().execute(text(create_table_sql))
        print("✅ Database table 'raw_data' ready")
    except Exception as e:
        print(f"⚠️  Table creation failed: {e}")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest air quality data")
    parser.add_argument("--full-refresh", action="store_true", help="Fetch all data (ignore existing file)")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum pages to fetch per sensor (useful to limit load)")
    args = parser.parse_args()

    # Inisialisasi database
    init_database()

    print("Getting sensor list...")
    sensor_map = get_sensor_list()

    print("Sensors found:")
    for param, sensor_id in sensor_map.items():
        print(f"  {param}: {sensor_id}")

    print("\nFetching sensor data...")
    final_df = merge_all_sensors(sensor_map, max_pages=args.max_pages, full_refresh=args.full_refresh)

    if final_df is None or (hasattr(final_df, 'empty') and final_df.empty):
        print("No new data to save. Exiting.")
        sys.exit(0)

    print("\nPreview:")
    print(final_df.tail())

    print("\nSaving data...")
    save_data(final_df)

    print("\nDone.")