import pandas as pd
import os
from sqlalchemy import create_engine, text

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ============================================
# POSTGRESQL CONFIG
# ============================================
DB_USER = os.getenv('POSTGRES_USER', 'mlflow')
DB_PASS = os.getenv('POSTGRES_PASSWORD', 'mlflow_password')
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'mlflowdb')

try:
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("✅ PostgreSQL connection established")
    USE_DB = True
except Exception as e:
    print(f"⚠️  PostgreSQL not available: {e}")
    print("⚠️  Will use CSV only")
    USE_DB = False


# ============================================
# GET RAW DATA (PostgreSQL atau CSV)
# ============================================
def get_raw_data():
    # 1. Coba dari PostgreSQL dulu
    if USE_DB:
        try:
            df = pd.read_sql("SELECT * FROM raw_data ORDER BY datetime", engine)
            if not df.empty:
                print(f"✅ Loaded {len(df)} rows from PostgreSQL")
                return df
            else:
                print("⚠️  PostgreSQL table is empty")
        except Exception as e:
            print(f"⚠️  PostgreSQL read failed: {e}")
    
    # 2. Fallback ke CSV
    file_path = os.path.join(RAW_DIR, "air_quality.csv")
    if os.path.exists(file_path):
        print(f"✅ Loaded from CSV: {file_path}")
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"No data found! Neither PostgreSQL nor CSV available.")


# ============================================
# PREPROCESSING
# ============================================
def preprocess(df):
    # convert datetime (FIX: pakai utc=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # urutkan berdasarkan waktu
    df = df.sort_values("datetime").reset_index(drop=True)

    # hapus duplicate
    df = df.drop_duplicates().reset_index(drop=True)

    # sensor columns
    sensor_columns = [
        "pm1",
        "pm25",
        "relativehumidity",
        "temperature",
        "um003"
    ]

    # convert ke numeric
    for col in sensor_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # standardisasi precision
    df[sensor_columns] = df[sensor_columns].round(2)

    # handle missing values (forward fill)
    df[sensor_columns] = df[sensor_columns].ffill()

    # hapus baris yang masih NaN
    df = df.dropna()

    # feature engineering
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek

    # rolling average PM2.5 (3 jam terakhir)
    df["pm25_rolling"] = df["pm25"].rolling(window=3).mean()

    # lag feature (PM2.5 jam sebelumnya)
    df["pm25_lag1"] = df["pm25"].shift(1)

    # target forecasting (PM2.5 jam berikutnya)
    df["future_pm25"] = df["pm25"].shift(-1)

    # hapus NaN akibat rolling, lag, dan target shift
    df = df.dropna().reset_index(drop=True)

    return df


# ============================================
# SAVE PROCESSED DATA (PostgreSQL + CSV)
# ============================================
def save_processed(df):
    # 1. Simpan ke CSV (backup)
    filename = os.path.join(PROCESSED_DIR, "processed_data.csv")
    df.to_csv(filename, index=False)
    print(f"✅ Processed data saved to {filename}")
    
    # 2. Simpan ke PostgreSQL
    if USE_DB:
        try:
            df.to_sql('processed_data', engine, if_exists='replace', index=False)
            result = engine.connect().execute(text("SELECT COUNT(*) FROM processed_data"))
            total = result.fetchone()[0]
            print(f"✅ Processed data saved to PostgreSQL. Total: {total} rows")
        except Exception as e:
            print(f"⚠️  PostgreSQL save failed: {e}")


# ============================================
# CREATE TABLE IF NOT EXISTS
# ============================================
def init_database():
    """Create processed_data table if not exists"""
    if not USE_DB:
        return
    
    try:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS processed_data (
            id SERIAL PRIMARY KEY,
            datetime TIMESTAMP NOT NULL,
            pm1 FLOAT,
            pm25 FLOAT,
            relativehumidity FLOAT,
            temperature FLOAT,
            um003 FLOAT,
            hour INTEGER,
            day INTEGER,
            month INTEGER,
            day_of_week INTEGER,
            pm25_rolling FLOAT,
            pm25_lag1 FLOAT,
            future_pm25 FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        engine.connect().execute(text(create_table_sql))
        print("✅ Database table 'processed_data' ready")
    except Exception as e:
        print(f"⚠️  Table creation failed: {e}")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    init_database()
    
    print("Loading raw data...")
    df = get_raw_data()

    print("Preprocessing...")
    df_clean = preprocess(df)

    print("\nPreview processed data:")
    print(df_clean.tail())

    print("\nDataset shape:", df_clean.shape)

    print("\nSaving...")
    save_processed(df_clean)

    print("Done.")
