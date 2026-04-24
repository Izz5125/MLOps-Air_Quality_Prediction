import pandas as pd
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)


# GET RAW FILE
def get_raw_file():
    return os.path.join(RAW_DIR, "air_quality.csv")


# PREPROCESSING
def preprocess(df):
    # convert datetime
    df["datetime"] = pd.to_datetime(df["datetime"])

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
    # untuk menangani perbedaan decimal
    # dari source API/sensor
    df[sensor_columns] = df[sensor_columns].round(2)

    # handle missing values
    # forward fill karena data time-series
    df[sensor_columns] = df[sensor_columns].ffill()

    # jika masih ada missing di awal dataset
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

    # target forecasting
    # prediksi PM2.5 jam berikutnya
    df["future_pm25"] = df["pm25"].shift(-1)

    # hapus NaN akibat rolling,
    # lag, dan target shift
    df = df.dropna().reset_index(drop=True)

    return df


# SAVE
def save_processed(df):
    filename = os.path.join(PROCESSED_DIR, "processed_data.csv")

    df.to_csv(filename, index=False)
    print(f"Processed data saved to {filename}")


# MAIN
if __name__ == "__main__":
    print("Loading raw data...")
    file_path = get_raw_file()

    df = pd.read_csv(file_path)

    print("Preprocessing...")
    df_clean = preprocess(df)

    print("\nPreview processed data:")
    print(df_clean.tail())

    print("\nDataset shape:", df_clean.shape)

    print("\nSaving...")
    save_processed(df_clean)

    print("Done.")