import pandas as pd
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def get_latest_file():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    files.sort(reverse=True)
    return os.path.join(RAW_DIR, files[0])


def preprocess(df):
    # hapus missing
    df = df.dropna()

    # hapus duplikat
    df = df.drop_duplicates()

    # pastikan tipe data
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")

    # feature engineering
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day

    # rolling average
    df["pm25_rolling"] = df["pm25"].rolling(window=3).mean()

    return df


def save_processed(df):
    filename = f"{PROCESSED_DIR}/processed_data.csv"
    df.to_csv(filename, index=False)
    print(f"Processed data saved to {filename}")


if __name__ == "__main__":
    print("Loading latest raw data...")
    file_path = get_latest_file()

    df = pd.read_csv(file_path)

    print("Preprocessing...")
    df_clean = preprocess(df)

    print("Saving...")
    save_processed(df_clean)

    print("Done.")