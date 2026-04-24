import pandas as pd
import joblib

from aqi import pm25_to_aqi, classify_aqi, get_recommendation

MODEL_PATH = "models/best_model.pkl"
DATA_PATH = "data/processed/processed_data.csv"

feature_columns = [
    "pm1",
    "pm25",
    "relativehumidity",
    "temperature",
    "um003",
    "hour",
    "day",
    "month",
    "day_of_week",
    "pm25_rolling",
    "pm25_lag1"
]

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)

latest_data = df.iloc[-1:][feature_columns]

pred_pm25 = model.predict(latest_data)[0]
pred_aqi = pm25_to_aqi(pred_pm25)

aqi_status = classify_aqi(pred_aqi)

recommendation = get_recommendation(aqi_status)

print(f"Predicted next PM2.5: {pred_pm25:.2f}")
print(f"Predicted AQI: {pred_aqi:.2f}")
print(f"Air Quality Status: {aqi_status}")
print(f"Rekomendasi: {recommendation}")