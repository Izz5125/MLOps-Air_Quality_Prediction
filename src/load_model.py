import mlflow.pyfunc
import pandas as pd

MODEL_NAME = "AQI_Predictor"

print("Loading model from Production...")

# load model dari stage Production
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/Production"
)

print("Model loaded successfully!")

# DATA DUMMY (harus sesuai fitur)
data = pd.DataFrame([{
    "pm1": 10,
    "pm25": 20,
    "relativehumidity": 70,
    "temperature": 25,
    "um003": 500,
    "hour": 12,
    "day": 15,
    "month": 4,
    "day_of_week": 2,
    "pm25_rolling": 18,
    "pm25_lag1": 19
}])

print("\nInput data:")
print(data)

# PREDIKSI
pred = model.predict(data)

print("\nPrediction result:")
print(pred)