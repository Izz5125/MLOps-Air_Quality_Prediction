import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# CONFIG
DATA_PATH = "data/processed/processed_data.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# LOAD DATA
print("Loading processed dataset...")
df = pd.read_csv(DATA_PATH)
df = df.sort_values("datetime")


# last 60 days (~1440 hours)
df = df.tail(1440)

print("Using last 60 days dataset")
print("Dataset shape:", df.shape)


# FEATURES & TARGET
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

X = df[feature_columns]
y = df["future_pm25"]


# TIME SERIES SPLIT
split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")


# MLFLOW
mlflow.set_experiment("Air_Quality_PM25_Prediction")

experiments = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 15}
]

best_model = None
best_mae = float("inf")
best_params = None

for params in experiments:
    print(f"\nRunning experiment: {params}")

    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        print(f"R2: {r2}")

        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_params = params


# SAVE BEST MODEL
joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")

print("\nBEST MODEL SAVED")
print("Best MAE:", best_mae)
print("Best Params:", best_params)


# FEATURE IMPORTANCE
importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": best_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFeature Importance:")
print(importance)

importance.to_csv(
    f"{MODEL_DIR}/feature_importance.csv",
    index=False
)

plt.figure(figsize=(8,5))
plt.bar(
    importance["feature"],
    importance["importance"]
)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/feature_importance.png")

print("Feature importance saved.")