import json
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "AQI_Predictor"

mlflow.set_tracking_uri("file:./mlruns")

# Load run info & metrics
with open("run_info.json", "r") as f:
    run_info = json.load(f)

with open("metrics.json", "r") as f:
    new_metrics = json.load(f)

best_run_id = run_info["best_run_id"]
new_mae = new_metrics["MAE"]
model_uri = f"runs:/{best_run_id}/model"

# Register model
result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
print(f"✅ Model registered: Version {result.version}")
print(f"📊 New MAE: {new_mae:.4f}")

client = MlflowClient()

# Cek model di Production
try:
    prod_model = client.get_model_version_by_alias(MODEL_NAME, "production")
    prod_run = client.get_run(prod_model.run_id)
    prod_mae = prod_run.data.metrics.get("MAE", float("inf"))
    print(f"📊 Production MAE: {prod_mae:.4f}")

    if new_mae < prod_mae:
        # Model baru lebih baik → replace Production
        client.delete_registered_model_alias(MODEL_NAME, "production")
        client.set_registered_model_alias(MODEL_NAME, "production", result.version)
        print(f"✅ Version {result.version} PROMOTED to Production (better!)")
    else:
        # Tetap di Staging
        client.set_registered_model_alias(MODEL_NAME, "staging", result.version)
        print(f"⚠️  Version {result.version} kept in Staging (not better)")

except Exception:
    # Belum ada Production → langsung Production
    client.set_registered_model_alias(MODEL_NAME, "production", result.version)
    print(f"✅ Version {result.version} promoted to Production (first model)")
