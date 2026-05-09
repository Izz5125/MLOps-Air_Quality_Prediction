import json
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "AQI_Predictor"

mlflow.set_tracking_uri("file:./mlruns")

# LOAD RUN INFO
with open("run_info.json", "r") as f:
    run_info = json.load(f)

best_run_id = run_info["best_run_id"]

model_uri = f"runs:/{best_run_id}/model"

# REGISTER MODEL
result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print("Model registered successfully!")

print("Model Version:", result.version)

# TRANSITION TO STAGING
client = MlflowClient()

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=result.version,
    stage="Staging"
)

print(
    f"Model version "
    f"{result.version} "
    f"moved to Staging."
)