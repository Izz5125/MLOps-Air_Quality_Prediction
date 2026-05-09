import json

BASELINE_MAE = 7.5

MAX_ALLOWED_MAE = (
    BASELINE_MAE * 1.15
)

with open("metrics.json", "r") as f:
    metrics = json.load(f)

mae = metrics["MAE"]

print(f"Current MAE: {mae}")

print(
    f"Maximum Allowed MAE: "
    f"{MAX_ALLOWED_MAE}"
)

if mae > MAX_ALLOWED_MAE:

    raise Exception(
        "Model validation failed. "
        "Performance degraded."
    )

print("Model validation passed.")