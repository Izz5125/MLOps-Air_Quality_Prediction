# MLOps Air Quality Prediction — Malang

Comprehensive documentation for building, training, and serving an Air Quality Index (AQI) prediction pipeline for Malang using PM2.5 sensor data. This repository implements an MLOps workflow: data ingestion → preprocessing → training → model registration → serving.

## Contents
- Project overview and goals
- Developer Quick Start (local + Docker)
- Data pipeline and model lifecycle
- MLflow & Model Registry integration
- DVC data versioning
- API usage examples and test instructions
- Deployment and operations runbook (extended documentation in /docs)

## Project Goals
- Provide reproducible pipelines for AQI prediction using PM2.5 and auxiliary sensors
- Track experiments and metrics using MLflow
- Version datasets with DVC and track model lineage
- Provide a production-ready API (FastAPI) with containerized deployment (Docker Compose)

## Key Features
- Data ingestion from OpenAQ (PM2.5)
- Automated preprocessing and feature engineering
- Training with experiment tracking (MAE, RMSE, R2)
- Automatic model registration to MLflow Model Registry
- Simple REST API for single and batch inference
- Docker Compose orchestration for API and MLflow

## Repository layout (important files)
- data/: raw and processed datasets
- src/: core scripts
  - src/ingest_data.py — ingest raw data
  - src/preprocess.py — preprocessing pipeline
  - src/train.py — training & MLflow logging
  - src/register_model.py — model registration logic
  - src/predict.py — prediction utilities
  - src/load_model.py — load model helper
- models/: saved artifacts and feature importance
- notebooks/: exploratory analysis
- api_service.py — FastAPI application entrypoint
- docker-compose.yaml — orchestration (API, MLflow)
- requirements.txt — Python dependencies

## Requirements
- Python 3.10+ (venv recommended)
- Docker & Docker Compose (for containerized dev/ops)
- DVC for data versioning
- See `requirements.txt` for exact Python packages: [requirements.txt](requirements.txt)

## Quick Start — Local (developer)
1) Create virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Ingest raw data (writes to `data/raw/air_quality.csv`):

```bash
python src/ingest_data.py
```

3) Preprocess data (writes to `data/processed/processed_data.csv`):

```bash
python src/preprocess.py
```

4) Train models and log experiments to MLflow (local MLflow required or use Docker Compose):

```bash
python src/train.py
```

5) Run the API locally (development):

```bash
python api_service.py
# then open http://localhost:8000/docs for Swagger UI
```

## Quick Start — Docker Compose (recommended for reproducibility)
1) Start services (API + MLflow by default):

```bash
docker compose up -d --scale aqi-api=3
```

2) View logs:

```bash
docker compose logs -f
```

3) Tear down:

```bash
docker compose down
```

## Data pipeline
- Ingest: `src/ingest_data.py` pulls PM2.5 data from OpenAQ and stores raw CSV in `data/raw/`.
- Preprocess: `src/preprocess.py` cleans, imputes missing values, encodes features, and writes processed CSV to `data/processed/`.
- Training: `src/train.py` runs experiments (random forest variations), logs parameters/metrics to MLflow, and saves the best model locally under `mlruns/` and `models/`.
- Registration: `src/register_model.py` registers the best model to MLflow Model Registry and optionally promotes it to `Production` if metrics improve.

## MLflow & Model Registry
- MLflow tracks experiments (MAE, RMSE, R2). Configure tracking URI in environment or `docker-compose.yaml` when using the containerized MLflow server.
- Model registration flow:
  1. Training script logs model and metrics
  2. Registration script promotes best model to Model Registry
  3. Promotion policy: if new model MAE < production MAE → promote to `Production`

## DVC (data versioning)
- Use DVC to version large datasets and track dataset changes.
  - Example:

```bash
dvc add data/raw/air_quality.csv
git add data/raw/air_quality.csv.dvc
git commit -m "Add raw air quality data"
```

- Configure a remote (S3/GCS) in `.dvc/config` for team workflows.

## API Usage
- `GET /health` — health & readiness
- `POST /predict` — single prediction (JSON payload)
- `POST /predict/batch` — batch predictions (array payload)
- `GET /model/info` — active model metadata

## Single prediction example (curl):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pm1":15.0,"pm25":25.0,"relativehumidity":65.0,"temperature":28.0,"um003":500.0}'
```

## Expected response example:

```json
{
  "predicted_pm25": 21.93,
  "predicted_aqi": 71.67,
  "aqi_category": "Moderate",
  "recommendation": "Udara cukup aman, kelompok sensitif kurangi aktivitas luar."
}
```

## Testing
- Unit tests: `tests/test_aqi.py`.
- Run tests:

```bash
pytest -q
```

## CI / CD
- A GitHub Actions workflow is included in `.github/workflows/` (pipeline triggers, ingestion, training, registration).
- Typical CI steps: lint → unit tests → build container (optional) → run pipeline steps (ingest/test/train) → register model (on success).

## Deployment & Operations
- For production deployments, use the `docker-compose.yaml` or convert to Kubernetes manifests.
- Persist MLflow artifacts via a mounted volume or remote storage.
- Ensure DVC remote is configured and accessible to CI/CD for data reproducibility.

## Troubleshooting & Notes
- If MLflow cannot be reached, ensure its service port (default 5000) is available and `MLFLOW_TRACKING_URI` is set.
- If model fails to load, check `mlruns/` and `models/` for saved artifacts and matching `mlflow.pyfunc` versions.

## License
- See `LICENSE` at the project root.


More detailed operational runbook and deployment instructions are available in `/docs/DEPLOYMENT.md`.

## Author
**Faiz Habibina Umiyabi**  
235150200111045