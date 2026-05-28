# MLOps Air Quality Prediction - Malang

## Project Overview

This project aims to build a machine learning system to predict the daily Air Quality Index (AQI) in Malang using real-time air quality sensor data (PM2.5).

The system is designed following MLOps principles, enabling:
- Continuous data ingestion from external APIs
- Automated data preprocessing
- Model training and experiment tracking
- Model versioning and lifecycle management
- Scalable and reproducible pipeline
- Data versioning using DVC
- Containerized services with Docker Compose orchestration
- Horizontal scaling with Docker Compose replicas


## Project Structure

```
MLOps-Air_Quality_Prediction/
|
+-- data/
|   +-- raw/
|   |   +-- air_quality.csv
|   |   +-- air_quality.csv.dvc
|   +-- processed/
|       +-- processed_data.csv
|       +-- processed_data.csv.dvc
|
+-- src/
|   +-- aqi.py
|   +-- ingest_data.py
|   +-- preprocess.py
|   +-- train.py
|   +-- evaluate.py
|   +-- predict.py
|   +-- load_model.py
|   +-- register_model.py
|
+-- models/
|   +-- best_model.pkl
|   +-- feature_importance.csv
|   +-- feature_importance.png
|
+-- notebooks/
|   +-- 01_initial_data_exploration.ipynb
|
+-- config/
+-- tests/
|   +-- test_aqi.py
|
+-- api_service.py
+-- app_ui.py
+-- Dockerfile.api
+-- docker-compose.yaml
|
+-- model_registry.yaml
+-- model_registry.yaml.dvc
+-- requirements.txt
|
+-- .github/
|   +-- workflows/
|       +-- mlops-pipeline.yml
|
+-- .dvc/
+-- README.md
```


## Data Source

Data is collected from the OpenAQ API.

- Parameter: PM2.5
- Format: JSON
- Update frequency: ~1 hour
- Data type: Time-series


## Development Environment

This project uses GitHub Codespaces.

Main dependencies:
- Python 3.10+
- Pandas
- Scikit-learn
- MLflow
- FastAPI
- Uvicorn
- Requests
- DVC
- Docker & Docker Compose


## Running the Data Pipeline

### 1. Data Ingestion

```bash
python src/ingest_data.py
```

Output: Stored in ```data/raw/air_quality.csv```

### 2. Data Preprocessing

```bash
python src/preprocess.py
```

Output: ```data/processed/processed_data.csv```


## Model Training & Experiment Tracking

Training is performed using multiple Random Forest configurations and tracked with MLflow.

```bash
python src/train.py
```

This step includes:
- Training multiple experiments
- Logging parameters and metrics (MAE, RMSE, R2)
- Saving the best model
- Automatically registering the best model to MLflow Model Registry


## Model Registry & Versioning (MLflow)

The best model from each training process is registered in MLflow Model Registry.

```bash
python src/register_model.py
```

Model lifecycle:
- New model -> Staging
- If better than Production -> Auto-promote to Production
- If not better -> Kept in Staging

The Production model is the active model used for inference.


## Active Model for Inference

The current model used for inference:

- Model Name: ```AQI_Predictor```
- Version: ```1```
- Stage: ```Production```
- MAE: 6.7158

Reason: This model has the best performance (lowest MAE) compared to other versions.


## Model Inference

### Using Python Script

```bash
python src/load_model.py
```

### Using Prediction Script

```bash
python src/predict.py
```

### Using REST API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pm1":15,"pm25":25,"relativehumidity":65,"temperature":28,"um003":500}'
```


## Data Versioning with DVC

DVC is used to track dataset changes.

```bash
dvc add data/raw/air_quality.csv
git add data/raw/air_quality.csv.dvc
git commit -m "Update dataset"
```


## Model-Data Lineage

```yaml
model:
  name: AQI_Predictor
  version: 1
  stage: Production
  run_id: ab082927
  dataset_version: v3
```


## Docker Compose Orchestration

### Architecture

```
+-----------------------------------------------+
|         Docker Network (aqi-network)          |
|                                               |
|  +------------------+   +------------------+  |
|  |   API Service    |   |  MLflow Server   |  |
|  |   (FastAPI)      +---+  (Tracking)      |  |
|  |   Port: 8000     |   |  Port: 5000      |  |
|  +------------------+   +------------------+  |
|                                               |
|  +------------------+   +------------------+  |
|  |   API Replica 2  |   |   API Replica 3  |  |
|  |   (FastAPI)      |   |   (FastAPI)      |  |
|  +------------------+   +------------------+  |
|                                               |
|  +------------------+   +------------------+  |
|  |  Model Artifacts |   |  MLflow Volume   |  |
|  +------------------+   +------------------+  |
+-----------------------------------------------+
```

### Services

| Service       | Container Name | Port | Description                    |
|---------------|---------------|------|---------------------------------|
| API Service   | aqi-api-N     | 8000 | REST API (3 replicas)           |
| MLflow Server | mlflow-server | 5000 | Tracking & model registry       |


### Quick Start

```bash
docker compose up -d --scale aqi-api=3
docker compose ps
docker compose logs -f
docker compose down
```


## Horizontal Scaling

### Run 3 API replicas

```bash
docker compose up -d --scale aqi-api=3
```

### Check replicas

```bash
docker ps --filter "name=aqi-api"
```

### Scale up dynamically

```bash
docker compose up -d --scale aqi-api=5
```

### Scale down

```bash
docker compose up -d --scale aqi-api=2
```

### Verify scaling

```bash
docker ps --filter "name=aqi-api" --format "table {{.Names}}\t{{.Status}}"
docker exec mlops-air_quality_prediction-aqi-api-1 curl -s http://localhost:8000/health
docker exec mlops-air_quality_prediction-aqi-api-1 curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pm1":15,"pm25":25,"relativehumidity":65,"temperature":28,"um003":500}'
```


## API Endpoints

| Method | Endpoint       | Description                     |
|--------|----------------|---------------------------------|
| GET    | /              | Root endpoint                   |
| GET    | /health        | Health check                    |
| POST   | /predict       | Single prediction               |
| POST   | /predict/batch | Batch prediction                |
| GET    | /model/info    | Model info                      |
| GET    | /latest-data   | Latest data                     |
| GET    | /docs          | Swagger documentation           |


## Testing the API

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "mlflow_connected": true,
    "timestamp": "2026-05-28T15:53:31"
}
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pm1":15.0,"pm25":25.0,"relativehumidity":65.0,"temperature":28.0,"um003":500.0}'
```

Response:
```json
{
    "predicted_pm25": 21.93,
    "predicted_aqi": 71.67,
    "aqi_category": "Moderate",
    "recommendation": "Kualitas udara cukup aman, tetapi kelompok sensitif disarankan mengurangi aktivitas luar ruangan yang terlalu lama."
}
```


## MLflow UI

Open in browser: ```http://localhost:5000```


## GitHub Actions CI/CD

Automated pipeline runs every 6 hours:
1. Data ingestion from OpenAQ API
2. Preprocessing
3. Training & evaluation
4. Model registration (auto-promote if better)
5. Commit updated model & data

Manual trigger: Via GitHub Actions tab -> Run workflow


## Data Pipeline Workflow

1. Fetch data from OpenAQ API
2. Store raw data
3. Version data using DVC
4. Preprocess data
5. Train model & track experiments (MLflow)
6. Register model (Model Registry)
7. Auto-promote to Production if better
8. Deploy as API service with 3 replicas
9. Perform inference via REST API


## Key Features

- API-based dynamic data ingestion
- Dataset versioning with DVC
- Experiment tracking with MLflow
- Model versioning and registry
- Model lifecycle management (Staging -> Production)
- Auto-promote model based on performance
- Reproducible ML pipeline
- Containerized API service with Docker
- Multi-service orchestration with Docker Compose
- Horizontal scaling with replicas
- Persistent storage with Docker volumes
- Health monitoring
- CI/CD with GitHub Actions
- Streamlit UI for public access


## Author

**Faiz Habibina Umiyabi**  
235150200111045