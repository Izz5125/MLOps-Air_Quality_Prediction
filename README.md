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
+-- Dockerfile.api
+-- docker-compose.yaml
|
+-- model_registry.yaml
+-- model_registry.yaml.dvc
+-- requirements.txt
+-- commandlist.txt
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

Output:
- Stored in `data/raw/air_quality.csv`


### 2. Data Preprocessing

```bash
python src/preprocess.py
```

Output:
- `data/processed/processed_data.csv`


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

Example:
- Version 1 → Staging
- Version 2 → Production

The Production model is the active model used for inference.


## Active Model for Inference

The current model used for inference:

- Model Name: `AQI_Predictor`
- Version: `2`
- Stage: `Production`

Reason: This model has the best performance (lowest MAE) compared to other versions and has passed evaluation.


## Model Inference

### Using Python Script

```bash
python src/load_model.py
```

This will:
- Load the model from MLflow Model Registry
- Use the Production stage
- Perform prediction on new input data

### Using Prediction Script

```bash
python src/predict.py
```


## Data Versioning with DVC

DVC is used to track dataset changes.

### Track Dataset

```bash
dvc add data/raw/air_quality.csv
git add data/raw/air_quality.csv.dvc
git commit -m "Update dataset"
```

### Compare Dataset Versions

```bash
dvc diff HEAD~1 HEAD
```


## Model-Data Lineage

The relationship between model and dataset is stored in `model_registry.yaml`:

```yaml
model:
  name: AQI_Predictor
  version: 2
  stage: Production
  run_id: 44c4e07699d44b148254a2bf5d458324
  dataset_version: v3
```

This ensures:
- Model traceability to dataset version
- Reproducibility of the pipeline


## Docker Compose Orchestration

### Architecture

```
+--------------------------------------------------+
|           Docker Network (aqi-network)            |
|                                                  |
|  +------------------+   +------------------+      |
|  |   API Service    |   |  MLflow Server   |      |
|  |   (FastAPI)      +---+  (Tracking)      |      |
|  |   Port: 8000     |   |  Port: 5000      |      |
|  +--------+---------+   +--------+---------+      |
|           |                      |                |
|           v                      v                |
|  +------------------+   +------------------+      |
|  |  Model Artifacts |   |  SQLite Database |      |
|  |  (Volume Mount)  |   |  (Volume: mlflow)|      |
|  +------------------+   +------------------+      |
+--------------------------------------------------+
```

### Services

| Service       | Container Name    | Port | Description                     |
|---------------|------------------|------|---------------------------------|
| API Service   | aqi-api-service  | 8000 | REST API for AQI inference      |
| MLflow Server | mlflow-server    | 5000 | Tracking & model registry       |


### Quick Start

```bash
docker compose up -d
docker compose ps
docker compose logs -f
docker compose down
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

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pm1":15.0,"pm25":25.0,"relativehumidity":65.0,"temperature":28.0,"um003":500.0}'
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"pm1":10.0,"pm25":20.0,"relativehumidity":60.0,"temperature":26.0,"um003":400.0},
      {"pm1":35.0,"pm25":45.0,"relativehumidity":80.0,"temperature":30.0,"um003":800.0}
    ]
  }'
```


## MLflow UI

Open in browser:

```
http://localhost:5000
```


## Data Pipeline Workflow

1. Fetch data from OpenAQ API  
2. Store raw data  
3. Version data using DVC  
4. Preprocess data  
5. Train model & track experiments (MLflow)  
6. Register model (Model Registry)  
7. Promote model to Production  
8. Deploy as API service  
9. Perform inference via REST API  


## Key Features

- API-based dynamic data ingestion  
- Dataset versioning with DVC  
- Experiment tracking with MLflow  
- Model versioning and registry  
- Model lifecycle management (Staging → Production)  
- Reproducible ML pipeline  
- Containerized API service with Docker  
- Multi-service orchestration with Docker Compose  
- Persistent storage with Docker volumes  
- Health monitoring  


## Author

Faiz Habibina Umiyabi  
Informatics Engineering - Universitas Brawijaya