# MLOps Air Quality Prediction – Malang

## Project Overview

This project aims to build a machine learning system to predict the daily Air Quality Index (AQI) in Malang using real-time air quality sensor data (PM2.5).

The system is designed following MLOps principles, enabling:
- Continuous data ingestion from external APIs  
- Automated data preprocessing  
- Model training and experiment tracking  
- Model versioning and lifecycle management  
- Scalable and reproducible pipeline  
- Data versioning using DVC  

---

## Project Structure

```
project/
│
├── data/
│   ├── raw/          
│   ├── processed/    
│
├── src/
│   ├── ingest_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── load_model.py
│
├── models/
├── notebooks/
├── config/
│
├── model_registry.yaml
├── .dvc/             
├── *.dvc             
```

---

## Data Source

Data is collected from the OpenAQ API.

- Parameter: PM2.5  
- Format: JSON  
- Update frequency: ~1 hour  
- Data type: Time-series  

---

## Development Environment

This project uses GitHub Codespaces.

Main dependencies:
- Python 3.10  
- Pandas  
- Scikit-learn  
- MLflow  
- Requests  
- DVC  

---

## Running the Data Pipeline

### 1. Data Ingestion

```bash
python src/ingest_data.py
```

Output:
- Stored in `data/raw/`

---

### 2. Data Preprocessing

```bash
python src/preprocess.py
```

Output:
- `data/processed/processed_data.csv`

---

## Model Training & Experiment Tracking

Training is performed using multiple Random Forest configurations and tracked with MLflow.

```bash
python src/train.py
```

This step includes:
- Training multiple experiments  
- Logging parameters and metrics (MAE, RMSE, R²)  
- Saving the best model  
- Automatically registering the best model to MLflow Model Registry  

---

## Model Registry & Versioning (MLflow)

The best model from each training process is registered in MLflow Model Registry.

Example:
- Version 1 → Staging  
- Version 2 → Production  

The Production model is the active model used for inference.

---

## Active Model for Inference

The current model used for inference:

- Model Name: AQI_Predictor  
- Version: 2  
- Stage: Production  

Reason:
This model has the best performance (lowest MAE) compared to other versions and has passed evaluation.

---

## Model Inference

Run the following to use the Production model:

```bash
python src/load_model.py
```

This will:
- Load the model from MLflow Model Registry  
- Use the Production stage  
- Perform prediction on new input data  

---

## Data Versioning with DVC

DVC is used to track dataset changes.

### Track Dataset

```bash
dvc add data/raw/air_quality.csv
git add data/raw/air_quality.csv.dvc
git commit -m "Update dataset"
```

---

### Compare Dataset Versions

```bash
dvc diff HEAD~1 HEAD
```

---

## Model–Data Lineage

The relationship between model and dataset is stored in:

model_registry.yaml

Example:

```yaml
model:
  name: AQI_Predictor
  version: 2
  stage: Production
  run_id: <run_id>
  dataset_version: v3
```

This ensures:
- Model traceability to dataset version  
- Reproducibility of the pipeline  

---

## Data Pipeline Workflow

1. Fetch data from API  
2. Store raw data  
3. Version data using DVC  
4. Preprocess data  
5. Train model & track experiments (MLflow)  
6. Register model (Model Registry)  
7. Promote model to Production  
8. Use model for inference  

---

## Key Features

- API-based dynamic data ingestion  
- Dataset versioning with DVC  
- Experiment tracking with MLflow  
- Model versioning and registry  
- Model lifecycle management (Staging → Production)  
- Reproducible ML pipeline  

---

## Author

Faiz Habibina Umiyabi  
Informatics Engineering – Universitas Brawijaya  