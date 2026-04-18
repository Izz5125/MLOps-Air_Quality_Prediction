# MLOps Air Quality Prediction – Malang

## Project Overview

This project aims to build a machine learning system to predict the daily Air Quality Index (AQI) in Malang using real-time air quality sensor data (PM2.5).

The system is designed following MLOps principles, enabling:
- Continuous data ingestion from external APIs  
- Automated data preprocessing  
- Support for continual learning  
- Scalable and reproducible pipeline  
- Data versioning using DVC  

---

## Project Structure

```
project/
│
├── data/
│   ├── raw/          # Raw data (tracked by DVC)
│   ├── processed/    # Cleaned dataset
│
├── src/
│   ├── ingest_data.py
│   ├── preprocess.py
│
├── models/
├── notebooks/
├── config/
│
├── .dvc/             # DVC configuration
├── *.dvc             # Dataset metadata
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
- Requests  
- DVC  

---

## Running the Data Pipeline

### 1. Data Ingestion

Fetch latest data from API:

```bash
python src/ingest_data.py
```

Output:
- Stored in `data/raw/`
- File uses timestamp

---

### 2. Data Preprocessing

Clean and prepare data:

```bash
python src/preprocess.py
```

Output:
- `data/processed/processed_data.csv`

---

## Data Versioning with DVC

DVC is used to manage dataset versions without storing large files directly in Git.

### Initialize DVC

```bash
dvc init
git add .
git commit -m "Initialize DVC"
```

---

### Track Initial Dataset (v1)

```bash
dvc add data/raw/air_quality.csv
git add data/raw/air_quality.csv.dvc
git commit -m "Dataset v1"
```

---

### Simulate Continual Learning

Fetch new data:

```bash
python src/ingest_data.py
```

Overwrite main dataset:

```bash
cp data/raw/<new_file>.csv data/raw/air_quality.csv
```

---

### Track Updated Dataset (v2)

```bash
dvc add data/raw/air_quality.csv
git add data/raw/air_quality.csv.dvc
git commit -m "Dataset v2"
```

---

### Compare Dataset Versions

```bash
dvc diff HEAD~1 HEAD
```

Example output:

```
Modified:
    data/raw/air_quality.csv
```

This indicates that the dataset content has changed.

---

## Data Pipeline Workflow

1. Fetch data from API  
2. Store raw data  
3. Update main dataset (`air_quality.csv`)  
4. Track changes using DVC  
5. Preprocess data  
6. Prepare for model training  

---

## Key Features

- API-based dynamic data ingestion  
- Dataset versioning with DVC  
- Efficient handling of large data  
- Reproducible data pipeline  
- Support for continual learning  

---

## Author

Faiz Habibina Umiyabi  
Informatics Engineering – Universitas Brawijaya  