# MLOps Air Quality Prediction – Malang

## Project Overview

This project aims to build a machine learning system to predict the daily Air Quality Index (AQI) in Malang using real-time air quality sensor data (PM2.5).

The system is designed following MLOps principles, enabling:
- Continuous data ingestion from external APIs
- Automated data preprocessing
- Support for continuous training (continual learning)
- Scalable and reproducible pipeline

---

## Project Structure

```
project/
│
├── data/
│   ├── raw/          # Raw data from API (timestamped)
│   ├── processed/    # Cleaned and ready-to-use dataset
│
├── src/
│   ├── ingest_data.py     # Data ingestion script
│   ├── preprocess.py      # Data preprocessing script
│
├── models/           # Trained machine learning models
├── notebooks/        # Exploratory data analysis
├── config/           # Configuration files
```

---

## Data Source

Data is collected from the OpenAQ API, which provides real-time air quality measurements.

- Parameter: PM2.5
- Format: JSON
- Update frequency: ~1 hour
- Data type: Time-series

---

## Development Environment

This project uses GitHub Codespaces for a reproducible development setup.

Main dependencies:
- Python 3.10
- Pandas
- Scikit-learn
- Requests
- Jupyter Notebook

---

## Running the Data Pipeline

### 1. Data Ingestion

Fetch latest data from OpenAQ API:

```bash
python src/ingest_data.py
```

Output:
- Saved automatically to `data/raw/`
- File uses timestamp to avoid overwriting

---

### 2. Data Preprocessing

Clean and prepare data:

```bash
python src/preprocess.py
```

Output:
- Saved to `data/processed/processed_data.csv`

---

## Data Pipeline Workflow

1. Fetch data from OpenAQ API (dynamic source)
2. Store raw data with timestamp (data versioning)
3. Clean data:
   - Remove missing values
   - Remove duplicates
   - Convert data types
4. Feature engineering:
   - Hour
   - Day
   - Rolling average
5. Save processed dataset for training

---

## Key Features

- Dynamic data ingestion (API-based)
- Non-destructive storage (timestamped files)
- Automated preprocessing
- Ready for continuous training pipeline

---

## Author

Faiz Habibina Umiyabi  
Informatics Engineering – Universitas Brawijaya