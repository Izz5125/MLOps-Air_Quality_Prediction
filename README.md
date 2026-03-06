# MLOps Air Quality Prediction – Malang

## Project Overview

This project aims to build an AI system capable of predicting daily Air Quality Index (AQI) in Malang using sensor-based air quality data.

The system is designed using MLOps principles to support continuous data ingestion, automated model retraining, and monitoring to ensure long-term model reliability.

## Project Structure
data/
raw/ → raw sensor data
processed/ → cleaned dataset

models/
trained ML models

notebooks/
exploratory analysis and experiments

src/
data → data ingestion
features → feature engineering
models → model training

config/
configuration files


## Development Environment

This project uses **GitHub Codespaces** to provide a reproducible development environment.

The environment includes:

- Python 3.10
- Pandas
- Scikit-learn
- Jupyter
- Requests

## Running the Project

1. Open repository in Codespaces
Code → Codespaces → Create Codespace


2. Install dependencies
pip install -r requirements.txt


3. Start working with notebooks or scripts
notebooks/
src/


## Branching Strategy

This project follows **GitHub Flow**:

- main → stable version
- feat/* → experimental features

Example:
feat/initial-eda


All changes are merged through Pull Requests after validation.

## Author

Faiz Habibina Umiyabi  
Informatics Engineering – Universitas Brawijaya