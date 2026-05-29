#!/bin/bash
set -e

echo "========================================"
echo "  AQI MLOps - Full Pipeline Runner"
echo "========================================"

# Step 1: Ingest data
echo -e "\n[1/5] Ingesting data..."
python src/ingest_data.py --max-pages 3
echo "✅ Done"

# Step 2: Preprocessing
echo -e "\n[2/5] Preprocessing..."
python src/preprocess.py
echo "✅ Done"

# Step 3: Training
echo -e "\n[3/5] Training model..."
python src/train.py
echo "✅ Done"

# Step 4: Register model
echo -e "\n[4/5] Registering model..."
python src/register_model.py
echo "✅ Done"

# Step 5: Deploy with Docker Compose
echo -e "\n[5/5] Deploying services..."
docker compose down 2>/dev/null
docker compose up -d --scale aqi-api=3
sleep 15
echo "✅ Done"

# Status
echo -e "\n========================================"
echo "  Status"
echo "========================================"
docker compose ps

echo -e "\n========================================"
echo "  Services"
echo "========================================"
echo "API:      http://localhost:8000"
echo "MLflow:   http://localhost:5000"
echo "Prometheus: http://localhost:9090"
echo "Grafana:  http://localhost:3000 (admin/admin)"
echo "Streamlit UI: https://aqi-predictor.streamlit.app"
echo "========================================"
