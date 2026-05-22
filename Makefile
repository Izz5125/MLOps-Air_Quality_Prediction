.PHONY: help build up down ps logs train predict clean

help:
	@echo "AQI MLOps Pipeline Commands:"
	@echo "  make build     - Build Docker images"
	@echo "  make up        - Start all services"
	@echo "  make down      - Stop all services"
	@echo "  make ps        - Check container status"
	@echo "  make logs      - View all logs"
	@echo "  make train     - Run training pipeline"
	@echo "  make predict   - Test API prediction"
	@echo "  make clean     - Clean all data"

build:
	docker compose build

up:
	docker compose up -d
	@echo "Waiting for services..."
	@sleep 5
	docker compose ps

down:
	docker compose down

ps:
	docker compose ps

logs:
	docker compose logs --tail=50

train:
	python src/ingest_data.py
	python src/preprocess.py
	python src/train.py
	python src/register_model.py

predict:
	curl -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"pm1":15,"pm25":25,"relativehumidity":65,"temperature":28,"um003":500}' | python3 -m json.tool

health:
	curl -s http://localhost:8000/health | python3 -m json.tool

clean:
	docker compose down -v
	rm -rf mlruns/ mlflow.db models/best_model.pkl
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
