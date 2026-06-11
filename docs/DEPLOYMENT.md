# Deployment & Operations Runbook

This document provides operational steps and best practices for deploying and operating the MLOps Air Quality Prediction stack.

## Overview
- Services: FastAPI (API), MLflow (tracking & registry)
- Orchestration: Docker Compose (prod/dev) or Kubernetes (recommended for larger environments)
- Storage: DVC remote (S3/GCS), MLflow artifact store (volume or remote), persistent DB for metadata if needed

## Environment variables
- `MLFLOW_TRACKING_URI` — MLflow tracking server address (e.g. http://mlflow:5000)
- `DVC_REMOTE` — DVC remote name and configuration
- `AZURE_STORAGE_ACCOUNT` / `AWS_*` — credentials for cloud remotes when needed

## Docker Compose (recommended quick deployment)
1) Configure environment variables in `.env` or in the compose file.
2) Start services:

```bash
docker compose up -d --scale aqi-api=3
```

3) Check service health:

```bash
docker compose ps
docker compose logs -f aqi-api
```

4) Stop & cleanup:

```bash
docker compose down
```

## MLflow setup
- By default the repository expects MLflow accessible at `http://localhost:5000` when running locally.
- For production, point `MLFLOW_TRACKING_URI` to a dedicated MLflow server and ensure artifact storage is durable (S3 or mounted volume).

## DVC setup
- Configure a remote storage for datasets:

```bash
# example: s3 remote
dvc remote add -d s3remote s3://my-bucket/mlops-air-quality
# configure credentials in environment or CI secret
```

- Push/pull data in CI/CD jobs as needed:

```bash
dvc push
# in CI runner
dvc pull
```

## Model promotion policy
- `src/register_model.py` implements an automatic promotion heuristic:
  - If the candidate model MAE is lower than current production MAE, promote to `Production`.
  - Otherwise, keep in `Staging` for manual review.

## CI/CD hints (GitHub Actions)
- Secrets required: cloud credentials (S3/GCS), MLflow URL (if external), DVC remote credentials.
- Workflow steps:
  - Checkout
  - Setup Python env
  - dvc pull (if needed)
  - Run tests
  - Run training & evaluation (optionally in matrix)
  - Run `src/register_model.py` to register/promote model

## Health checks & Monitoring
- API: `/health` endpoint should return readiness and model status.
- MLflow: check UI on its service port.
- Add Prometheus exporters if needed and configure alerting rules in `alert.rules.yml` and `prometheus.yml` provided in repo.

## Backups & Persistence
- MLflow artifacts: persist using remote artifact store (S3/GCS) or bind-mounted volumes in Docker Compose.
- DVC data: ensure `dvc push` runs in CI after ingest/preprocess to persist datasets.

## Troubleshooting
- MLflow `connection refused`: verify `MLFLOW_TRACKING_URI`, firewall, and that the MLflow server is running.
- DVC permission errors: verify credentials and bucket/prefix permissions.
- Model load errors: confirm model artifact exists in `mlruns/` and `models/` and versions match the code.

## Rollback
- Use MLflow Model Registry to revert to previous `Production` version.
- Use `dvc checkout` to revert data files to previous commits if dataset changes caused issues.

## Security
- Never commit secrets to repo. Use CI secrets and `.env` files excluded from VCS.
- Limit MLflow write access to CI worker accounts.

## Contact
- For operational incidents contact the maintainer: Faiz Habibina Umiyabi
