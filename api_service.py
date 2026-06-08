"""
AQI Prediction API Service
Menyediakan endpoint untuk prediksi kualitas udara
Terintegrasi dengan MLflow untuk model tracking
"""

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
import os
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import mlflow
import os
from datetime import datetime
from typing import List, Optional, Dict
import sys

# Tambahkan src ke path
sys.path.append('/app/src')
sys.path.append('./src')

# Import fungsi AQI
try:
    from aqi import pm25_to_aqi, classify_aqi, get_recommendation
    print("✅ AQI module loaded successfully")
except ImportError as e:
    print(f"⚠️  Could not import AQI module: {e}")
    # Fallback functions
    def pm25_to_aqi(pm25):
        if pm25 <= 12:
            return (50 / 12) * pm25
        elif pm25 <= 35.4:
            return ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
        elif pm25 <= 55.4:
            return ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
        elif pm25 <= 150.4:
            return ((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151
        else:
            return 300

    def classify_aqi(aqi):
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    def get_recommendation(aqi_status):
        recommendations = {
            "Good": "Udara sedang baik. Cocok untuk beraktivitas di luar ruangan.",
            "Moderate": "Kualitas udara cukup aman, kelompok sensitif disarankan mengurangi aktivitas luar ruangan.",
            "Unhealthy for Sensitive Groups": "Kelompok sensitif disarankan memakai masker dan membatasi aktivitas di luar.",
            "Unhealthy": "Udara kurang sehat. Disarankan memakai masker saat keluar rumah.",
            "Very Unhealthy": "Udara sangat buruk. Hindari aktivitas di luar ruangan.",
            "Hazardous": "Udara berbahaya! Sangat tidak disarankan keluar rumah."
        }
        return recommendations.get(aqi_status, "Tidak ada rekomendasi tersedia.")

# Konfigurasi
# Prometheus Metrics
PREDICTION_COUNT = Counter('aqi_predictions_total', 'Total predictions')
PREDICTION_LATENCY = Histogram('aqi_prediction_latency_seconds', 'Prediction latency')
REQUEST_COUNT = Counter('aqi_requests_total', 'Total requests', ['endpoint'])
PREDICTION_SCORE = Gauge('aqi_predicted_score', 'Predicted AQI score')
MODEL_STATUS = Gauge('aqi_model_loaded', 'Model loaded status')
INPUT_PM25_AVG = Gauge('aqi_input_pm25_avg', 'Average PM2.5 input over last requests')

app = FastAPI(
    title="AQI Prediction API",
    description="API untuk prediksi Air Quality Index (AQI)",
    version="1.0.0"
)

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000')
MODEL_NAME = os.getenv('MODEL_NAME', 'AQI_Predictor')
MODEL_STAGE = os.getenv('MODEL_STAGE', 'Staging')
MODEL_PATH = os.getenv('MODEL_PATH', 'models/best_model.pkl')
DATA_PATH = os.getenv('DATA_PATH', 'data/processed/processed_data.csv')

FEATURE_COLUMNS = [
    "pm1", "pm25", "relativehumidity", "temperature", "um003",
    "hour", "day", "month", "day_of_week",
    "pm25_rolling", "pm25_lag1"
]

model = None
model_source = None

print("=" * 50)
print("AQI Prediction API Configuration:")
print(f"  MLflow URI: {MLFLOW_TRACKING_URI}")
print(f"  Model: {MODEL_NAME}/{MODEL_STAGE}")
print(f"  Model Path: {MODEL_PATH}")
print(f"  Data Path: {DATA_PATH}")
print("=" * 50)

# ===========================================
# Pydantic Models
# ===========================================
class SensorData(BaseModel):
    pm1: float = Field(..., description="PM1.0 (ug/m3)")
    pm25: float = Field(..., description="PM2.5 (ug/m3)")
    relativehumidity: float = Field(..., description="Humidity (%)")
    temperature: float = Field(..., description="Temperature (C)")
    um003: float = Field(..., description="Ultrafine particles")
    hour: Optional[int] = Field(None, description="Hour (0-23)")
    day: Optional[int] = Field(None, description="Day")
    month: Optional[int] = Field(None, description="Month")
    day_of_week: Optional[int] = Field(None, description="Day of week")
    pm25_rolling: Optional[float] = Field(None, description="PM2.5 rolling avg")
    pm25_lag1: Optional[float] = Field(None, description="PM2.5 lag 1h")


class BatchSensorData(BaseModel):
    data: List[SensorData]


class PredictionResponse(BaseModel):
    predicted_pm25: float
    predicted_aqi: float
    aqi_category: str
    recommendation: str
    model_source: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str
    mlflow_connected: bool
    timestamp: str

# ===========================================
# Model Loading Functions
# ===========================================
def load_model_from_mlflow():
    """Load model from MLflow Registry with timeout"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        print(f"🔄 Trying MLflow: {model_uri}")
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Model loaded from MLflow")
        return loaded_model, "mlflow_registry"
    except Exception as e:
        print(f"⚠️  MLflow not available: {str(e)[:80]}")
        return None, None


def load_model_from_file():
    """Load model from local file"""
    try:
        # Coba beberapa path
        paths_to_try = [
            MODEL_PATH,
            '/app/' + MODEL_PATH,
            'models/best_model.pkl'
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                print(f"📂 Loading from: {path}")
                loaded_model = joblib.load(path)
                print(f"✅ Model loaded from file")
                return loaded_model, "local_file"
        
        print(f"⚠️  Model file not found in any path")
        return None, None
    except Exception as e:
        print(f"⚠️  File load failed: {str(e)[:80]}")
        return None, None


def load_model():
    """Load model with priority: Local file -> MLflow"""
    global model, model_source
    
    print("🔄 Loading model...")
    
    # Coba local file dulu (lebih cepat)
    model, model_source = load_model_from_file()
    
    # Jika tidak ada, coba MLflow
    if model is None:
        print("📡 Trying MLflow...")
        model, model_source = load_model_from_mlflow()
    
    if model is not None:
        print(f"🎯 Model ready! (source: {model_source})")
        return True
    else:
        print("❌ No model available! Train model first: python src/train.py")
        return False


def prepare_features(data: SensorData, current_time: datetime = None) -> pd.DataFrame:
    """Prepare features DataFrame from sensor input"""
    if current_time is None:
        current_time = datetime.now()
    
    features_dict = {
        "pm1": data.pm1,
        "pm25": data.pm25,
        "relativehumidity": data.relativehumidity,
        "temperature": data.temperature,
        "um003": data.um003,
        "hour": data.hour if data.hour is not None else current_time.hour,
        "day": data.day if data.day is not None else current_time.day,
        "month": data.month if data.month is not None else current_time.month,
        "day_of_week": data.day_of_week if data.day_of_week is not None else current_time.weekday(),
        "pm25_rolling": data.pm25_rolling if data.pm25_rolling is not None else data.pm25,
        "pm25_lag1": data.pm25_lag1 if data.pm25_lag1 is not None else data.pm25
    }
    
    df = pd.DataFrame([features_dict])
    return df[FEATURE_COLUMNS]


def make_prediction(features_df: pd.DataFrame) -> float:
    """Make PM2.5 prediction"""
    if model is None:
        raise ValueError("Model is not loaded")
    
    prediction = model.predict(features_df)
    
    if hasattr(prediction, 'flatten'):
        return float(prediction.flatten()[0])
    elif isinstance(prediction, (list, pd.Series)):
        return float(prediction[0])
    else:
        return float(prediction)


# ===========================================
# FastAPI Startup
# ===========================================
@app.on_event("startup")
async def startup_event():
    """Run on API startup"""
    print("🚀 Starting AQI Prediction API...")
    load_model()


# ===========================================
# API Endpoints
# ===========================================
@app.get("/")
async def root():
    return {
        "service": "AQI Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    mlflow_connected = False
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        client.search_registered_models(max_results=1)
        mlflow_connected = True
    except:
        pass
    
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_source=model_source if model_source else "none",
        mlflow_connected=mlflow_connected,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(data: SensorData):
    REQUEST_COUNT.labels(endpoint='/predict').inc()
    start_time = time.time()
    """Predict AQI for single sensor data"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")
    
    try:
        features_df = prepare_features(data)
        predicted_pm25 = make_prediction(features_df)
        predicted_aqi = pm25_to_aqi(predicted_pm25)
        aqi_category = classify_aqi(predicted_aqi)
        recommendation = get_recommendation(aqi_category)
        
        PREDICTION_COUNT.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        PREDICTION_SCORE.set(predicted_aqi)
        MODEL_STATUS.set(1)
        
        return PredictionResponse(
            predicted_pm25=round(predicted_pm25, 2),
            predicted_aqi=round(predicted_aqi, 2),
            aqi_category=aqi_category,
            recommendation=recommendation,
            model_source=model_source if model_source else "unknown",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(data: BatchSensorData):
    """Predict AQI for multiple data"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for item in data.data:
        try:
            features_df = prepare_features(item)
            predicted_pm25 = make_prediction(features_df)
            predicted_aqi = pm25_to_aqi(predicted_pm25)
            aqi_category = classify_aqi(predicted_aqi)
            recommendation = get_recommendation(aqi_category)
            
            results.append(PredictionResponse(
                predicted_pm25=round(predicted_pm25, 2),
                predicted_aqi=round(predicted_aqi, 2),
                aqi_category=aqi_category,
                recommendation=recommendation,
                model_source=model_source if model_source else "unknown",
                timestamp=datetime.now().isoformat()
            ))
        except Exception as e:
            results.append(PredictionResponse(
                predicted_pm25=0.0,
                predicted_aqi=0.0,
                aqi_category="error",
                recommendation=f"Error: {str(e)[:50]}",
                model_source=model_source if model_source else "unknown",
                timestamp=datetime.now().isoformat()
            ))
    
    return results


@app.get("/model/info")
async def model_info():
    """Get model information"""
    info = {
        "model_name": MODEL_NAME,
        "current_stage": MODEL_STAGE,
        "model_source": model_source,
        "model_loaded": model is not None,
        "versions": []
    }
    
    # Try to get MLflow info
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        for mv in versions:
            info["versions"].append({
                "version": mv.version,
                "stage": mv.current_stage,
                "run_id": mv.run_id
            })
    except:
        info["mlflow_status"] = "not_available"
    
    return info


@app.get("/latest-data")
async def get_latest_data():
    """Get latest data and predict"""
    if not os.path.exists(DATA_PATH):
        return {"error": f"Data not found at {DATA_PATH}", "prediction": None}
    
    df = pd.read_csv(DATA_PATH)
    latest_row = df.iloc[-1:]
    latest_features = latest_row[FEATURE_COLUMNS]
    
    # Ambil datetime untuk ditampilkan
    latest_data_dict = latest_features.to_dict('records')[0]
    if 'datetime' in df.columns:
        latest_data_dict['datetime'] = str(latest_row['datetime'].values[0])
    
    result = {"latest_data": latest_data_dict, "prediction": None}
    
    if model is not None:
        try:
            predicted_pm25 = make_prediction(latest_features)
            predicted_aqi = pm25_to_aqi(predicted_pm25)
            result["prediction"] = {
                "predicted_pm25": round(predicted_pm25, 2),
                "predicted_aqi": round(predicted_aqi, 2),
                "aqi_category": classify_aqi(predicted_aqi),
                "recommendation": get_recommendation(classify_aqi(predicted_aqi))
            }
        except Exception as e:
            result["prediction"] = {"error": str(e)}
    
    return result


@app.get("/metrics")
async def metrics():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("Starting AQI Prediction API Server...")
    print("="*50 + "\n")
    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
