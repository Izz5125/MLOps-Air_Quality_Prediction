import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="AQI Predictor - Malang", page_icon="🌫️", layout="wide")

WIB = timezone(timedelta(hours=7))

def get_latest_data():
    model = load_model()
    df = load_data()
    if model is None or df is None:
        return None
    
    latest = df.iloc[-1:]
    features = ['pm1','pm25','relativehumidity','temperature','um003','hour','day','month','day_of_week','pm25_rolling','pm25_lag1']
    pred_pm25 = model.predict(latest[features])[0]
    pred_aqi = pm25_to_aqi(pred_pm25)
    pred_cat, _ = get_aqi_info(pred_aqi)
    
    return {
        "latest_data": latest.to_dict('records')[0],
        "prediction": {
            "predicted_pm25": float(pred_pm25),
            "predicted_aqi": float(pred_aqi),
            "aqi_category": pred_cat,
            "recommendation": get_recommendation(pred_cat)
        }
    }

@st.cache_resource
def load_model():
    model_path = os.getenv('MODEL_PATH', 'models/best_model.pkl')
    if not os.path.exists(model_path):
        for p in ['best_model.pkl', './models/best_model.pkl']:
            if os.path.exists(p):
                model_path = p
                break
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_data():
    data_path = os.getenv('DATA_PATH', 'data/processed/processed_data.csv')
    if not os.path.exists(data_path):
        # Coba fallback paths
        for p in ['processed_data.csv', './data/processed/processed_data.csv', 'processed_data.csv']:
            if os.path.exists(p):
                data_path = p
                break
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        return df
    return None

def predict_manual(data):
    model = load_model()
    if model is None:
        return None
    features = ['pm1','pm25','relativehumidity','temperature','um003','hour','day','month','day_of_week','pm25_rolling','pm25_lag1']
    # Isi missing features
    for f in ['hour','day','month','day_of_week','pm25_rolling','pm25_lag1']:
        if f not in data:
            data[f] = 0
    df = pd.DataFrame([data])[features]
    pred_pm25 = model.predict(df)[0]
    pred_aqi = pm25_to_aqi(pred_pm25)
    pred_cat, _ = get_aqi_info(pred_aqi)
    return {
        "predicted_pm25": float(pred_pm25),
        "predicted_aqi": float(pred_aqi),
        "aqi_category": pred_cat,
        "recommendation": get_recommendation(pred_cat)
    }

def get_aqi_info(aqi):
    if aqi <= 50: return "Good", "🟢"
    elif aqi <= 100: return "Moderate", "🟡"
    elif aqi <= 150: return "Unhealthy (Sensitive)", "🟠"
    elif aqi <= 200: return "Unhealthy", "🔴"
    elif aqi <= 300: return "Very Unhealthy", "🟣"
    else: return "Hazardous", "⚫"


def get_recommendation(cat):
    rec = {
        "Good": "Udara sehat, aman beraktivitas di luar.",
        "Moderate": "Udara cukup aman, kelompok sensitif kurangi aktivitas luar.",
        "Unhealthy (Sensitive)": "Kelompok sensitif disarankan pakai masker.",
        "Unhealthy": "Udara kurang sehat, disarankan pakai masker.",
        "Very Unhealthy": "Udara sangat buruk, hindari aktivitas luar.",
        "Hazardous": "Udara berbahaya! Jangan keluar rumah."
    }
    return rec.get(cat, "")

def pm25_to_aqi(pm25):
    if pm25 <= 12: return (50/12)*pm25
    elif pm25 <= 35.4: return ((100-51)/(35.4-12.1))*(pm25-12.1)+51
    elif pm25 <= 55.4: return ((150-101)/(55.4-35.5))*(pm25-35.5)+101
    elif pm25 <= 150.4: return ((200-151)/(150.4-55.5))*(pm25-55.5)+151
    elif pm25 <= 250.4: return ((300-201)/(250.4-150.5))*(pm25-150.5)+201
    else: return 300

st.title("Air Quality Index Predictor - Malang")

# Status bar
model = load_model()
if model:
    st.success("Sistem Aktif | Model: Production")
else:
    st.error("Model tidak ditemukan")

st.divider()
latest = get_latest_data()

# ============================================
# ROW 1: DATA SAAT INI
# ============================================
if latest and "latest_data" in latest:
    data = latest["latest_data"]
    
    data_time = data.get("datetime", None)
    if data_time:
        try:
            dt_utc = pd.to_datetime(data_time)
            if dt_utc.tz is None:
                dt_utc = dt_utc.tz_localize('UTC')
            dt_wib = dt_utc.tz_convert(WIB)
            time_str = dt_wib.strftime('%d %B %Y, %H:%M WIB')
        except:
            time_str = str(data_time)
    else:
        time_str = "Tidak diketahui"
    
    current_pm25 = data.get('pm25', 0)
    current_aqi = pm25_to_aqi(current_pm25)
    current_cat, current_emoji = get_aqi_info(current_aqi)
    
    st.subheader(f"Data Terbaru ({time_str})")
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("PM2.5", f"{current_pm25:.1f} µg/m³")
    with c2:
        st.metric("PM1.0", f"{data.get('pm1', 0):.1f} µg/m³")
    with c3:
        st.metric("AQI", f"{current_aqi:.0f}", help=f"{current_emoji} {current_cat}")
    with c4:
        st.metric("Humidity", f"{data.get('relativehumidity', 0):.0f}%")
    with c5:
        st.metric("Temp", f"{data.get('temperature', 0):.1f}°C")
    with c6:
        st.metric("Status", f"{current_emoji} {current_cat}")

else:
    st.warning("Data tidak tersedia")

st.divider()

# ============================================
# ROW 2: PREDIKSI
# ============================================
if latest and "prediction" in latest:
    pred = latest["prediction"]
    
    if "error" not in pred:
        pred_pm25 = pred['predicted_pm25']
        pred_aqi = pred['predicted_aqi']
        pred_cat = pred['aqi_category']
        
        emoji_map = {"Good": "🟢", "Moderate": "🟡", "Unhealthy for Sensitive Groups": "🟠", "Unhealthy": "🔴", "Very Unhealthy": "🟣", "Hazardous": "⚫"}
        pred_emoji = emoji_map.get(pred_cat, "❓")
        
        st.subheader("Prediksi 1 Jam ke Depan")
        
        p1, p2, p3 = st.columns(3)
        with p1:
            delta = pred_pm25 - current_pm25 if latest else 0
            st.metric("PM2.5", f"{pred_pm25:.1f} µg/m³", delta=f"{delta:+.1f}")
        with p2:
            st.metric("AQI", f"{pred_aqi:.0f}", help=f"{pred_emoji} {pred_cat}")
        with p3:
            st.metric("Kategori", f"{pred_emoji} {pred_cat}")
        
        st.info(f"{pred['recommendation']}")
    else:
        st.warning("Prediksi gagal")
else:
    st.warning("Prediksi tidak tersedia")

st.divider()

# ============================================
# ROW 3: INPUT MANUAL
# ============================================
with st.expander("Prediksi Manual", expanded=False):
    with st.form("manual_form"):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            pm1 = st.number_input("PM1.0", 0.0, 500.0, 15.0)
        with c2:
            pm25 = st.number_input("PM2.5", 0.0, 500.0, 25.0)
        with c3:
            hum = st.number_input("Humidity %", 0.0, 100.0, 65.0)
        with c4:
            temp = st.number_input("Temp °C", 0.0, 50.0, 28.0)
        with c5:
            um = st.number_input("Ultrafine", 0.0, 5000.0, 500.0)
        
        if st.form_submit_button("Predict"):
            result = predict_manual({"pm1": pm1, "pm25": pm25, "relativehumidity": hum, "temperature": temp, "um003": um})
            if result:
                st.success(f"PM2.5: {result['predicted_pm25']:.1f} | AQI: {result['predicted_aqi']:.0f} | {result['aqi_category']}")
                st.info(result['recommendation'])
            else:
                st.error("Gagal memprediksi")

st.divider()
st.caption(f"MLOps Air Quality Prediction - Malang | {datetime.now(WIB).strftime('%Y-%m-%d %H:%M WIB')}")