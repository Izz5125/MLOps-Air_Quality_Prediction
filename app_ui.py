import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timezone
import time

st.set_page_config(
    page_title="AQI Predictor - Malang",
    page_icon="🌫️",
    layout="wide"
)

@st.cache_data(ttl=300)
def get_latest_data():
    try:
        response = requests.get("http://localhost:8000/latest-data", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

@st.cache_data(ttl=300)
def get_model_info():
    try:
        response = requests.get("http://localhost:8000/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def get_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def predict_manual(data):
    try:
        response = requests.post("http://localhost:8000/predict", json=data, timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def time_ago(timestamp_str):
    try:
        if not timestamp_str:
            return "tidak diketahui"
        data_time = pd.to_datetime(timestamp_str)
        now = pd.Timestamp.now(tz=data_time.tz)
        diff = now - data_time
        minutes = int(diff.total_seconds() / 60)
        hours = minutes // 60
        days = hours // 24
        if minutes < 1:
            return "baru saja"
        elif minutes < 60:
            return f"{minutes} menit yang lalu"
        elif hours < 24:
            return f"{hours} jam yang lalu"
        else:
            return f"{days} hari yang lalu"
    except:
        return "tidak diketahui"

st.title("🌫️ Air Quality Index Predictor - Malang")
st.markdown("Prediksi kualitas udara real-time menggunakan Machine Learning")

health = get_health()
col_status1, col_status2, col_status3 = st.columns(3)

if health and health.get("model_loaded"):
    with col_status1:
        st.success("🟢 API Aktif")
    with col_status2:
        st.success("🤖 Model Loaded")
    with col_status3:
        if health.get("mlflow_connected"):
            st.success("📊 MLflow Terhubung")
        else:
            st.warning("📊 MLflow Offline")
else:
    st.error("🔴 Sistem Offline - Jalankan docker compose up -d")

st.divider()
st.header("📡 Prediksi Kualitas Udara Saat Ini")

latest = get_latest_data()

if latest and "latest_data" in latest and latest["latest_data"]:
    data = latest["latest_data"]
    pred = latest.get("prediction", {})
    
    data_timestamp = data.get("datetime", None)
    
    time_col1, time_col2, time_col3 = st.columns(3)
    
    with time_col1:
        if data_timestamp:
            try:
                data_time = pd.to_datetime(data_timestamp)
                st.info(f"📅 Data Sensor: {data_time.strftime('%d %B %Y, %H:%M WIB')}")
            except:
                st.info(f"📅 Data Sensor: {data_timestamp}")
        else:
            st.warning("📅 Data Sensor: Tidak diketahui")
    
    with time_col2:
        ago = time_ago(data_timestamp)
        if "menit" in ago and int(ago.split()[0]) < 30:
            st.success(f"⏱️ Update: {ago}")
        elif "jam" in ago:
            st.warning(f"⏱️ Update: {ago}")
        else:
            st.error(f"⏱️ Update: {ago}")
    
    with time_col3:
        now = datetime.now().strftime("%d %B %Y, %H:%M WIB")
        st.info(f"🕐 Sekarang: {now}")
    
    if "menit" in ago and int(ago.split()[0]) < 30:
        st.success("✅ Data masih realtime (kurang dari 30 menit)")
    elif "jam" in ago and int(ago.split()[0]) < 2:
        st.warning(f"⚠️ Data cukup baru ({ago})")
    else:
        st.error(f"❌ Data sudah lama ({ago}). Perlu update data!")
        st.info("💡 Jalankan: python src/ingest_data.py untuk update data")
    
    st.divider()
    st.subheader("📊 Data Sensor Terkini")
    
    cols = st.columns(5)
    metrics_map = [
        ("PM1.0", "pm1", "ug/m3"),
        ("PM2.5", "pm25", "ug/m3"),
        ("Humidity", "relativehumidity", "%"),
        ("Temperature", "temperature", "C"),
        ("Ultrafine", "um003", "particles/cm3"),
    ]
    
    values = {}
    for i, (label, key, unit) in enumerate(metrics_map):
        val = data.get(key, "N/A")
        values[key] = val
        with cols[i]:
            st.metric(f"{label}", f"{val:.1f}" if isinstance(val, (int, float)) else val)
    
    st.divider()
    st.subheader("🔮 Hasil Prediksi AQI (1 Jam ke Depan)")
    
    if pred and "error" not in pred:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("PM2.5 Diprediksi", f"{pred['predicted_pm25']:.1f} ug/m3")
        with col2:
            aqi = pred['predicted_aqi']
            st.metric("Indeks AQI", f"{aqi:.0f}")
        with col3:
            cat = pred['aqi_category']
            emoji = {"Good": "🟢", "Moderate": "🟡", "Unhealthy for Sensitive Groups": "🟠", "Unhealthy": "🔴", "Very Unhealthy": "🟣", "Hazardous": "⚫"}
            st.metric("Kategori", f"{emoji.get(cat, '❓')} {cat}")
        
        st.markdown("**Level AQI:**")
        st.progress(min(aqi / 300, 1.0))
        st.info(f"💡 Rekomendasi: {pred['recommendation']}")
        
        st.subheader("📈 Perbandingan PM2.5 Saat Ini vs Prediksi")
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.metric("PM2.5 Saat Ini", f"{values.get('pm25', 0):.1f} ug/m3")
        with comp_col2:
            delta = pred['predicted_pm25'] - values.get('pm25', 0)
            st.metric("PM2.5 Prediksi (1 jam depan)", f"{pred['predicted_pm25']:.1f} ug/m3", 
                     delta=f"{delta:+.1f} ug/m3", delta_color="inverse")
        
        if abs(delta) < 5:
            st.success("✅ PM2.5 diprediksi stabil dalam 1 jam ke depan")
        elif delta > 10:
            st.error(f"⚠️ PM2.5 diprediksi naik signifikan (+{delta:.1f} ug/m3)")
        elif delta < -10:
            st.success(f"🟢 PM2.5 diprediksi turun signifikan ({delta:.1f} ug/m3)")
    
    elif pred and "error" in pred:
        st.warning(f"⚠️ Prediksi gagal: {pred['error']}")
    else:
        st.warning("⚠️ Belum ada data prediksi")

else:
    st.warning("⚠️ Tidak dapat mengambil data terbaru.")
    st.info("💡 Pastikan API berjalan: docker compose up -d")
    st.info("💡 Update data sensor: python src/ingest_data.py")

st.divider()

with st.expander("🔧 Prediksi Manual (Input Sendiri)", expanded=False):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Data Sensor")
        
        with st.form("manual_form"):
            sub1, sub2 = st.columns(2)
            with sub1:
                pm1 = st.number_input("PM1.0 (ug/m3)", 0.0, 500.0, 15.0, 0.1, key="m_pm1")
                pm25 = st.number_input("PM2.5 (ug/m3)", 0.0, 500.0, 25.0, 0.1, key="m_pm25")
                humidity = st.number_input("Humidity (%)", 0.0, 100.0, 65.0, 0.1, key="m_hum")
            with sub2:
                temp = st.number_input("Temperature (C)", 0.0, 50.0, 28.0, 0.1, key="m_temp")
                um003 = st.number_input("Ultrafine Particles", 0.0, 5000.0, 500.0, 0.1, key="m_um003")
                hour = st.number_input("Hour (0-23)", 0, 23, datetime.now().hour, key="m_hour")
            
            manual_submit = st.form_submit_button("🔮 Predict", use_container_width=True)
    
    with col2:
        st.subheader("📊 Hasil Prediksi Manual")
        
        if manual_submit:
            data = {
                "pm1": pm1, "pm25": pm25,
                "relativehumidity": humidity,
                "temperature": temp, "um003": um003, "hour": hour
            }
            result = predict_manual(data)
            
            if result:
                st.success("✅ Prediksi Berhasil!")
                st.metric("PM2.5", f"{result['predicted_pm25']:.1f} ug/m3")
                st.metric("AQI", f"{result['predicted_aqi']:.0f}")
                st.metric("Kategori", result['aqi_category'])
                st.info(f"💡 {result['recommendation']}")
            else:
                st.error("❌ Gagal memprediksi")
        else:
            st.info("👈 Isi data dan klik Predict")

with st.sidebar:
    st.header("📊 Informasi Model")
    info = get_model_info()
    if info:
        st.success("✅ Model Terhubung")
        st.write(f"Nama: {info.get('model_name', 'AQI_Predictor')}")
        st.write(f"Stage: {info.get('current_stage', 'Production')}")
        st.write(f"Source: {info.get('model_source', 'local')}")
        st.write(f"Total Versi: {info.get('total_versions', 1)}")
        if info.get('versions'):
            with st.expander("📋 Detail Versi"):
                for v in info['versions']:
                    st.caption(f"v{v['version']}: {v['stage']} ({v['run_id'][:8]}...)")
    else:
        st.error("❌ Tidak terhubung")
    
    st.divider()
    
    st.header("📖 Kategori AQI")
    st.markdown('''
    | AQI | Kategori | Emoji |
    |-----|----------|-------|
    | 0-50 | Good | 🟢 |
    | 51-100 | Moderate | 🟡 |
    | 101-150 | Sensitive | 🟠 |
    | 151-200 | Unhealthy | 🔴 |
    | 201-300 | Very Unhealthy | 🟣 |
    | 300+ | Hazardous | ⚫ |
    ''')
    
    st.divider()
    
    st.header("🔄 Update Data")
    st.code("python src/ingest_data.py", language="bash")
    
    st.divider()
    
    st.header("🔗 Akses Cepat")
    st.markdown('''
    - [API Docs](http://localhost:8000/docs)
    - [MLflow UI](http://localhost:5000)
    - [GitHub](https://github.com/Izz5125/MLOps-Air_Quality_Prediction)
    ''')

st.divider()
st.caption(f"🤖 MLOps Air Quality Prediction - Malang | © 2026 | Page loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")