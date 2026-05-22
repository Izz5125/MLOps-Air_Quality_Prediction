import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="AQI Predictor - Malang",
    page_icon="🌫️",
    layout="wide"
)

# ============================================
# LOAD LATEST DATA
# ============================================
@st.cache_data(ttl=300)  # Cache 5 menit
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

# ============================================
# HEADER
# ============================================
st.title("🌫️ Air Quality Index Predictor - Malang")
st.markdown("Prediksi kualitas udara real-time menggunakan Machine Learning")

# Status bar
health = get_health()
if health and health.get("model_loaded"):
    st.success(f"🟢 Sistem Aktif | Model: {health.get('model_source', 'loaded')} | MLflow: {'Terhubung' if health.get('mlflow_connected') else 'Tidak'}")
else:
    st.error("🔴 Sistem Offline - Jalankan `docker compose up -d`")

st.divider()

# ============================================
# MENU UTAMA: PREDIKSI SAAT INI
# ============================================
st.header("📡 Prediksi Kualitas Udara Saat Ini")

latest = get_latest_data()

if latest and "latest_data" in latest and latest["latest_data"]:
    data = latest["latest_data"]
    pred = latest.get("prediction", {})
    
    # Tampilkan data sensor terkini
    st.subheader("📊 Data Sensor Terkini")
    
    cols = st.columns(5)
    metrics_map = {
        "PM1.0": ("pm1", "µg/m³"),
        "PM2.5": ("pm25", "µg/m³"),
        "Humidity": ("relativehumidity", "%"),
        "Temperature": ("temperature", "°C"),
        "Ultrafine": ("um003", "particles/cm³"),
    }
    
    values = {}
    for label, (key, unit) in metrics_map.items():
        val = data.get(key, "N/A")
        values[key] = val
        with cols[list(metrics_map.keys()).index(label)]:
            st.metric(f"{label}", f"{val:.1f}" if isinstance(val, (int, float)) else val, delta=None)
    
    st.divider()
    
    # Tampilkan prediksi
    if pred and "error" not in pred:
        st.subheader("🔮 Hasil Prediksi AQI")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("PM2.5 Diprediksi", f"{pred['predicted_pm25']:.1f} µg/m³")
        with col2:
            aqi = pred['predicted_aqi']
            st.metric("Indeks AQI", f"{aqi:.0f}")
        with col3:
            cat = pred['aqi_category']
            emoji = {"Good": "🟢", "Moderate": "🟡", "Unhealthy for Sensitive Groups": "🟠", "Unhealthy": "🔴", "Very Unhealthy": "🟣", "Hazardous": "⚫"}
            st.metric("Kategori", f"{emoji.get(cat, '❓')} {cat}")
        
        # Progress bar
        st.markdown("**Level AQI:**")
        st.progress(min(aqi / 300, 1.0))
        
        # Rekomendasi
        st.info(f"💡 **Rekomendasi:** {pred['recommendation']}")
        
        # Grafik perbandingan
        st.subheader("📈 Perbandingan PM2.5")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.metric("PM2.5 Saat Ini", f"{values.get('pm25', 0):.1f} µg/m³")
        with chart_col2:
            delta = pred['predicted_pm25'] - values.get('pm25', 0)
            st.metric("PM2.5 Prediksi (1 jam ke depan)", f"{pred['predicted_pm25']:.1f} µg/m³", 
                     delta=f"{delta:+.1f} µg/m³", delta_color="inverse")
    
    elif pred and "error" in pred:
        st.warning(f"⚠️ Prediksi gagal: {pred['error']}")
    else:
        st.warning("⚠️ Belum ada data prediksi")
    
    # Timestamp
    st.caption(f"🕐 Data diperbarui: {datetime.now().strftime('%Y-%m-%d %H:%M:%S WIB')}")

else:
    st.warning("⚠️ Tidak dapat mengambil data terbaru. Pastikan API berjalan.")
    st.info("💡 Jalankan: `docker compose up -d`")

st.divider()

# ============================================
# MENU TAMBAHAN: INPUT MANUAL
# ============================================
with st.expander("🔧 Prediksi Manual (Input Sendiri)", expanded=False):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Data Sensor")
        
        with st.form("manual_form"):
            sub1, sub2 = st.columns(2)
            with sub1:
                pm1 = st.number_input("PM1.0 (µg/m³)", 0.0, 500.0, 15.0, 0.1)
                pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 500.0, 25.0, 0.1)
                humidity = st.number_input("Humidity (%)", 0.0, 100.0, 65.0, 0.1)
            with sub2:
                temp = st.number_input("Temperature (°C)", 0.0, 50.0, 28.0, 0.1)
                um003 = st.number_input("Ultrafine Particles", 0.0, 5000.0, 500.0, 0.1)
                hour = st.number_input("Hour (0-23)", 0, 23, datetime.now().hour)
            
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
                st.metric("PM2.5", f"{result['predicted_pm25']:.1f} µg/m³")
                st.metric("AQI", f"{result['predicted_aqi']:.0f}")
                st.metric("Kategori", result['aqi_category'])
                st.info(f"💡 {result['recommendation']}")
            else:
                st.error("❌ Gagal memprediksi")
        else:
            st.info("👈 Isi data dan klik Predict")

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("📊 Informasi Model")
    info = get_model_info()
    if info:
        st.success("✅ Model Terhubung")
        st.write(f"**Nama:** {info.get('model_name', 'AQI_Predictor')}")
        st.write(f"**Stage:** {info.get('current_stage', 'Production')}")
        st.write(f"**Source:** {info.get('model_source', 'local')}")
        st.write(f"**Total Versi:** {info.get('total_versions', 1)}")
        if info.get('versions'):
            with st.expander("📋 Detail Versi"):
                for v in info['versions']:
                    st.caption(f"v{v['version']}: {v['stage']} ({v['run_id'][:8]}...)")
    else:
        st.error("❌ Tidak terhubung")
    
    st.divider()
    
    st.header("📖 Kategori AQI")
    st.markdown("""
    | AQI | Kategori | Emoji |
    |-----|----------|-------|
    | 0-50 | Good | 🟢 |
    | 51-100 | Moderate | 🟡 |
    | 101-150 | Unhealthy (Sensitive) | 🟠 |
    | 151-200 | Unhealthy | 🔴 |
    | 201-300 | Very Unhealthy | 🟣 |
    | 300+ | Hazardous | ⚫ |
    """)
    
    st.divider()
    
    st.header("🔗 Akses Cepat")
    st.markdown(f"""
    - [API Docs](http://localhost:8000/docs)
    - [MLflow UI](http://localhost:5000)
    - [GitHub Repo](https://github.com/Izz5125/MLOps-Air_Quality_Prediction)
    """)

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption(f"🤖 MLOps Air Quality Prediction - Malang | © 2026 | Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
