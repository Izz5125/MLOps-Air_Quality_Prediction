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
# HEADER
# ============================================
st.title("🌫️ Air Quality Index Predictor")
st.markdown("Prediksi kualitas udara Kota Malang menggunakan Machine Learning")
st.divider()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("📊 Model Information")
    try:
        response = requests.get("http://localhost:8000/model/info")
        if response.status_code == 200:
            info = response.json()
            st.success("✅ Model Connected")
            st.write(f"**Model:** {info.get('model_name', 'AQI_Predictor')}")
            st.write(f"**Stage:** {info.get('current_stage', 'Production')}")
            st.write(f"**Source:** {info.get('model_source', 'local')}")
            st.write(f"**Versions:** {info.get('total_versions', 1)}")
    except:
        st.error("❌ Cannot connect to API")
        st.info("Make sure Docker is running: `docker compose up -d`")
    
    st.divider()
    
    st.header("📖 AQI Categories")
    st.markdown("""
    | Range | Category | Color |
    |-------|----------|-------|
    | 0-50 | Good | 🟢 |
    | 51-100 | Moderate | 🟡 |
    | 101-150 | Unhealthy (Sensitive) | 🟠 |
    | 151-200 | Unhealthy | 🔴 |
    | 201-300 | Very Unhealthy | 🟣 |
    | 300+ | Hazardous | ⚫ |
    """)
    
    st.divider()
    
    st.header("🕐 Last Predictions")
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if st.session_state.history:
        for pred in st.session_state.history[-5:]:
            st.caption(f"{pred['time']}: AQI {pred['aqi']} - {pred['category']}")
    else:
        st.caption("No predictions yet")

# ============================================
# MAIN CONTENT
# ============================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Sensor Data")
    
    with st.form("prediction_form"):
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            pm1 = st.number_input("PM1.0 (µg/m³)", 0.0, 500.0, 15.0, 0.1, help="Particulate Matter 1.0")
            pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 500.0, 25.0, 0.1, help="Particulate Matter 2.5")
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 65.0, 0.1, help="Relative Humidity")
        
        with subcol2:
            temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 28.0, 0.1, help="Temperature in Celsius")
            um003 = st.number_input("Ultrafine Particles", 0.0, 5000.0, 500.0, 0.1, help="Ultrafine particle count")
            hour = st.number_input("Hour (optional)", 0, 23, datetime.now().hour, help="Current hour, auto-filled")
        
        submit = st.form_submit_button("🔮 Predict AQI", use_container_width=True)

with col2:
    st.subheader("📊 Prediction Results")
    
    if submit:
        with st.spinner("🔄 Predicting..."):
            try:
                data = {
                    "pm1": pm1,
                    "pm25": pm25,
                    "relativehumidity": humidity,
                    "temperature": temperature,
                    "um003": um003,
                    "hour": hour
                }
                
                response = requests.post(
                    "http://localhost:8000/predict",
                    json=data,
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Simpan history
                    st.session_state.history.append({
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'aqi': result['predicted_aqi'],
                        'category': result['aqi_category']
                    })
                    
                    # Tampilkan hasil
                    st.success("✅ Prediction Complete!")
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(
                            "PM2.5 Prediction",
                            f"{result['predicted_pm25']:.1f} µg/m³",
                            delta=None
                        )
                    
                    with metric_col2:
                        st.metric(
                            "AQI Index",
                            f"{result['predicted_aqi']:.0f}",
                            delta=None
                        )
                    
                    with metric_col3:
                        aqi_cat = result['aqi_category']
                        emoji = "🟢" if "Good" in aqi_cat else "🟡" if "Moderate" in aqi_cat else "🟠" if "Sensitive" in aqi_cat else "🔴" if "Unhealthy" in aqi_cat else "🟣"
                        st.metric(
                            "Category",
                            f"{emoji} {aqi_cat}"
                        )
                    
                    # Progress bar AQI
                    st.markdown("**AQI Level:**")
                    aqi_val = result['predicted_aqi']
                    st.progress(min(aqi_val / 300, 1.0))
                    
                    # Rekomendasi
                    st.info(f"💡 **Rekomendasi:** {result['recommendation']}")
                    
                    # Detail
                    with st.expander("📋 Detail Response"):
                        st.json(result)
                    
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.code(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API server")
                st.info("💡 Run: `docker compose up -d`")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.info("👈 Masukkan data sensor dan klik **Predict AQI** untuk melihat hasil")

# ============================================
# BATCH PREDICTION
# ============================================
st.divider()
st.subheader("📦 Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview data:")
    st.dataframe(df.head())
    
    if st.button("🚀 Predict All"):
        with st.spinner("Processing..."):
            try:
                response = requests.post(
                    "http://localhost:8000/predict/batch",
                    json={"data": df.to_dict('records')},
                    timeout=30
                )
                if response.status_code == 200:
                    results = response.json()
                    st.success(f"✅ {len(results)} predictions completed!")
                    
                    # Tampilkan hasil
                    result_df = pd.DataFrame(results)
                    st.dataframe(result_df)
                    
                    # Download
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "predictions.csv",
                        "text/csv"
                    )
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption(f"🤖 MLOps Air Quality Prediction - Malang | © 2026 | Last update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
