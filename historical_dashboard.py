"""
Weather Forecasting Dashboard
Streamlit app untuk analisis historical data dan ML prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from cassandra.cluster import Cluster
from datetime import datetime, timedelta
import joblib

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Weather Forecasting",
    page_icon="🌦️",
    layout="wide"
)

# =============================
# CONFIG
# =============================
LOOK_BACK = 48

FEATURE_COLS = [
    "temp", "rain", "humidity", "wind_speed",
    "hour", "dayofweek",
    "temp_lag1", "temp_lag3", "temp_lag6"
]

# =============================
# LOAD MODEL & SCALER
# =============================
@st.cache_resource
def load_models():
    try:
        model = joblib.load("temp_rf_multifeature.pkl")
        scaler = joblib.load("scaler.pkl")
        
        metadata = {}
        try:
            import json
            with open("model_metadata.json", "r") as f:
                metadata = json.load(f)
        except:
            pass
        
        return model, scaler, metadata
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("💡 Run: python3 train_model_dashboard.py")
        st.stop()

model, scaler, metadata = load_models()

# =============================
# CONNECT TO CASSANDRA
# =============================
@st.cache_resource
def get_cassandra_session():
    try:
        cluster = Cluster(["127.0.0.1"])
        session = cluster.connect("weather")
        return session
    except Exception as e:
        st.error(f"❌ Cannot connect to Cassandra: {e}")
        st.info("💡 Make sure Cassandra is running")
        st.stop()

session = get_cassandra_session()

# =============================
# LOAD DATA
# =============================
@st.cache_data(ttl=300)
def load_data(city, start_date, end_date):
    # FIX: Ganti historical_weather jadi weather_data
    query = """
        SELECT city, datetime, temp, rain, humidity, wind_speed
        FROM weather_data
        WHERE city = %s
          AND datetime >= %s
          AND datetime <= %s
    """
    
    rows = session.execute(query, (
        city,
        datetime.combine(start_date, datetime.min.time()),
        datetime.combine(end_date, datetime.max.time())
    ))
    
    return pd.DataFrame(rows)

# =============================
# SIDEBAR
# =============================
st.sidebar.title("🌦️ Weather Forecasting")
st.sidebar.markdown("### Configuration")

city = st.sidebar.text_input("City:", "Surakarta")
start_date = st.sidebar.date_input(
    "Start Date:", 
    datetime.now() - timedelta(days=30)
)
end_date = st.sidebar.date_input(
    "End Date:", 
    datetime.now()
)

if metadata:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Info")
    st.sidebar.metric("MAE", f"{metadata.get('mae', 0):.2f}°C")
    st.sidebar.metric("RMSE", f"{metadata.get('rmse', 0):.2f}°C")
    st.sidebar.metric("R²", f"{metadata.get('r2', 0):.3f}")

# =============================
# MAIN PAGE
# =============================
st.title("🌦️ Weather Forecasting Dashboard")
st.markdown(f"**City:** {city} | **Period:** {start_date} to {end_date}")

# =============================
# LOAD DATA
# =============================
with st.spinner("Loading data..."):
    df_raw = load_data(city, start_date, end_date)

if df_raw.empty:
    st.warning("⚠️ No data found")
    st.info("💡 Run: python3 weather_producer.py && python3 weather_consumer.py")
    st.stop()

# =============================
# PREPROCESSING
# =============================
df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
df_raw = df_raw.sort_values("datetime")

# Resample hourly
# Resample hourly (rain dijumlahkan)
df_hour = (
    df_raw.set_index("datetime")
          .resample("1H")
          .agg({
              "temp": "mean",
              "humidity": "mean",
              "wind_speed": "mean",
              "rain": "sum",
          })
          .reset_index()
)
df = df_hour
df["city"] = city


# Feature engineering
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["temp_lag1"] = df["temp"].shift(1)
df["temp_lag3"] = df["temp"].shift(3)
df["temp_lag6"] = df["temp"].shift(6)

df = df.dropna().copy()

if len(df) < LOOK_BACK:
    st.warning(f"⚠️ Need at least {LOOK_BACK} rows")
    st.stop()

# =============================
# METRICS
# =============================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Records", len(df))

with col2:
    st.metric("🌡️ Avg Temp", f"{df['temp'].mean():.1f}°C")

with col3:
    st.metric("🔥 Max Temp", f"{df['temp'].max():.1f}°C")

with col4:
    st.metric("☔ Total Rain", f"{df['rain'].sum():.1f}mm")

st.markdown("---")

# =============================
# TEMPERATURE CHART
# =============================
st.subheader("🌡️ Temperature Over Time")

temp_chart = alt.Chart(df).mark_line(color="#FF4B4B", strokeWidth=2).encode(
    x=alt.X("datetime:T", title="Date/Time"),
    y=alt.Y("temp:Q", title="Temperature (°C)", scale=alt.Scale(zero=False)),
    tooltip=[
        alt.Tooltip("datetime:T", title="Time"),
        alt.Tooltip("temp:Q", title="Temp", format=".1f")
    ]
).properties(height=400)

st.altair_chart(temp_chart, use_container_width=True)

# =============================
# ENVIRONMENTAL FACTORS
# =============================
st.subheader("🌧️ Environmental Factors")

env_df = df.melt(
    id_vars="datetime",
    value_vars=["rain", "humidity", "wind_speed"],
    var_name="Variable",
    value_name="Value"
)

env_chart = alt.Chart(env_df).mark_line().encode(
    x=alt.X("datetime:T", title="Date/Time"),
    y=alt.Y("Value:Q", title="Value"),
    color=alt.Color("Variable:N", legend=alt.Legend(title="Metric")),
    tooltip=["datetime:T", "Variable:N", "Value:Q"]
).properties(height=400)

st.altair_chart(env_chart, use_container_width=True)

st.markdown("---")

# =============================
# ML PREDICTION
# =============================
st.subheader("🤖 Next Hour Prediction")

try:
    features = df[FEATURE_COLS].astype(float).values
    
    last_window = features[-LOOK_BACK:]
    scaled = scaler.transform(last_window)
    X_input = scaled.flatten().reshape(1, -1)
    
    predicted_temp = float(model.predict(X_input)[0])
    current_temp = df["temp"].iloc[-1]
    diff = predicted_temp - current_temp
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌡️ Current", f"{current_temp:.1f}°C")
    
    with col2:
        st.metric("🔮 Predicted", f"{predicted_temp:.1f}°C", delta=f"{diff:+.1f}°C")
    
    with col3:
        if predicted_temp > 35:
            st.error("🔥 High Temperature!")
        elif predicted_temp < 20:
            st.info("❄️ Cool")
        else:
            st.success("✅ Normal")

except Exception as e:
    st.error(f"❌ Prediction error: {e}")

st.markdown("---")

# =============================
# ACTUAL vs PREDICTED (24H)
# =============================
st.subheader("📊 Actual vs Predicted (Last 24 Hours)")

try:
    end_dt = df["datetime"].max()
    start_24h = end_dt - timedelta(hours=24)
    
    df = df.reset_index(drop=True)
    start_idx = df[df["datetime"] >= start_24h].index[0]
    
    predicted_list = []
    actual_list = []
    datetime_list = []
    
    for i in range(start_idx, len(df)):
        if i - LOOK_BACK < 0:
            continue
        
        window = df.loc[i-LOOK_BACK:i-1, FEATURE_COLS].astype(float).values
        window_scaled = scaler.transform(window).flatten().reshape(1, -1)
        
        p_temp = float(model.predict(window_scaled)[0])
        predicted_list.append(float(p_temp))
        actual_list.append(float(df.loc[i, "temp"]))
        datetime_list.append(df.loc[i, "datetime"])
    
    if len(predicted_list) > 0:
        df_compare = pd.DataFrame({
            "datetime": datetime_list,
            "Actual": actual_list,
            "Predicted": predicted_list
        })
        
        compare_chart = alt.Chart(df_compare).transform_fold(
            ["Actual", "Predicted"],
            as_=["Type", "Temperature"]
        ).mark_line(strokeWidth=2).encode(
            x=alt.X("datetime:T", title="Time"),
            y=alt.Y("Temperature:Q", title="Temp (°C)"),
            color=alt.Color("Type:N", 
                          scale=alt.Scale(domain=["Actual", "Predicted"],
                                        range=["#FF4B4B", "#0068C9"])),
            tooltip=["datetime:T", "Type:N", "Temperature:Q"]
        ).properties(height=400)
        
        st.altair_chart(compare_chart, use_container_width=True)
        
        mae_24h = np.mean(np.abs(np.array(actual_list) - np.array(predicted_list)))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 24h MAE", f"{mae_24h:.2f}°C")
        with col2:
            accuracy = max(0, 100 - (mae_24h / np.mean(actual_list) * 100))
            st.metric("✅ Accuracy", f"{accuracy:.1f}%")
    
except Exception as e:
    st.error(f"❌ Comparison error: {e}")

# =============================
# RAW DATA
# =============================
with st.expander("📄 Raw Data"):
    st.dataframe(
        df[["datetime", "temp", "rain", "humidity", "wind_speed"]].tail(100),
        use_container_width=True
    )

st.markdown("---")
st.markdown("<div style='text-align:center'>🌦️ Weather Forecasting System</div>", 
            unsafe_allow_html=True)

