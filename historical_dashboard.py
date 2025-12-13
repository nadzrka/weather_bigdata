import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from cassandra.cluster import Cluster
from datetime import datetime, timedelta
import joblib

# =============================
# CONFIG (HARUS SAMA DENGAN TRAINING)
# =============================
LOOK_BACK = 48   # samakan dengan training (24 / 48)

FEATURE_COLS = [
    "temp", "rain", "humidity", "wind_speed",
    "hour", "dayofweek",
    "temp_lag1", "temp_lag3", "temp_lag6"
]

# =============================
# LOAD MODEL & SCALER
# =============================
model = joblib.load("temp_rf_multifeature.pkl")
scaler = joblib.load("scaler.pkl")

# =============================
# CONNECT TO CASSANDRA
# =============================
cluster = Cluster(["127.0.0.1"])
session = cluster.connect("weather")

def load_data(city, start_date, end_date):
    query = """
        SELECT city, datetime, temp, rain, humidity, wind_speed
        FROM historical_weather
        WHERE city = %s
          AND datetime >= %s
          AND datetime <= %s
        ALLOW FILTERING;
    """
    rows = session.execute(query, (
        city,
        datetime.combine(start_date, datetime.min.time()),
        datetime.combine(end_date, datetime.max.time())
    ))

    return pd.DataFrame(rows)

# =============================
# STREAMLIT UI
# =============================
st.sidebar.title("🌦️ Weather Forecasting App")
st.sidebar.markdown("Cassandra + Random Forest")

city = st.sidebar.text_input("City:", "Surakarta")
start_date = st.sidebar.date_input(
    "Start Date:", datetime.now() - timedelta(days=30)
)
end_date = st.sidebar.date_input(
    "End Date:", datetime.now()
)

# =============================
# LOAD DATA
# =============================
df = load_data(city, start_date, end_date)

if df.empty:
    st.warning("⚠️ No data found for this city.")
    st.stop()

df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

# =============================
# FEATURE ENGINEERING (SAMA DENGAN TRAINING)
# =============================
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["temp_lag1"] = df["temp"].shift(1)
df["temp_lag3"] = df["temp"].shift(3)
df["temp_lag6"] = df["temp"].shift(6)

df = df.dropna()

# =============================
# RAW DATA
# =============================
st.subheader("📄 Weather Data (from Cassandra)")
st.dataframe(df.tail(20))

# =============================
# TEMPERATURE OVER TIME
# =============================
st.subheader("🌡️ Temperature Over Time")

st.altair_chart(
    alt.Chart(df).mark_line(color="red").encode(
        x="datetime:T",
        y="temp:Q",
        tooltip=["datetime:T", "temp:Q"]
    ),
    use_container_width=True
)

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

st.altair_chart(
    alt.Chart(env_df).mark_line().encode(
        x="datetime:T",
        y="Value:Q",
        color="Variable:N",
        tooltip=["datetime:T", "Variable:N", "Value:Q"]
    ),
    use_container_width=True
)

# =============================
# ML PREDICTION (NEXT HOUR)
# =============================
st.subheader("🤖 Random Forest – Next Hour Temperature Prediction")

features = df[FEATURE_COLS].values

if len(features) < LOOK_BACK:
    st.warning(f"Need at least {LOOK_BACK} data points for prediction.")
    st.stop()

last_window = features[-LOOK_BACK:]
scaled = scaler.transform(last_window)
X_input = scaled.flatten().reshape(1, -1)

pred_scaled = model.predict(X_input)[0]

n_features = scaler.n_features_in_
dummy = np.zeros((1, n_features))
dummy[0, 0] = pred_scaled

predicted_temp = scaler.inverse_transform(dummy)[0, 0]

st.success(f"🌡️ Predicted Next-Hour Temperature: **{predicted_temp:.2f} °C**")

# =============================
# ACTUAL vs PREDICTED
# =============================
st.subheader("📊 Actual vs Predicted Temperature (Last 24 Hours)")

actual = df["temp"].tail(24).values
predicted = []

for i in range(len(features) - 24, len(features)):
    window = features[i-LOOK_BACK:i]
    window_scaled = scaler.transform(window).flatten().reshape(1, -1)
    p_scaled = model.predict(window_scaled)[0]

    dummy = np.zeros((1, n_features))
    dummy[0, 0] = p_scaled
    predicted.append(scaler.inverse_transform(dummy)[0, 0])

df_compare = pd.DataFrame({
    "datetime": df["datetime"].tail(24).values,
    "Actual": actual,
    "Predicted": predicted
})

st.altair_chart(
    alt.Chart(df_compare).transform_fold(
        ["Actual", "Predicted"],
        as_=["Type", "Temperature"]
    ).mark_line().encode(
        x="datetime:T",
        y="Temperature:Q",
        color="Type:N",
        tooltip=["datetime:T", "Type:N", "Temperature:Q"]
    ),
    use_container_width=True
)

