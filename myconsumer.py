import os
import json
import requests
import numpy as np
import pandas as pd
from joblib import load
from kafka import KafkaConsumer
from cassandra.cluster import Cluster

# ===========================
# CONFIG
# ===========================
TOPIC = "weather_realtime"
BOOTSTRAP_SERVERS = ["localhost:9092"]

# Real-time model (Option C): per-message, TANPA window 48 jam
MODEL_BUNDLE_PATH = "temp_rf_realtime.pkl"  # output dari train_model_realtime.py

# Optional Telegram (kalau tidak diset, hanya print)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# Anti-spam telegram
COOLDOWN_SECONDS = 60  # 1 menit
last_sent_ts = 0.0

# ===========================
# LOAD MODEL BUNDLE
# ===========================
bundle = load(MODEL_BUNDLE_PATH)
model = bundle["model"]
feature_cols = bundle["feature_cols"]

print("=" * 60)
print("REAL-TIME WEATHER CONSUMER (Option C: per-message model)")
print("=" * 60)
print(f"Topic: {TOPIC}")
print(f"Model: {MODEL_BUNDLE_PATH}")
print(f"Features: {feature_cols}")
print(f"Telegram enabled: {TELEGRAM_ENABLED}\n")

# ===========================
# CASSANDRA
# ===========================
cluster = Cluster(["127.0.0.1"])
session = cluster.connect("weather")

insert_stmt = session.prepare(
    "INSERT INTO weather_data (city, datetime, temp, rain, humidity, wind_speed) VALUES (?, ?, ?, ?, ?, ?)"
)

def send_telegram(msg: str):
    global last_sent_ts
    if not TELEGRAM_ENABLED:
        return
    now = pd.Timestamp.now().timestamp()
    if now - last_sent_ts < COOLDOWN_SECONDS:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, json=payload, timeout=10)
        last_sent_ts = now
    except Exception:
        pass

# ===========================
# KAFKA CONSUMER
# ===========================
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="weather_realtime_consumer_modelC",
)

for message in consumer:
    payload = message.value

    # Parse time: pakai waktu dari API (payload["datetime"])
    # Jika mau event time yang lebih detail, bisa pakai fetched_at.
    dt = pd.to_datetime(payload.get("fetched_at") or payload.get("datetime"), errors="coerce")
    if pd.isna(dt):
        dt = pd.Timestamp.now()
    dt_py = dt.to_pydatetime()

    # Save raw observation
    try:
        session.execute(
            insert_stmt,
            (
                payload.get("city", "Unknown"),
                dt_py,
                float(payload["temp"]),
                float(payload["rain"]),
                float(payload["humidity"]),
                float(payload["wind_speed"]),
            ),
        )
    except Exception as e:
        print(f" Cassandra insert error: {e}")

    # Build features (no lags, no window)
    feat = {
        "temp": float(payload["temp"]),
        "rain": float(payload["rain"]),
        "humidity": float(payload["humidity"]),
        "wind_speed": float(payload["wind_speed"]),
        "hour": int(dt.hour),
        "dayofweek": int(dt.dayofweek),
    }

    X = np.array([[feat[c] for c in feature_cols]], dtype=float)
    pred_next = float(model.predict(X)[0])

    now_str = pd.Timestamp.now().strftime("%H:%M:%S")
    print(
        f" [{now_str}] {payload.get('city','')} "
        f"temp_now={feat['temp']:.2f}°C → pred_next={pred_next:.2f}°C "
        f"(dt={dt.strftime('%Y-%m-%d %H:%M:%S')})"
    )

    # Optional alert example: kalau pred naik/turun ≥ 1°C
    delta = pred_next - feat["temp"]
    if abs(delta) >= 1.0:
        direction = "naik" if delta > 0 else "turun"
        msg = (
            f" Weather Alert ({payload.get('city','')})\n"
            f"Suhu sekarang: {feat['temp']:.2f}°C\n"
            f"Prediksi next tick: {pred_next:.2f}°C ({direction} {abs(delta):.2f}°C)\n"
            f"Hujan: {feat['rain']}mm | RH: {feat['humidity']}% | Wind: {feat['wind_speed']}km/h"
        )
        send_telegram(msg)

