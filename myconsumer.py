from kafka import KafkaConsumer
import json
import numpy as np
import requests
from joblib import load

# ================= CONFIG =================
TOPIC_NAME = "weather_realtime"
KAFKA_BOOTSTRAP = ["localhost:9092"]

BOT_TOKEN = "8374103851:AAE_0rJqVKpCsuIDLtJu4KeHFyWnmdR_AGw"
CHAT_ID = 5149097504

model = load("temp_rf_multifeature.pkl")
scaler = load("scaler.pkl")

# ================= TELEGRAM =================
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": msg})

# ================= KAFKA CONSUMER =================
consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=KAFKA_BOOTSTRAP,
    auto_offset_reset="latest",
    group_id="weather_realtime_group",
    value_deserializer=lambda v: json.loads(v.decode("utf-8"))
)

print("🌦️ Weather REALTIME Consumer started...\n")

# ================= MAIN LOOP =================
for message in consumer:
    data = message.value

    if not isinstance(data, dict):
        continue

    city = data["city"]
    time_now = data["datetime"]

    temp_now = float(data["temp"])
    rain_now = float(data["rain"])
    hum_now = float(data["humidity"])
    wind_now = float(data["wind_speed"])

    # ================= STATUS =================
    if rain_now > 0.1:
        status = "HUJAN"
        icon = "🌧️"
    elif temp_now > 35:
        status = "PANAS EKSTRIM"
        icon = "🔥"
    else:
        status = "NORMAL"
        icon = "🌤️"

    msg = (
        f"{icon} CUACA REALTIME\n\n"
        f"Kota: {city}\n"
        f"Waktu: {time_now}\n\n"
        f"Suhu: {temp_now:.2f} °C\n"
        f"Hujan: {rain_now:.2f} mm\n"
        f"Kelembapan: {hum_now:.0f} %\n"
        f"Angin: {wind_now:.2f} km/h\n"
        f"Status: {status}"
    )

    send_telegram(msg)
    print(f"[{city}] Telegram sent | {status}")

