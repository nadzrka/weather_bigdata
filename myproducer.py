import requests
import json
import time
from kafka import KafkaProducer
from datetime import datetime

# ===========================
# CONFIG
# ===========================
CITY = "Surakarta"
LAT = -7.56
LON = 110.82

TOPIC = "weather_realtime"
BOOTSTRAP_SERVERS = ["localhost:9092"]

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

print("🌤️ Weather REALTIME Producer started...\n")

while True:
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={LAT}&longitude={LON}"
            "&current=temperature_2m,rain,relative_humidity_2m,windspeed_10m"
            "&timezone=Asia/Jakarta"
        )

        res = requests.get(url, timeout=10).json()
        cur = res["current"]

        payload = {
            "city": CITY,
            "datetime": cur["time"],        # JAM SEKARANG
            "temp": cur["temperature_2m"],
            "rain": cur["rain"],
            "humidity": cur["relative_humidity_2m"],
            "wind_speed": cur["windspeed_10m"]
        }

        producer.send(TOPIC, value=payload)
        producer.flush()

        print("Sent realtime weather:", payload)

    except Exception as e:
        print("Error:", e)

    time.sleep(1)   # kirim tiap 1 menit

