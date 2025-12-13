import requests
import json
from kafka import KafkaProducer
import time
from datetime import datetime

CITY = "Surakarta"
LAT = -7.56
LON = 110.82

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("Starting REAL-TIME Open-Meteo weather producer...\n")

while True:
    try:
        # Real-time & near real-time weather data
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={LAT}&longitude={LON}"
            f"&current=temperature_2m,rain,relative_humidity_2m,windspeed_10m"
            f"&timezone=Asia/Jakarta"
        )

        res = requests.get(url).json()
        current = res["current"]

        msg = {
            "city": CITY,
            "datetime": datetime.now().isoformat(),
            "temp": current["temperature_2m"],
            "rain": current["rain"],
            "humidity": current["relative_humidity_2m"],
            "wind_speed": current["windspeed_10m"]
        }

        producer.send("weather_hourly", value=msg)
        producer.flush()

        print(f"[{datetime.now()}] Sent REAL-TIME weather:", msg)

    except Exception as e:
        print("Error:", e)

    # Kirim tiap 60 detik
    time.sleep(1)

