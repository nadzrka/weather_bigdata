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

# Realtime interval (seconds)
FETCH_INTERVAL = 5  # 5 detik

# ===========================
# KAFKA PRODUCER
# ===========================
producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    acks="all",
    retries=3,
)

print("=" * 60)
print("🌤️ REAL-TIME WEATHER PRODUCER (5s)")
print("=" * 60)
print(f"📍 City: {CITY} ({LAT}, {LON})")
print(f"⏱️ Interval: {FETCH_INTERVAL}s")
print(f"📡 Topic: {TOPIC}\n")

MAX_RETRIES = 3

# ===========================
# MAIN LOOP
# ===========================
while True:
    loop_start = time.time()

    for attempt in range(MAX_RETRIES):
        try:
            url = (
                "https://api.open-meteo.com/v1/forecast?"
                f"latitude={LAT}&longitude={LON}"
                "&current=temperature_2m,rain,relative_humidity_2m,windspeed_10m"
                "&timezone=Asia/Jakarta"
            )

            res = requests.get(url, timeout=15)
            res.raise_for_status()
            data = res.json()

            cur = data["current"]

            payload = {
                "city": CITY,
                # Gunakan waktu dari API (bisa tidak berubah kalau API belum update)
                "datetime": cur["time"],
                "temp": float(cur["temperature_2m"]),
                "rain": float(cur["rain"]),
                "humidity": float(cur["relative_humidity_2m"]),
                "wind_speed": float(cur["windspeed_10m"]),
                # Tambahan untuk memastikan event unik pada stream (opsional)
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
            }

            future = producer.send(TOPIC, value=payload)
            producer.flush()
            record_metadata = future.get(timeout=10)

            print(
                f"✅ [{datetime.now().strftime('%H:%M:%S')}] "
                f"{payload['temp']}°C | {payload['rain']}mm | "
                f"Partition: {record_metadata.partition}"
            )
            break

        except requests.exceptions.RequestException as e:
            print(f"❌ API Error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES - 1:
                print("⚠️ Max retries reached")
            else:
                time.sleep(2)

        except Exception as e:
            print(f"❌ Error: {e}")
            break

    print(f"⏳ Next fetch in {FETCH_INTERVAL}s...\n")
    elapsed = time.time() - loop_start
    time.sleep(max(0, FETCH_INTERVAL - elapsed))

