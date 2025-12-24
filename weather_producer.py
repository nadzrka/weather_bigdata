"""
Batch Producer - Load Historical Weather Data
Mengambil data 30 hari terakhir dari Open-Meteo Archive API
"""

import requests
import json
import time
from kafka import KafkaProducer
from datetime import datetime, timedelta

# ===========================
# CONFIG
# ===========================
CITY = "Surakarta"
LAT = -7.56
LON = 110.82

TOPIC = "weather_hourly"
BOOTSTRAP_SERVERS = ["localhost:9092"]

DAYS_BACK = 30
ARCHIVE_DELAY_DAYS = 5
SLEEP_BETWEEN_SEND = 0.02

# ===========================
# FUNCTIONS
# ===========================
def fetch_hourly_history(lat, lon, start_date, end_date):
    """Fetch historical hourly data from Open-Meteo Archive API"""
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,rain,relative_humidity_2m,windspeed_10m"
        "&timezone=Asia/Jakarta"
    )
    
    print(f"🌐 Fetching: {url}")
    r = requests.get(url, timeout=30)
    
    if r.status_code != 200:
        print(f"❌ HTTP {r.status_code}")
        print(f"Response: {r.text[:500]}")
        r.raise_for_status()
    
    return r.json()

# ===========================
# MAIN
# ===========================
def main():
    print("=" * 60)
    print("📦 BATCH WEATHER PRODUCER")
    print("=" * 60)
    print(f"📍 City: {CITY} ({LAT}, {LON})")
    print(f"📅 Loading {DAYS_BACK} days of historical data")
    print(f"📡 Kafka Topic: {TOPIC}\n")

    # Calculate date range
    today = datetime.now().date()
    end = today - timedelta(days=ARCHIVE_DELAY_DAYS)
    start = end - timedelta(days=DAYS_BACK)
    
    print(f"📆 Date Range: {start} to {end}\n")

    # Initialize Kafka Producer
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks='all',
        retries=3
    )

    try:
        # Fetch data from API
        data = fetch_hourly_history(LAT, LON, start.isoformat(), end.isoformat())
        hourly = data.get("hourly", {})

        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        rains = hourly.get("rain", [])
        hums = hourly.get("relative_humidity_2m", [])
        winds = hourly.get("windspeed_10m", [])

        n = min(len(times), len(temps), len(rains), len(hums), len(winds))
        
        if n == 0:
            print("❌ No data received from API!")
            return

        print(f"✅ Fetched {n} hourly records")
        print(f"🚀 Sending to Kafka...\n")

        success_count = 0

        for i in range(n):
            msg = {
                "city": CITY,
                "datetime": times[i],
                "temp": float(temps[i]) if temps[i] is not None else 0.0,
                "rain": float(rains[i]) if rains[i] is not None else 0.0,
                "humidity": float(hums[i]) if hums[i] is not None else 0.0,
                "wind_speed": float(winds[i]) if winds[i] is not None else 0.0,
            }

            producer.send(TOPIC, value=msg)
            success_count += 1

            if (i + 1) % 100 == 0:
                producer.flush()
                progress = (i + 1) / n * 100
                print(f"📊 Progress: {i+1}/{n} ({progress:.1f}%)")

            time.sleep(SLEEP_BETWEEN_SEND)

        producer.flush()
        
        print("\n" + "=" * 60)
        print("✅ BATCH LOAD COMPLETE")
        print("=" * 60)
        print(f"✅ Success: {success_count}")
        print(f"📊 Total: {n}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        raise
    
    finally:
        producer.close()

if __name__ == "__main__":
    main()
