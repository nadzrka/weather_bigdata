from kafka import KafkaConsumer
from cassandra.cluster import Cluster
import json
from datetime import datetime

TOPIC_NAME = "weather_hourly"
CASSANDRA_HOST = "127.0.0.1"

# ===========================
#   SETUP KAFKA CONSUMER
# ===========================
consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='weather-group',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

# ===========================
#   CONNECT TO CASSANDRA
# ===========================
cluster = Cluster([CASSANDRA_HOST])
session = cluster.connect("weather")

# Prepared statement → lebih cepat & aman
insert_query = session.prepare("""
    INSERT INTO weather_data (city, datetime, temp, rain, humidity, wind_speed)
    VALUES (?, ?, ?, ?, ?, ?)
""")

print("Weather Consumer Started...\n")

# ===========================
#      MAIN LOOP
# ===========================
for message in consumer:
    try:
        data = message.value

        # Parsing datetime ISO ke Python datetime
        dt_value = datetime.fromisoformat(data["datetime"])

        session.execute(
            insert_query,
            (
                data["city"],
                dt_value,
                float(data["temp"]),
                float(data["rain"]),
                float(data["humidity"]),
                float(data["wind_speed"])
            )
        )

        print(f"Saved to Cassandra: {data}")

    except Exception as e:
        print("Error while saving:", e)

