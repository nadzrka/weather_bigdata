"""
Batch Consumer - Save Weather Data to Cassandra
Membaca dari Kafka topic 'weather_hourly' dan simpan ke Cassandra
"""

from kafka import KafkaConsumer
from cassandra.cluster import Cluster
import json
from datetime import datetime

# ===========================
# CONFIG
# ===========================
TOPIC_NAME = "weather_hourly"
CASSANDRA_HOST = "127.0.0.1"

# ===========================
# SETUP KAFKA CONSUMER
# ===========================
consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='weather-batch-group',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

# ===========================
# CONNECT TO CASSANDRA
# ===========================
print(" Connecting to Cassandra...")
cluster = Cluster([CASSANDRA_HOST])
session = cluster.connect()

# Create keyspace if not exists
session.execute("""
    CREATE KEYSPACE IF NOT EXISTS weather 
    WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
""")

session.set_keyspace("weather")

# Create table if not exists
session.execute("""
    CREATE TABLE IF NOT EXISTS weather_data (
        city text,
        datetime timestamp,
        temp double,
        rain double,
        humidity double,
        wind_speed double,
        PRIMARY KEY ((city), datetime)
    ) WITH CLUSTERING ORDER BY (datetime DESC)
""")

# Prepared statement
insert_query = session.prepare("""
    INSERT INTO weather_data (city, datetime, temp, rain, humidity, wind_speed)
    VALUES (?, ?, ?, ?, ?, ?)
""")

print(" Connected to Cassandra")
print(f" Listening to Kafka topic: {TOPIC_NAME}\n")

# ===========================
# MAIN LOOP
# ===========================
counter = 0

print("=" * 60)
print(" SAVING DATA TO CASSANDRA")
print("=" * 60)

for message in consumer:
    try:
        data = message.value

        # Parse datetime
        dt_value = datetime.fromisoformat(data["datetime"].replace('Z', '+00:00'))

        # Insert to Cassandra
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

        counter += 1

        if counter % 100 == 0:
            print(f" Saved {counter} records...")

    except Exception as e:
        print(f" Error: {e}")
