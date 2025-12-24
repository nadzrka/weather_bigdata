"""
Training model untuk Real-Time Consumer
- TANPA lag features
- Simple per-message prediction
"""

from cassandra.cluster import Cluster
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ===========================
# CONFIG
# ===========================
FEATURE_COLS = [
    "temp", "rain", "humidity", "wind_speed",
    "hour", "dayofweek"
]

# ===========================
# LOAD DATA
# ===========================
print("📊 Connecting to Cassandra...")
cluster = Cluster(["127.0.0.1"])
session = cluster.connect("weather")

query = "SELECT city, datetime, temp, rain, humidity, wind_speed FROM weather_data"
print("📥 Loading data...")
df = pd.DataFrame(session.execute(query))

if df.empty:
    print("❌ No data in Cassandra!")
    print("💡 Run: python3 weather_producer.py")
    exit(1)

print(f"✅ Loaded {len(df)} rows")

# ===========================
# PREPROCESSING
# ===========================
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values(["city", "datetime"])

df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["temp_next"] = df.groupby("city")["temp"].shift(-1)

df = df.dropna()

print(f"✅ After preprocessing: {len(df)} rows")

# ===========================
# PREPARE DATA
# ===========================
X = df[FEATURE_COLS]
y = df["temp_next"]

print(f"   Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")

# ===========================
# TRAIN MODEL
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n🤖 Training Random Forest (Real-Time Model)...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ===========================
# EVALUATE
# ===========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Performance:")
print(f"   MAE:  {mae:.3f} °C")
print(f"   RMSE: {rmse:.3f} °C")
print(f"   R²:   {r2:.3f}")

# ===========================
# SAVE
# ===========================
bundle = {
    "model": model,
    "feature_cols": FEATURE_COLS,
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "trained_at": pd.Timestamp.now().isoformat()
}

joblib.dump(bundle, "temp_rf_realtime.pkl")

print("\n✅ Model saved: temp_rf_realtime.pkl")
print("   (for real-time streaming, no lag features)")
print("\n✅ Training complete!")
