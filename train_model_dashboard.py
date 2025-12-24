"""
Training model untuk Dashboard (Historical Analysis)
- Menggunakan lag features
- Window-based prediction (LOOK_BACK=48)
"""

from cassandra.cluster import Cluster
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ===========================
# CONFIG
# ===========================
LOOK_BACK = 48
FEATURE_COLS = [
    "temp", "rain", "humidity", "wind_speed",
    "hour", "dayofweek",
    "temp_lag1", "temp_lag3", "temp_lag6"
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
df["temp_lag1"] = df.groupby("city")["temp"].shift(1)
df["temp_lag3"] = df.groupby("city")["temp"].shift(3)
df["temp_lag6"] = df.groupby("city")["temp"].shift(6)
df["temp_next"] = df.groupby("city")["temp"].shift(-1)

df = df.dropna()

print(f"✅ After preprocessing: {len(df)} rows")

# ===========================
# CREATE SEQUENCES
# ===========================
def create_sequences(group_df, look_back):
    sequences = []
    targets = []
    
    features = group_df[FEATURE_COLS].values
    target = group_df["temp_next"].values
    
    for i in range(look_back, len(features)):
        window = features[i-look_back:i]
        sequences.append(window)
        targets.append(target[i])
    
    return sequences, targets

X_list = []
y_list = []

print("🔄 Creating sequences...")
for city, group in df.groupby("city"):
    if len(group) >= LOOK_BACK + 10:
        seqs, targs = create_sequences(group, LOOK_BACK)
        X_list.extend(seqs)
        y_list.extend(targs)
        print(f"  {city}: {len(seqs)} sequences")

X = np.array(X_list)
y = np.array(y_list)

print(f"\n✅ Total sequences: {len(X)}")
print(f"   Shape: X={X.shape}, y={y.shape}")

# ===========================
# SCALE & FLATTEN
# ===========================
n_samples, n_timesteps, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)

X_flat = X_scaled.reshape(n_samples, -1)

print(f"✅ Flattened shape: {X_flat.shape}")

# ===========================
# TRAIN MODEL
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=42
)

print("\n🤖 Training Random Forest...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
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
joblib.dump(model, "temp_rf_multifeature.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Models saved:")
print("   - temp_rf_multifeature.pkl")
print("   - scaler.pkl")

metadata = {
    "look_back": LOOK_BACK,
    "feature_cols": FEATURE_COLS,
    "n_features": n_features,
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "trained_at": pd.Timestamp.now().isoformat()
}

import json
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("   - model_metadata.json")
print("\n✅ Training complete!")
