"""
Training model untuk Dashboard (Historical Analysis)
- Lag features
- Window-based prediction (LOOK_BACK=48)
- Time-based split (no leakage)
- Correct window->target alignment
"""

from cassandra.cluster import Cluster
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json

# ===========================
# CONFIG
# ===========================
LOOK_BACK = 48
TEST_RATIO = 0.2
MIN_SEQS_PER_CITY = 10  # minimal sequence supaya kota ikut
FEATURE_COLS = [
    "temp", "rain", "humidity", "wind_speed",
    "hour", "dayofweek",
    "temp_lag1", "temp_lag3", "temp_lag6"
]

# ===========================
# LOAD DATA
# ===========================
print(" Connecting to Cassandra...")
cluster = Cluster(["127.0.0.1"])
session = cluster.connect("weather")

query = "SELECT city, datetime, temp, rain, humidity, wind_speed FROM weather_data"
print(" Loading data...")
df = pd.DataFrame(session.execute(query))

if df.empty:
    print(" No data in Cassandra!")
    print(" Pastikan batch pipeline sudah jalan (producer+consumer) dan data masuk ke weather_data.")
    raise SystemExit(1)

print(f" Loaded {len(df)} rows")

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

# NOTE: temp_next tidak dipakai lagi untuk menghindari off-by-one
# df["temp_next"] = df.groupby("city")["temp"].shift(-1)

df = df.dropna()
print(f" After preprocessing: {len(df)} rows")

# ===========================
# CREATE SEQUENCES
# ===========================
def create_sequences(group_df: pd.DataFrame, look_back: int):
    """
    Window: features[i-look_back : i]  (berakhir di i-1)
    Target: temp[i]                   (jam berikutnya setelah window)
    """
    features = group_df[FEATURE_COLS].values
    target = group_df["temp"].values  # <-- FIX alignment

    sequences, targets = [], []
    for i in range(look_back, len(features)):
        window = features[i - look_back : i]
        sequences.append(window)
        targets.append(target[i])

    return np.array(sequences), np.array(targets)

# ===========================
# TIME-BASED SPLIT PER CITY
# ===========================
X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

print(" Creating sequences + time split per city...")
used_cities = 0

for city, group in df.groupby("city"):
    if len(group) < LOOK_BACK + MIN_SEQS_PER_CITY:
        print(f"  {city}: skipped (rows={len(group)} < {LOOK_BACK + MIN_SEQS_PER_CITY})")
        continue

    X_city, y_city = create_sequences(group, LOOK_BACK)
    if len(X_city) < MIN_SEQS_PER_CITY:
        print(f"  {city}: skipped (seqs={len(X_city)} < {MIN_SEQS_PER_CITY})")
        continue

    split_idx = int(len(X_city) * (1 - TEST_RATIO))
    X_train_list.append(X_city[:split_idx])
    y_train_list.append(y_city[:split_idx])
    X_test_list.append(X_city[split_idx:])
    y_test_list.append(y_city[split_idx:])

    print(f"  {city}: train={split_idx}, test={len(X_city) - split_idx}")
    used_cities += 1

if used_cities == 0:
    print(" Tidak ada kota yang memenuhi syarat untuk training.")
    raise SystemExit(1)

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)
X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

print(f"\n Total train sequences: {len(X_train)}  | shape={X_train.shape}")
print(f" Total test  sequences: {len(X_test)}   | shape={X_test.shape}")

# ===========================
# SCALE (FIT ONLY TRAIN) & FLATTEN
# ===========================
n_train, n_timesteps, n_features = X_train.shape
n_test = X_test.shape[0]

scaler = StandardScaler()

X_train_2d = X_train.reshape(-1, n_features)
X_test_2d = X_test.reshape(-1, n_features)

X_train_scaled = scaler.fit_transform(X_train_2d)   # <-- FIT di train saja
X_test_scaled = scaler.transform(X_test_2d)

X_train_scaled = X_train_scaled.reshape(n_train, n_timesteps, n_features)
X_test_scaled = X_test_scaled.reshape(n_test, n_timesteps, n_features)

X_train_flat = X_train_scaled.reshape(n_train, -1)
X_test_flat = X_test_scaled.reshape(n_test, -1)

print(f" Flattened: train={X_train_flat.shape}, test={X_test_flat.shape}")

# ===========================
# TRAIN MODEL
# ===========================
print("\n Training Random Forest...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_flat, y_train)

# ===========================
# EVALUATE
# ===========================
y_pred = model.predict(X_test_flat)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n Model Performance (time-based split):")
print(f"   MAE:  {mae:.3f} °C")
print(f"   RMSE: {rmse:.3f} °C")
print(f"   R²:   {r2:.3f}")

# ===========================
# SAVE
# ===========================
joblib.dump(model, "temp_rf_multifeature.pkl")
joblib.dump(scaler, "scaler.pkl")

metadata = {
    "look_back": LOOK_BACK,
    "test_ratio": TEST_RATIO,
    "feature_cols": FEATURE_COLS,
    "n_features": int(n_features),
    "train_sequences": int(len(X_train)),
    "test_sequences": int(len(X_test)),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "trained_at": pd.Timestamp.now().isoformat(),
    "split_strategy": "per-city time-based (first 80% train, last 20% test)",
    "target_definition": "temp at time i (predict next hour after window end)"
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n Models saved:")
print("   - temp_rf_multifeature.pkl")
print("   - scaler.pkl")
print("   - model_metadata.json")
print(" Training complete!")
