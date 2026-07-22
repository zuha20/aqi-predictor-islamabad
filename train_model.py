import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# --- Connect and pull features back from Hopsworks ---
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
df = aqi_fg.read()

print(f"Pulled {len(df)} rows from the Feature Store")
print(df.head())

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# --- Sort by date (critical for time-series splitting) ---
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# --- Define features (X) and target (y) ---
# We drop columns that shouldn't be used as inputs:
# date (not numeric), and the other target columns we're not predicting right now
feature_cols = [col for col in df.columns if col not in
                 ["date", "target_day1", "target_day2", "target_day3"]]

X = df[feature_cols]
y = df["target_day1"]

# --- Time-based split: last 20% of rows = test set ---
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")

# --- Train a Ridge Regression model ---
model = Ridge()
model.fit(X_train, y_train)

# --- Evaluate ---
predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- Ridge Regression Results (predicting tomorrow's AQI) ---")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")
