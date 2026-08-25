import pandas as pd

INPUT_FILE = "merged_historical.csv"
OUTPUT_FILE = "features.csv"

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# --- Time-based features ---
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_month"] = df["date"].dt.day
df["month"] = df["date"].dt.month

# --- Derived feature: AQI change rate ---
df["aqi_change_rate"] = df["median"].diff()

# --- NEW: Lag features (AQI from previous days) ---
df["aqi_lag_1"] = df["median"].shift(1)
df["aqi_lag_2"] = df["median"].shift(2)
df["aqi_lag_3"] = df["median"].shift(3)

# --- NEW: Rolling averages (recent trend) ---
df["aqi_rolling_mean_3"] = df["median"].shift(1).rolling(window=3).mean()
df["aqi_rolling_mean_7"] = df["median"].shift(1).rolling(window=7).mean()

# --- NEW: Weather lag features (yesterday's weather) ---
df["temp_lag_1"] = df["temperature_2m_mean"].shift(1)
df["humidity_proxy_lag_1"] = df["precipitation_sum"].shift(1)

# --- Targets: AQI 1, 2, and 3 days into the future ---
df["target_day1"] = df["median"].shift(-1)
df["target_day2"] = df["median"].shift(-2)
df["target_day3"] = df["median"].shift(-3)

df = df.dropna().reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(df)} rows with {len(df.columns)} columns to {OUTPUT_FILE}")
print(df.columns.tolist())
