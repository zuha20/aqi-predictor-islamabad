import pandas as pd

INPUT_FILE = "merged_historical.csv"
OUTPUT_FILE = "features.csv"

df = pd.read_csv(INPUT_FILE)

# Make sure the data is sorted by date, oldest first
# This matters because "change rate" and "future target" only make sense in order
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# --- Time-based features ---
df["day_of_week"] = df["date"].dt.dayofweek   # 0 = Monday, 6 = Sunday
df["day_of_month"] = df["date"].dt.day
df["month"] = df["date"].dt.month

# --- Derived feature: AQI change rate ---
# "median" is our stand-in for daily AQI (from the pollution file)
# .diff() computes: today's value minus yesterday's value
df["aqi_change_rate"] = df["median"].diff()

# --- Targets: AQI 1, 2, and 3 days into the future ---
# .shift(-1) pulls tomorrow's value into today's row
df["target_day1"] = df["median"].shift(-1)
df["target_day2"] = df["median"].shift(-2)
df["target_day3"] = df["median"].shift(-3)

# The first row has no "yesterday" to compare to (aqi_change_rate is empty there)
# The last 3 rows have no "future" to predict (targets are empty there)
# We drop those incomplete rows since a model can't learn from missing values
df = df.dropna().reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(df)} rows with {len(df.columns)} columns to {OUTPUT_FILE}")
print(df.head())
