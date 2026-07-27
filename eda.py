import os
import pandas as pd
import matplotlib.pyplot as plt
import hopsworks
from dotenv import load_dotenv

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

aqi_fg = fs.get_feature_group(name="aqi_features", version=2)
df = aqi_fg.read()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

os.makedirs("eda_plots", exist_ok=True)

# --- Plot 1: AQI over time ---
plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["median"], color="darkred", linewidth=1)
plt.title("Daily Median AQI Over Time - Islamabad")
plt.xlabel("Date")
plt.ylabel("AQI (median)")
plt.tight_layout()
plt.savefig("eda_plots/aqi_over_time.png")
plt.close()

# --- Plot 2: AQI distribution by month ---
plt.figure(figsize=(10, 5))
df.boxplot(column="median", by="month", ax=plt.gca())
plt.title("AQI Distribution by Month")
plt.suptitle("")
plt.xlabel("Month")
plt.ylabel("AQI (median)")
plt.tight_layout()
plt.savefig("eda_plots/aqi_by_month.png")
plt.close()

# --- Plot 3: AQI vs Temperature ---
plt.figure(figsize=(8, 6))
plt.scatter(df["temperature_2m_mean"], df["median"], alpha=0.5, color="steelblue")
plt.title("AQI vs Average Temperature")
plt.xlabel("Mean Temperature (°C)")
plt.ylabel("AQI (median)")
plt.tight_layout()
plt.savefig("eda_plots/aqi_vs_temperature.png")
plt.close()

# --- Plot 4: Correlation heatmap of key features ---
import numpy as np
key_cols = ["median", "temperature_2m_mean", "windspeed_10m_max",
            "surface_pressure_mean", "precipitation_sum", "aqi_lag_1"]
corr = df[key_cols].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(key_cols)), key_cols, rotation=45, ha="right")
plt.yticks(range(len(key_cols)), key_cols)
for i in range(len(key_cols)):
    for j in range(len(key_cols)):
        plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Correlation Between Key Features")
plt.tight_layout()
plt.savefig("eda_plots/correlation_heatmap.png")
plt.close()

print("Saved 4 EDA plots to eda_plots/")
print("\nSummary statistics:")
print(df["median"].describe())
