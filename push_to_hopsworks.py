import hopsworks
import pandas as pd

import os
from dotenv import load_dotenv

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
FEATURES_FILE = "features.csv"

# --- Connect to Hopsworks ---
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# --- Load our computed features ---
df = pd.read_csv(FEATURES_FILE)
df["date"] = pd.to_datetime(df["date"])

# --- Create (or get) the feature group ---
aqi_fg = fs.get_or_create_feature_group(
    name="aqi_features",
    version=2,
    primary_key=["date"],
    event_time="date",
    description="Daily AQI, weather, and derived features for Islamabad",
    time_travel_format="HUDI"
)

# --- Push the data in ---
aqi_fg.insert(df)

print(f"Successfully pushed {len(df)} rows to the 'aqi_features' feature group.")
