import os
import pandas as pd
import hopsworks
import joblib
import shap
from dotenv import load_dotenv
from sklearn.linear_model import Ridge

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# --- Connect and pull features ---
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

aqi_fg = fs.get_feature_group(name="aqi_features", version=2)
df = aqi_fg.read()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

target_cols = ["target_day1", "target_day2", "target_day3"]
feature_cols = [col for col in df.columns if col not in ["date"] + target_cols]

X = df[feature_cols]
y = df["target_day1"]

split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train = y[:split_index]

# --- Train the day1 Ridge model again (fresh, for explanation) ---
model = Ridge()
model.fit(X_train, y_train)

# --- Explain predictions using SHAP ---
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# --- Print feature importance summary (average impact per feature) ---
importance = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": abs(shap_values.values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("\n--- Feature Importance (SHAP) for tomorrow's AQI prediction ---")
print(importance.to_string(index=False))

# --- Save a summary plot ---
import matplotlib.pyplot as plt
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
print("\nSaved SHAP summary plot to shap_summary.png")
