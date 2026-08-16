import os
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.ensemble import GradientBoostingRegressor

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# --- Connect and pull features from Hopsworks ---
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()
mr = project.get_model_registry()

aqi_fg = fs.get_feature_group(name="aqi_features", version=2)
df = aqi_fg.read()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

target_cols = ["target_day1", "target_day2", "target_day3"]
feature_cols = [col for col in df.columns if col not in ["date"] + target_cols]

X = df[feature_cols]
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]

# --- Train, evaluate, and save one model per target day ---
for target in target_cols:
    y = df[target]
    y_train, y_test = y[:split_index], y[split_index:]

    model = Ridge()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n--- Ridge Regression: {target} ---")
    print(f"RMSE: {rmse:.2f}  MAE: {mae:.2f}  R²: {r2:.3f}")

    # Save model locally first (Hopsworks needs a file to upload)
    model_dir = f"model_{target}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/model.pkl")

    # Register the model in the Hopsworks Model Registry
    aqi_model = mr.python.create_model(
        name=f"aqi_ridge_{target}",
        metrics={"rmse": rmse, "mae": mae, "r2": r2},
        description=f"Ridge Regression predicting AQI {target.replace('target_', '')}"
    )
    aqi_model.save(model_dir)

    print(f"Saved model '{aqi_model.name}' to Model Registry")

# --- Try Gradient Boosting as a second model ---
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    gb_predictions = gb_model.predict(X_test)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))
    gb_mae = mean_absolute_error(y_test, gb_predictions)
    gb_r2 = r2_score(y_test, gb_predictions)

    print(f"--- Gradient Boosting: {target} ---")
    print(f"RMSE: {gb_rmse:.2f}  MAE: {gb_mae:.2f}  R²: {gb_r2:.3f}")
# --- Neural Network (deep learning) model ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nn_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    nn_model.compile(optimizer="adam", loss="mse")
    nn_model.fit(X_train_scaled, y_train, epochs=50, verbose=0, validation_split=0.1)

    nn_predictions = nn_model.predict(X_test_scaled, verbose=0).flatten()
    nn_rmse = np.sqrt(mean_squared_error(y_test, nn_predictions))
    nn_mae = mean_absolute_error(y_test, nn_predictions)
    nn_r2 = r2_score(y_test, nn_predictions)

    print(f"--- Neural Network: {target} ---")
    print(f"RMSE: {nn_rmse:.2f}  MAE: {nn_mae:.2f}  R²: {nn_r2:.3f}")
