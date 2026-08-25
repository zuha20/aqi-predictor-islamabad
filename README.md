# 🍃 aqi-predictor-islamabad

**An end-to-end, serverless machine learning pipeline that forecasts Islamabad's Air Quality Index (AQI) three days in advance.**

Built as part of a Data Science Internship at **10Pearls**.

---

## 📊 What This Project Does

Pearls AQI Predictor collects live weather and air-quality data every hour, engineers features from it, trains and compares multiple machine learning models, and serves 3-day AQI forecasts through a live interactive dashboard — with no server maintained by hand. Every stage runs automatically via GitHub Actions.

| Stage | What happens |
|---|---|
| 🌦️ **Feature Pipeline** | Fetches live weather + AQI data hourly, engineers time-based and lag/rolling features |
| 🗄️ **Feature Store** | Stores processed features in Hopsworks (versioned, shared by training + inference) |
| 🧠 **Training Pipeline** | Trains & evaluates 4 model types daily, logs the best to the Model Registry |
| 📈 **Explainability** | SHAP explains *why* each prediction was made |
| ⏱️ **Automation** | GitHub Actions run the pipelines hourly (features) and daily (training) |
| 🖥️ **Dashboard** | Streamlit app shows live AQI, 3-day forecast, trend, and hazard alerts |

---

## 🏗️ Architecture

```
 Open-Meteo API ──▶ fetch_aqi.py ──▶ readings.csv ──▶ GitHub Actions (hourly)
                                                             │
 AQICN Historical ──▶ merge_data.py ──▶ build_features.py ──▶ push_to_hopsworks.py
                                                             │
                                                     Hopsworks Feature Store
                                                             │
                                    ┌────────────────────────┴────────────────────────┐
                                    ▼                                                 ▼
                          train_model.py (daily)                              app.py (Streamlit)
                          Ridge / RF / GB / Neural Net                        live 3-day forecast
                                    │                                          + SHAP explanation
                                    ▼                                          + hazard alerts
                          Hopsworks Model Registry ◀──────────────────────────────────┘
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **ML:** Scikit-learn (Ridge Regression, Random Forest, Gradient Boosting), TensorFlow/Keras
- **Feature Store & Model Registry:** Hopsworks
- **Automation:** GitHub Actions (scheduled workflows)
- **Dashboard:** Streamlit + Plotly
- **Live Data:** Open-Meteo API (Air Quality + Weather) — switched from AQICN after discovering the Islamabad station's live feed had gone stale
- **Historical Data:** AQICN historical platform + Open-Meteo Historical Weather API
- **Explainability:** SHAP
- **Version Control:** Git & GitHub

---

## 📈 Results

Four model types were compared across three forecast horizons using RMSE, MAE, and R² on a chronological 80/20 train-test split:

| Model | Day 1 R² | Day 2 R² | Day 3 R² |
|---|---|---|---|
| **Ridge Regression** | **0.511** | 0.003 | -0.157 |
| Gradient Boosting | 0.294 | **0.024** | **-0.075** |
| Random Forest | 0.285 | — | — |
| Neural Network (TensorFlow) | -0.221 | -0.299 | -0.412 |

**Key finding:** Ridge Regression — the simplest model tested — consistently outperformed more complex models. With only ~430–540 daily training samples, higher-capacity models (Random Forest, Gradient Boosting, and especially the neural network) tended to overfit rather than generalize. Forecast accuracy also drops sharply beyond a 1-day horizon, reflecting the genuine difficulty of multi-day AQI forecasting from single-station data.

Full metrics and the SHAP feature-importance analysis are in the [project report](./report/Pearls_AQI_Predictor_Report.docx).

---

## 🖥️ Dashboard

The live dashboard shows:
- Current AQI with a color-coded gauge and category (Good → Hazardous)
- Live weather conditions (temperature, wind, precipitation, pressure)
- 3-day forecast with per-day category labels
- 30-day historical AQI trend
- Live SHAP explanation for the current prediction
- Automatic hazard alerts when predicted AQI crosses unhealthy thresholds



## ⚙️ Setup & Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/zuha20/aqi-predictor-islamabad.git
cd aqi-predictor-islamabad
```

### 2. Set up a Python virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
You'll know it worked when your terminal prompt starts with `(venv)`.

### 3. Install required packages
```bash
pip install requests pandas numpy scikit-learn tensorflow hopsworks joblib python-dotenv streamlit plotly shap pyarrow confluent-kafka
```

### 4. Add your Hopsworks API key
Create a `.env` file in the project root:
```bash
echo "HOPSWORKS_API_KEY=your_key_here" > .env
```
Replace `your_key_here` with your own key from [app.hopsworks.ai](https://app.hopsworks.ai) → Settings → API Keys. This file is intentionally excluded from the repo (see `.gitignore`) since API keys should never be committed.

### 5. Run the dashboard
```bash
streamlit run app.py
```
This will open automatically in your browser at `http://localhost:8501`. It connects live to the Hopsworks Feature Store and Model Registry, so no local data files are required to view the dashboard.


## 🤖 Automation

Two GitHub Actions workflows keep the system running with no manual intervention:

- **`feature_pipeline.yml`** — runs every hour, fetches live data, commits updated readings
- **`training_pipeline.yml`** — runs daily, retrains all models, updates the Model Registry

Both are visible under the repository's **Actions** tab, with full run history and logs.

---

## 📁 Project Structure

```
aqi_project/
├── fetch_aqi.py                 # Live data fetch (Open-Meteo)
├── fetch_historical_weather.py  # Historical weather backfill
├── merge_data.py                # Merges pollution + weather data
├── build_features.py            # Feature engineering
├── push_to_hopsworks.py         # Pushes features to Feature Store
├── train_model.py               # Trains & compares 4 model types
├── explain_model.py             # Standalone SHAP analysis
├── eda.py                       # Exploratory data analysis
├── app.py                       # Streamlit dashboard
├── .github/workflows/
│   ├── feature_pipeline.yml     # Hourly automation
│   └── training_pipeline.yml    # Daily automation
└── report/                      # Final project report
```

---

## 🔍 Known Limitations

- 2- and 3-day forecasts have limited predictive power given the current dataset size and feature set.
- Historical data provides only aggregate daily pollutant statistics, not per-pollutant (PM2.5, PM10, O₃, etc.) breakdowns.
- Only one AQICN monitoring station exists for Islamabad, limiting live-data redundancy.

See the full report for a detailed discussion of challenges encountered and possible future improvements.

---

## 👤 Author

**Zuhanoor** — Data Science Intern, 10Pearls
Bahria University, Islamabad
