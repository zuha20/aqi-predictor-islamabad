import os
import pandas as pd
import numpy as np
import hopsworks
import streamlit as st
import plotly.graph_objects as go
import shap
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

st.set_page_config(page_title="Islamabad AQI", page_icon="🌤️", layout="centered")

def aqi_style(aqi):
    if aqi <= 50:
        return "Good", "☀️", "linear-gradient(160deg, #4FACFE 0%, #00C6FB 100%)"
    elif aqi <= 100:
        return "Moderate", "🌤️", "linear-gradient(160deg, #F6D365 0%, #FDA085 100%)"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "😷", "linear-gradient(160deg, #FDA085 0%, #F76B1C 100%)"
    elif aqi <= 200:
        return "Unhealthy", "🌫️", "linear-gradient(160deg, #F76B1C 0%, #C0392B 100%)"
    elif aqi <= 300:
        return "Very Unhealthy", "⚠️", "linear-gradient(160deg, #8E54E9 0%, #4776E6 100%)"
    else:
        return "Hazardous", "☠️", "linear-gradient(160deg, #414345 0%, #232526 100%)"

cat0, icon0, grad0 = "Good", "☀️", "linear-gradient(160deg, #4FACFE 0%, #00C6FB 100%)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Poppins', sans-serif; }}
.stApp {{ background: #F0F2F5; }}
.block-container {{ max-width: 480px; padding-top: 1rem; }}
header {{ visibility: hidden; }}
[data-testid="stHeader"] {{ display: none; }}
[data-testid="stToolbar"] {{ display: none; }}
footer {{ visibility: hidden; }}

.hero {{
    border-radius: 32px; padding: 30px 26px 26px 26px; color: white;
    box-shadow: 0 12px 28px rgba(0,0,0,0.15); position: relative; overflow: hidden;
}}
.hero-location {{ font-size: 15px; font-weight: 600; opacity: 0.95; }}
.hero-date {{ font-size: 12.5px; opacity: 0.8; margin-top: -2px; }}
.hero-icon {{ font-size: 56px; text-align: center; margin: 6px 0 -6px 0; }}
.hero-value {{ font-size: 76px; font-weight: 800; text-align: center; line-height: 1; }}
.hero-cat {{ text-align: center; font-size: 16px; font-weight: 600; opacity: 0.95; margin-top: 2px; }}
.hero-desc {{ text-align: center; font-size: 12.5px; opacity: 0.85; margin-top: 8px; padding: 0 10px; }}
.live-tag {{
    display: inline-flex; align-items: center; gap: 6px; background: rgba(255,255,255,0.22);
    padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600; margin-top: 12px;
}}
.dot {{ width: 6px; height: 6px; border-radius: 50%; background: #fff; }}

.glass-row {{ display: flex; gap: 10px; margin-top: -20px; padding: 0 4px; }}
.glass-chip {{
    flex: 1; background: rgba(255,255,255,0.9); backdrop-filter: blur(10px);
    border-radius: 18px; padding: 14px 8px; text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}}
.glass-chip .icon {{ font-size: 20px; }}
.glass-chip .val {{ font-size: 16px; font-weight: 700; color: #1F2937; margin-top: 4px; }}
.glass-chip .label {{ font-size: 10.5px; color: #9CA3AF; margin-top: 1px; }}

.section-title {{ font-size: 15px; font-weight: 700; color: #1F2937; margin: 26px 4px 10px 4px; }}

.forecast-strip {{ display: flex; gap: 10px; overflow-x: auto; padding: 2px 2px 8px 2px; }}
.forecast-pill {{
    min-width: 92px; background: white; border-radius: 20px; padding: 16px 8px;
    text-align: center; box-shadow: 0 4px 14px rgba(0,0,0,0.06);
}}
.forecast-pill .day {{ font-size: 11.5px; font-weight: 700; color: #9CA3AF; text-transform: uppercase; }}
.forecast-pill .icon {{ font-size: 26px; margin: 8px 0; }}
.forecast-pill .val {{ font-size: 20px; font-weight: 800; }}
.forecast-pill .cat {{ font-size: 9.5px; font-weight: 600; margin-top: 2px; }}

.card-white {{
    background: white; border-radius: 20px; padding: 18px; box-shadow: 0 4px 14px rgba(0,0,0,0.05);
}}
</style>
""", unsafe_allow_html=True)


def guidance_text(aqi):
    if aqi <= 50: return "Air quality is good. Perfect day to be outside."
    elif aqi <= 100: return "Air quality is acceptable for most. Sensitive groups take it easy outdoors."
    elif aqi <= 150: return "Sensitive groups should limit prolonged outdoor exertion."
    elif aqi <= 200: return "Everyone may feel effects. Limit outdoor exertion."
    else: return "Health warning: avoid outdoor exertion."


@st.cache_resource
def connect_to_hopsworks():
    return hopsworks.login(api_key_value=HOPSWORKS_API_KEY)


@st.cache_data(ttl=3600)
def load_latest_features(_project):
    fs = _project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=2)
    df = fg.read()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_models(_project):
    import joblib
    mr = _project.get_model_registry()
    models = {}
    for target in ["target_day1", "target_day2", "target_day3"]:
        m = mr.get_best_model(f"aqi_ridge_{target}", "rmse", "min")
        d = m.download()
        models[target] = joblib.load(f"{d}/model.pkl")
    return models


def compute_live_features(df_hist, readings_path="readings.csv"):
    if not os.path.exists(readings_path):
        return None, None
    r = pd.read_csv(readings_path)
    r["timestamp"] = pd.to_datetime(r["timestamp"], format="mixed")
    r["date"] = r["timestamp"].dt.date
    today = r["date"].max()
    today_rows = r[r["date"] == today]
    if len(today_rows) == 0 or (datetime.now().date() - today) > timedelta(days=3):
        return None, None

    aqi_vals = today_rows["aqi"].astype(float)
        precip_col = today_rows["precipitation"] if "precipitation" in today_rows.columns else pd.Series([0.0])
    row = {
        "date": pd.to_datetime(today), "min": aqi_vals.min(), "max": aqi_vals.max(),
        "median": aqi_vals.median(), "q1": aqi_vals.quantile(0.25), "q3": aqi_vals.quantile(0.75),
        "stdev": aqi_vals.std() if len(aqi_vals) > 1 else 0.0, "count": len(aqi_vals),
        "temperature_2m_max": today_rows["temperature"].max(),
        "temperature_2m_min": today_rows["temperature"].min(),
        "temperature_2m_mean": today_rows["temperature"].mean(),
        "current_temp": today_rows["temperature"].iloc[-1],
        "precipitation_sum": precip_col.sum(),
        "windspeed_10m_max": today_rows["wind"].max(),
        "surface_pressure_mean": today_rows["pressure"].mean(),
    }
    dt = pd.to_datetime(today)
    row["day_of_week"], row["day_of_month"], row["month"] = dt.dayofweek, dt.day, dt.month
    recent = df_hist["median"].tail(7).tolist()
    prev = recent[-1] if recent else row["median"]
    row["aqi_change_rate"] = row["median"] - prev
    row["aqi_lag_1"] = recent[-1] if len(recent) >= 1 else row["median"]
    row["aqi_lag_2"] = recent[-2] if len(recent) >= 2 else row["aqi_lag_1"]
    row["aqi_lag_3"] = recent[-3] if len(recent) >= 3 else row["aqi_lag_2"]
    row["aqi_rolling_mean_3"] = np.mean(recent[-3:]) if recent else row["median"]
    row["aqi_rolling_mean_7"] = np.mean(recent[-7:]) if recent else row["median"]
    row["temp_lag_1"] = df_hist["temperature_2m_mean"].iloc[-1] if len(df_hist) else row["temperature_2m_mean"]
    row["humidity_proxy_lag_1"] = df_hist["precipitation_sum"].iloc[-1] if len(df_hist) else 0.0
    return pd.DataFrame([row]), dt


with st.spinner("Loading..."):
    project = connect_to_hopsworks()
    df = load_latest_features(project)
    models = load_models(project)

live_row, live_date = compute_live_features(df)
is_live = live_row is not None
latest_row = live_row if is_live else df.iloc[[-1]]
latest_date = live_date if is_live else pd.to_datetime(latest_row["date"].values[0])
current_aqi = latest_row["median"].values[0]

feature_cols = [c for c in df.columns if c not in ["date", "target_day1", "target_day2", "target_day3"]]
X_latest = latest_row[feature_cols]
predictions = {t: models[t].predict(X_latest)[0] for t in ["target_day1", "target_day2", "target_day3"]}

cat, icon, grad = aqi_style(current_aqi)

# --- Hero ---
st.markdown(f"""
<div class="hero" style="background:{grad};">
    <div class="hero-location">📍 Islamabad, Pakistan</div>
    <div class="hero-date">{latest_date.strftime('%A, %B %d')}</div>
    <div class="hero-icon">{icon}</div>
    <div class="hero-value">{current_aqi:.0f}</div>
    <div class="hero-cat">{cat}</div>
    <div class="hero-desc">{guidance_text(current_aqi)}</div>
    <div style="text-align:center;">
        <span class="live-tag"><span class="dot"></span> {'Live data' if is_live else 'Recent historical data'}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Glass stat row (overlapping hero) ---
st.markdown(f"""
<div class="glass-row">
    <div class="glass-chip"><div class="icon">🌡️</div><div class="val">{(latest_row['current_temp'].values[0] if 'current_temp' in latest_row.columns else latest_row['temperature_2m_mean'].values[0]):.0f}°C</div><div class="label">Temp</div></div>
    <div class="glass-chip"><div class="icon">💨</div><div class="val">{latest_row['windspeed_10m_max'].values[0]:.0f} km/h</div><div class="label">Wind</div></div>
    <div class="glass-chip"><div class="icon">🌧️</div><div class="val">{latest_row['precipitation_sum'].values[0]:.1f} mm</div><div class="label">Rain</div></div>
    <div class="glass-chip"><div class="icon">🧭</div><div class="val">{latest_row['surface_pressure_mean'].values[0]:.0f}</div><div class="label">hPa</div></div>
</div>
""", unsafe_allow_html=True)

# --- 3-day forecast strip ---
st.markdown('<div class="section-title">3-Day Forecast</div>', unsafe_allow_html=True)
labels = ["Tomorrow", "Day 2", "Day 3"]
strip = '<div class="forecast-strip">'
for i, target in enumerate(["target_day1", "target_day2", "target_day3"]):
    pred = predictions[target]
    cat_f, icon_f, _ = aqi_style(pred)
    color_f = {"Good": "#22C55E", "Moderate": "#D97706", "Unhealthy for Sensitive Groups": "#EA580C",
               "Unhealthy": "#DC2626", "Very Unhealthy": "#9333EA", "Hazardous": "#7F1D1D"}[cat_f]
    strip += f"""
    <div class="forecast-pill">
        <div class="day">{labels[i]}</div>
        <div class="icon">{icon_f}</div>
        <div class="val" style="color:{color_f};">{pred:.0f}</div>
        <div class="cat" style="color:{color_f};">{cat_f}</div>
    </div>"""
strip += '</div>'
st.markdown(strip, unsafe_allow_html=True)

# --- Hazard alert ---
max_predicted = max(predictions.values())
if max_predicted > 150:
    st.error(f"⚠️ Hazard Alert: AQI may reach {max_predicted:.0f} within 3 days.")
elif max_predicted > 100:
    st.warning(f"⚠️ Moderate concern: AQI may reach {max_predicted:.0f}.")
else:
    st.success("✅ No hazardous AQI levels predicted in the next 3 days.")

# --- Trend ---
st.markdown('<div class="section-title">30-Day Trend</div>', unsafe_allow_html=True)
recent = df.tail(30)
fig = go.Figure(go.Scatter(x=recent["date"], y=recent["median"], mode="lines", fill="tozeroy",
                            line=dict(color="#4FACFE", width=3), fillcolor="rgba(79,172,254,0.15)"))
fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                   paper_bgcolor="white", xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F0F2F5"))
st.plotly_chart(fig, use_container_width=True)

# --- Why this prediction ---
st.markdown('<div class="section-title">Why This Prediction</div>', unsafe_allow_html=True)
split_index = int(len(df) * 0.8)
X_train = df[feature_cols][:split_index]
explainer = shap.Explainer(models["target_day1"], X_train)
shap_vals = explainer(X_latest)
shap_df = pd.DataFrame({"feature": feature_cols, "shap_value": shap_vals.values[0]}).sort_values(
    "shap_value", key=abs, ascending=True).tail(8)
colors_bar = ["#EA580C" if v > 0 else "#22C55E" for v in shap_df["shap_value"]]
fig2 = go.Figure(go.Bar(x=shap_df["shap_value"], y=shap_df["feature"], orientation="h", marker_color=colors_bar))
fig2.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(gridcolor="#F0F2F5"))
st.plotly_chart(fig2, use_container_width=True)
st.caption("🟠 Pushes higher · 🟢 Pushes lower")

st.markdown('<div style="text-align:center; color:#9CA3AF; font-size:11.5px; margin-top:20px;">Pearls AQI Predictor · 10Pearls · Built by Zuhanoor</div>', unsafe_allow_html=True)
