import requests
import pandas as pd

LATITUDE = 33.7235
LONGITUDE = 73.11822
START_DATE = "2025-01-09"
END_DATE = "2026-07-21"
OUTPUT_FILE = "historical_weather.csv"


def fetch_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "windspeed_10m_max",
            "surface_pressure_mean"
        ],
        "timezone": "Asia/Karachi"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "daily" not in data:
        raise ValueError(f"API returned an error: {data}")

    return data["daily"]


if __name__ == "__main__":
    daily = fetch_weather(LATITUDE, LONGITUDE, START_DATE, END_DATE)

    df = pd.DataFrame(daily)
    df = df.rename(columns={"time": "date"})
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(df)} days of weather data to {OUTPUT_FILE}")
    print(df.head())
