import requests
import os
import pandas as pd

CITY = "islamabad"
CSV_FILE = "readings.csv"
LATITUDE = 33.7235
LONGITUDE = 73.11822


def fetch_current_conditions(lat, lon):
    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {"latitude": lat, "longitude": lon, "current": "pm2_5,pm10,us_aqi"}
    aqi_resp = requests.get(aqi_url, params=aqi_params)
    aqi_data = aqi_resp.json()["current"]

    weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation"
    }
    weather_resp = requests.get(weather_url, params=weather_params)
    weather_data = weather_resp.json()["current"]

    return {
        "timestamp": aqi_data["time"],
        "aqi": aqi_data["us_aqi"],
        "pm25": aqi_data["pm2_5"],
        "temperature": weather_data["temperature_2m"],
        "humidity": weather_data["relative_humidity_2m"],
        "pressure": weather_data["surface_pressure"],
        "wind": weather_data["wind_speed_10m"],
        "precipitation": weather_data["precipitation"],
    }


def save_row(row, csv_file):
    new_row_df = pd.DataFrame([row])
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = pd.concat([df, new_row_df], ignore_index=True)
    else:
        df = new_row_df
    df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    row = fetch_current_conditions(LATITUDE, LONGITUDE)
    save_row(row, CSV_FILE)
    print("Saved new live reading:")
    print(row)
