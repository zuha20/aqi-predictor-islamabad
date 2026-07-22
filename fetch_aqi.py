import requests
import json
import os
import pandas as pd

import os
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv("AQICN_TOKEN")
CITY = "islamabad"
CSV_FILE = "readings.csv"


def fetch_current_aqi(city, token):
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    response = requests.get(url)
    data = response.json()

    if data.get("status") != "ok":
        raise ValueError(f"API returned an error: {data}")

    return data["data"]


def extract_row(raw):
    iaqi = raw.get("iaqi", {})

    row = {
        "timestamp": raw.get("time", {}).get("s"),
        "aqi": raw.get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "temperature": iaqi.get("t", {}).get("v"),
        "humidity": iaqi.get("h", {}).get("v"),
        "pressure": iaqi.get("p", {}).get("v"),
        "wind": iaqi.get("w", {}).get("v"),
    }
    return row


def save_row(row, csv_file):
    new_row_df = pd.DataFrame([row])

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = pd.concat([df, new_row_df], ignore_index=True)
    else:
        df = new_row_df

    df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    raw = fetch_current_aqi(CITY, API_TOKEN)
    row = extract_row(raw)
    save_row(row, CSV_FILE)

    print("Saved new reading:")
    print(row)
