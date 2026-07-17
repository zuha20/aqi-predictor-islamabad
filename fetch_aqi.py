import requests
import json

API_TOKEN = "281848c24bdeb71347e246adbd3798fafbd5c8a4"
CITY = "islamabad"

def fetch_current_aqi(city, token):
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    response = requests.get(url)
    data = response.json()

    if data.get("status") != "ok":
        raise ValueError(f"API returned an error: {data}")

    return data["data"]


if __name__ == "__main__":
    raw = fetch_current_aqi(CITY, API_TOKEN)
    print(json.dumps(raw, indent=2))

    print("\n--- Summary ---")
    print(f"City: {raw.get('city', {}).get('name')}")
    print(f"AQI: {raw.get('aqi')}")
    print(f"Timestamp: {raw.get('time', {}).get('s')}")
