import pandas as pd

pollution = pd.read_csv("historical_islamabad.csv")
weather = pd.read_csv("historical_weather.csv")

merged = pd.merge(pollution, weather, on="date", how="inner")

merged.to_csv("merged_historical.csv", index=False)

print(f"Merged dataset has {len(merged)} rows and {len(merged.columns)} columns")
print(merged.head())
