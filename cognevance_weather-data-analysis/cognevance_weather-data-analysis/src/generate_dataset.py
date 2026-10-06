"""Generate a realistic synthetic daily weather dataset.

"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
dates = pd.date_range("2019-01-01", "2023-12-31", freq="D")
n = len(dates)
doy = dates.dayofyear.values
t = 2 * np.pi * (doy - 65) / 365.25  # temperature peaks in early June

temp_mean = 25 + 10 * np.sin(t) + rng.normal(0, 1.8, n)
temp_range = 12 - 3 * (np.abs(np.sin(2 * np.pi * (doy - 200) / 365.25))) + rng.normal(0, 1, n)
temp_max = temp_mean + temp_range / 2
temp_min = temp_mean - temp_range / 2

# Monsoon (Jul-Sep) = humid + rainy
monsoon = np.exp(-((doy - 215) ** 2) / (2 * 30 ** 2))
winter_rain = np.exp(-((doy - 25) ** 2) / (2 * 20 ** 2)) * 0.15
rain_prob = np.clip(0.04 + 0.6 * monsoon + winter_rain, 0, 0.9)
rain = np.where(rng.random(n) < rain_prob, rng.gamma(1.4, 4 + 6 * monsoon), 0).round(1)

humidity = np.clip(38 + 42 * monsoon + 12 * (rain > 0) + rng.normal(0, 6, n)
                   - 0.25 * (temp_mean - 24), 15, 100).round(1)
wind = np.clip(rng.gamma(4, 2.2, n) + 2 * monsoon, 0, 45).round(1)
pressure = (1008 - 8 * np.sin(t) + rng.normal(0, 2.5, n)).round(1)

df = pd.DataFrame({
    "date": dates.strftime("%Y-%m-%d"),
    "temp_max_c": temp_max.round(1),
    "temp_min_c": temp_min.round(1),
    "temp_avg_c": temp_mean.round(1),
    "humidity_pct": humidity,
    "rainfall_mm": rain,
    "wind_speed_kmph": wind,
    "pressure_hpa": pressure,
})

# ---- inject data-quality problems ----
for col, frac in [("temp_max_c", .02), ("temp_min_c", .02), ("humidity_pct", .03),
                  ("rainfall_mm", .02), ("wind_speed_kmph", .02), ("pressure_hpa", .015)]:
    df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan
df.loc[rng.choice(n, 6, replace=False), "temp_max_c"] = 999       # sensor error codes
df.loc[rng.choice(n, 5, replace=False), "humidity_pct"] = -9999    # sensor error codes
df.loc[rng.choice(n, 4, replace=False), "humidity_pct"] = 140      # impossible
df.loc[rng.choice(n, 4, replace=False), "rainfall_mm"] = -5        # impossible
df = pd.concat([df, df.sample(15, random_state=1)]).sample(frac=1, random_state=3)  # duplicates
df.loc[df.sample(8, random_state=5).index, "date"] = "not available"             # bad dates
df.to_csv("data/weather_raw.csv", index=False)
print("saved data/weather_raw.csv", df.shape)
