"""Weather Data Analysis - cleaning, trend analysis, charts and insight report."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "outputs" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load():
    return pd.read_csv(ROOT / "data" / "weather_raw.csv")


def clean(raw: pd.DataFrame):
    log = {"raw_rows": len(raw)}
    df = raw.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    log["invalid_dates_removed"] = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])

    before = len(df)
    df = df.drop_duplicates(subset="date", keep="first")
    log["duplicates_removed"] = before - len(df)

    # Valid physical ranges -> anything outside becomes NaN (incorrect record)
    rules = {"temp_max_c": (-10, 55), "temp_min_c": (-10, 50), "temp_avg_c": (-10, 50),
             "humidity_pct": (0, 100), "rainfall_mm": (0, 500),
             "wind_speed_kmph": (0, 150), "pressure_hpa": (850, 1090)}
    bad = 0
    for col, (lo, hi) in rules.items():
        mask = (df[col] < lo) | (df[col] > hi)
        bad += int(mask.sum())
        df.loc[mask, col] = np.nan
    log["incorrect_values_set_to_nan"] = bad

    df = df.sort_values("date").set_index("date")
    log["missing_before_imputation"] = int(df.isna().sum().sum())
    # Time-based interpolation suits continuous series; rainfall gaps -> 0 (no report = no rain)
    cont = [c for c in df.columns if c != "rainfall_mm"]
    df[cont] = df[cont].interpolate(method="time", limit_direction="both")
    df["rainfall_mm"] = df["rainfall_mm"].fillna(0)
    log["missing_after_imputation"] = int(df.isna().sum().sum())
    log["clean_rows"] = len(df)

    df["year"], df["month"] = df.index.year, df.index.month
    df["season"] = df["month"].map(lambda m: "Winter" if m in (12, 1, 2) else
                                   "Summer" if m in (3, 4, 5, 6) else
                                   "Monsoon" if m in (7, 8, 9) else "Post-monsoon")
    df["rainy_day"] = df["rainfall_mm"] >= 2.5
    return df, log


def analyse(df: pd.DataFrame):
    nyears = df["year"].nunique()
    monthly = df.groupby("month").agg(
        avg_temp=("temp_avg_c", "mean"), max_temp=("temp_max_c", "mean"),
        min_temp=("temp_min_c", "mean"), humidity=("humidity_pct", "mean"),
        rain_total=("rainfall_mm", "sum"), rainy_days=("rainy_day", "sum"))
    monthly["rain_total"] /= nyears   # mean monthly rainfall per year
    monthly["rainy_days"] /= nyears
    monthly.index = MONTHS
    yearly = df.groupby("year").agg(avg_temp=("temp_avg_c", "mean"),
                                    rain=("rainfall_mm", "sum"), humidity=("humidity_pct", "mean"))
    season = df.groupby("season").agg(avg_temp=("temp_avg_c", "mean"), humidity=("humidity_pct", "mean"),
                                      rain=("rainfall_mm", "sum"))
    corr = df[["temp_avg_c", "humidity_pct", "rainfall_mm", "wind_speed_kmph", "pressure_hpa"]].corr()

    def save(fig, name):
        fig.tight_layout(); fig.savefig(CHARTS / name, dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df.index, df["temp_avg_c"], lw=.4, alpha=.5, label="Daily avg")
    ax.plot(df.index, df["temp_avg_c"].rolling(30, center=True).mean(), lw=2, color="crimson", label="30-day mean")
    ax.set(title="Daily average temperature, 2019-2023", ylabel="°C"); ax.legend()
    save(fig, "01_temperature_trend.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(MONTHS, monthly["min_temp"], monthly["max_temp"], alpha=.25, color="orange")
    ax.plot(MONTHS, monthly["avg_temp"], marker="o", color="firebrick", label="Mean")
    ax.set(title="Typical monthly temperature range", ylabel="°C"); ax.legend()
    save(fig, "02_monthly_temperature_range.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(MONTHS, monthly["rain_total"], color="steelblue")
    ax.set(title="Average rainfall per month", ylabel="mm"); ax2 = ax.twinx()
    ax2.plot(MONTHS, monthly["rainy_days"], color="darkorange", marker="o"); ax2.set_ylabel("rainy days"); ax2.grid(False)
    save(fig, "03_monthly_rainfall.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(x=df.index.month, y=df["humidity_pct"].values, ax=ax, color="skyblue")
    ax.set_xticks(range(12)); ax.set_xticklabels(MONTHS)
    ax.set(title="Humidity distribution by month", xlabel="", ylabel="%")
    save(fig, "04_humidity_by_month.png")

    pivot = df.pivot_table(index="year", columns="month", values="temp_avg_c", aggfunc="mean")
    pivot.columns = MONTHS
    fig, ax = plt.subplots(figsize=(11, 3.8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlBu_r", ax=ax, cbar_kws={"label": "°C"})
    ax.set(title="Mean temperature: year x month", ylabel="")
    save(fig, "05_temperature_heatmap.png")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation between weather variables")
    save(fig, "06_correlation.png")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.scatterplot(data=df, x="temp_avg_c", y="humidity_pct", hue="season", s=12, alpha=.6, ax=ax)
    ax.set(title="Temperature vs humidity by season", xlabel="Avg temperature (°C)", ylabel="Humidity (%)")
    save(fig, "07_temp_vs_humidity.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(x=yearly.index, y=yearly["avg_temp"], ax=axes[0], color="tomato")
    axes[0].set(title="Mean temperature by year", ylabel="°C", xlabel="")
    axes[0].set_ylim(yearly["avg_temp"].min() - 1, yearly["avg_temp"].max() + .5)
    sns.barplot(x=yearly.index, y=yearly["rain"], ax=axes[1], color="steelblue")
    axes[1].set(title="Total rainfall by year", ylabel="mm", xlabel="")
    save(fig, "08_yearly_summary.png")

    return monthly, yearly, season, corr


def write_report(df, log, monthly, yearly, season, corr):
    hot, cold = monthly["avg_temp"].idxmax(), monthly["avg_temp"].idxmin()
    wet, humid = monthly["rain_total"].idxmax(), monthly["humidity"].idxmax()
    dry = monthly["humidity"].idxmin()
    monsoon_share = df[df.season == "Monsoon"]["rainfall_mm"].sum() / df["rainfall_mm"].sum() * 100
    trend = np.polyfit(np.arange(len(yearly)), yearly["avg_temp"].values, 1)[0]
    hottest_day = df["temp_max_c"].idxmax()
    r_th = corr.loc["temp_avg_c", "humidity_pct"]; r_rh = corr.loc["rainfall_mm", "humidity_pct"]
    seasons = ", ".join(f"{s} {v:.1f} °C" for s, v in season["avg_temp"].items())

    md = f"""# Weather Data Analysis - Insights Report

**Period:** {df.index.min():%d %b %Y} - {df.index.max():%d %b %Y} ({len(df):,} daily records)

## 1. Data cleaning summary
| Step | Result |
|---|---|
| Raw rows | {log['raw_rows']:,} |
| Rows with unparseable dates removed | {log['invalid_dates_removed']} |
| Duplicate dates removed | {log['duplicates_removed']} |
| Impossible / sensor-error values set to missing | {log['incorrect_values_set_to_nan']} |
| Missing cells before imputation | {log['missing_before_imputation']} |
| Missing cells after imputation | {log['missing_after_imputation']} |
| Clean rows | {log['clean_rows']:,} |

Continuous variables were filled by time-based interpolation; missing rainfall was treated as 0 mm.

## 2. Key findings
* **Temperature:** {hot} is the hottest month (mean {monthly.loc[hot,'avg_temp']:.1f} °C, average daily high {monthly.loc[hot,'max_temp']:.1f} °C); {cold} is the coldest ({monthly.loc[cold,'avg_temp']:.1f} °C). Hottest single day: {hottest_day:%d %b %Y} at {df['temp_max_c'].max():.1f} °C.
* **Rainfall:** {wet} is the wettest month (about {monthly.loc[wet,'rain_total']:.0f} mm on average). The monsoon season (Jul-Sep) delivers **{monsoon_share:.0f}%** of total rainfall.
* **Humidity:** highest in {humid} ({monthly.loc[humid,'humidity']:.0f}%), lowest in {dry} ({monthly.loc[dry,'humidity']:.0f}%).
* **Seasonal contrast (mean temperature):** {seasons}.
* **Relationships:** temperature vs humidity r = {r_th:.2f}; rainfall vs humidity r = {r_rh:.2f}. Rainy days are noticeably more humid.
* **Year-on-year:** fitted trend in annual mean temperature is {trend:+.2f} °C/year over {len(yearly)} years (too short a window to say anything about climate change).

## 3. Monthly comparison
{monthly.round(1).to_markdown()}

## 4. Charts
All charts are in `outputs/charts/`:
1. `01_temperature_trend.png` 2. `02_monthly_temperature_range.png` 3. `03_monthly_rainfall.png` 4. `04_humidity_by_month.png`
5. `05_temperature_heatmap.png` 6. `06_correlation.png` 7. `07_temp_vs_humidity.png` 8. `08_yearly_summary.png`

## 5. Recommendations
* Schedule outdoor events and travel in **Oct-Mar** when it is dry and moderate; avoid the {hot} heat peak and the {wet} peak rainfall.
* Agriculture: plan sowing and irrigation around the monsoon concentration ({monsoon_share:.0f}% of rain in three months); store water for the dry months.
* Energy: cooling demand peaks around {hot}; heating demand is minimal in this climate.

## 6. Limitations
The dataset is synthetic (generated by `src/generate_dataset.py`) to mimic a north-Indian climate, so figures illustrate the method rather than real observations. Swap in a Kaggle dataset to analyse real data.
"""
    (ROOT / "reports" / "insights_report.md").write_text(md)
    monthly.round(2).to_csv(ROOT / "outputs" / "monthly_summary.csv")
    df.to_csv(ROOT / "data" / "weather_clean.csv")


if __name__ == "__main__":
    raw = load()
    clean_df, log = clean(raw)
    m, y, s, c = analyse(clean_df)
    write_report(clean_df, log, m, y, s, c)
    print("Done. Cleaning log:", log)
