# Weather Data Analysis (Cognevance - Level 1)

Analyses five years of daily weather records (temperature, humidity, rainfall, wind, pressure): cleans messy data, finds
seasonal trends, compares months and produces an insights report.

## Workflow
1. **Collect** - `src/generate_dataset.py` creates `data/weather_raw.csv` (synthetic North-India-style climate with injected
   errors). To use a real Kaggle dataset, drop it in `data/weather_raw.csv` using the same column names.
2. **Load & clean** (`src/analysis.py`) - parse dates, drop duplicates/invalid dates, flag impossible values
   (e.g. humidity > 100, sensor codes like 999 / -9999) as missing, interpolate gaps (rainfall gaps = 0).
3. **Analyse** - monthly/seasonal/yearly aggregates, correlations, rainy-day counts.
4. **Visualise** - 8 charts with Matplotlib/Seaborn in `outputs/charts/`.
5. **Report** - `reports/insights_report.md` is generated with the computed numbers.

## Run
```bash
pip install -r requirements.txt
python src/generate_dataset.py     # optional: regenerate the dataset
python src/analysis.py             # cleans data, makes charts, writes the report
```

## Structure
```
data/        weather_raw.csv, weather_clean.csv
src/         generate_dataset.py, analysis.py
outputs/     charts/*.png, monthly_summary.csv
reports/     insights_report.md
```

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn.

## Note
The dataset is synthetic, so findings illustrate the method, not real observations.
