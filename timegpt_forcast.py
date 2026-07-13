"""
TimeGPT Quarterly Revenue Forecast
====================================
Uses Nixtla's TimeGPT API to forecast P&G's next 12 quarters of revenue.

Why TimeGPT instead of SARIMA?
  - TimeGPT is a transformer-based foundation model for time series, trained on
    billions of data points. It often outperforms classical models on short-to-medium
    horizons (12-36 quarters) without requiring manual parameter tuning.
  - The API handles seasonality detection automatically.

Requirements:
  - nixtla package: pip install nixtla
  - A valid Nixtla API key (free tier available at app.nixtla.io)
  - quarterly_revenue(1).csv on the desktop (columns: Fiscal Quarter End Date,
    Quarterly Revenue (Millions USD))
"""

import pandas as pd
import matplotlib.pyplot as plt
from nixtla import NixtlaClient

# ---------------------------------------------------------------------------
# 1. Initialize the Nixtla client with your API key
# ---------------------------------------------------------------------------
# The client handles authentication and request batching. The key is tied to
# your Nixtla account — sign up at app.nixtla.io if you don't have one.
nixtla_client = NixtlaClient(api_key="nixak-fee1663bd0d4b783b666dc17bd26235bad52df1a10ea90528712dc13a22c3e028cf580d2b4ab17dd")
nixtla_client.validate_api_key()  # Quick sanity check — dies here if key is bad

# ---------------------------------------------------------------------------
# 2. Read the quarterly revenue data from the desktop
# ---------------------------------------------------------------------------
# P&G reports quarterly earnings, so we expect 4 rows per year.
# If you get a shape error here, double-check the CSV column names match exactly.
df = pd.read_csv("quarterly_revenue(1).csv")

# ---------------------------------------------------------------------------
# 3. Nixtla convention: rename columns to 'ds' (timestamp) and 'y' (target)
# ---------------------------------------------------------------------------
# Nixtla's API is opinionated — it expects these exact column names.
# Fiscal Quarter End Date → ds (timestamp)
# Quarterly Revenue (Millions USD) → y (what we want to forecast)
df['Fiscal Quarter End Date'] = pd.to_datetime(df['Fiscal Quarter End Date'])
df = df.rename(columns={
    'Fiscal Quarter End Date': 'ds',
    'Quarterly Revenue (Millions USD)': 'y'
})

# Keep only what Nixtla needs — extra columns just slow down the API call
df = df[['ds', 'y']]
print("Processed quarterly data preview:")
print(df.head())

# ---------------------------------------------------------------------------
# 4. Forecast the next 12 quarters (3 years)
# ---------------------------------------------------------------------------
# h=12: forecast 12 steps ahead
# freq='Q': tell the model we're working with quarterly data so it knows the
#           seasonal pattern (P&G's fiscal year has 4 quarters)
# time_col / target_col: explicitly name the columns in our DataFrame
# model='timegpt-1': use the standard TimeGPT model. For very long horizons
#                    (>40 quarters) consider 'timegpt-1-hierarchical'
# add_history=True: include historical data in the response so we can plot it
forecast_df = nixtla_client.forecast(
    df=df,
    h=12,                       # Horizon: next 12 quarters
    freq='Q',                   # Quarterly frequency
    time_col='ds',
    target_col='y',
    model='timegpt-1',
    add_history=True
)

# ---------------------------------------------------------------------------
# 5. Print the forecast results
# ---------------------------------------------------------------------------
print("\nForecast results (last 15 rows):")
print(forecast_df.tail(12))

# ---------------------------------------------------------------------------
# 6. Plot historical vs. forecast
# ---------------------------------------------------------------------------
# There's a known gotcha here: forecast_df may include a row for the last
# historical date (when add_history=True). We filter it out so we don't draw
# a double line that confuses the chart.
plt.figure(figsize=(14, 7))

# Historical data — solid blue line with markers
plt.plot(df['ds'], df['y'],
         label='Historical Quarterly Revenue',
         color='blue', marker='o', linewidth=2, markersize=6)

# Forecast data — red dashed line
last_historical_date = df['ds'].max()
future_forecast = forecast_df[forecast_df['ds'] > last_historical_date]

plt.plot(future_forecast['ds'], future_forecast['TimeGPT'],
         label='Forecasted Quarterly Revenue',
         color='red', marker='o', linestyle='--', linewidth=2, markersize=6)

plt.title('P&G Quarterly Revenue Forecast (Next 12 Quarters)', fontsize=16)
plt.ylabel('Revenue (Millions USD)', fontsize=12)
plt.xlabel('Quarter End Date', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()