"""
SARIMA Revenue Forecast — 3-Year Horizon
==========================================
A shorter-horizon SARIMA forecast (12 quarters / 3 years) for P&G quarterly
revenue, using the same methodology as the 10-year version but focused on
near-term precision.

Why a separate 3-year script?
  - Short-horizon forecasts are more reliable — confidence intervals are tighter
    and structural breaks are less likely to materialize.
  - This script uses quarterly data (not annual) for more data points and
    better seasonal pattern detection.

Key difference from sarima_10yr.py:
  - Uses quarterly data from quarterly_revenue(1).csv, not annual data
  - Includes a seasonal component (seasonal_order) since quarterly data has
    recurring intra-year patterns
  - Shorter horizon → tighter confidence intervals → more actionable

Dependencies: pandas, numpy, matplotlib, statsmodels
Input file: quarterly_revenue(1).csv on the desktop
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =============================================================================
# 1. Load Quarterly Revenue Data
# =============================================================================
# Read from desktop — the CSV should have columns:
#   Fiscal Quarter End Date, Quarterly Revenue (Millions USD)
# The file is sorted by date to ensure proper time-series ordering.
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
file_path = os.path.join(desktop_path, "quarterly_revenue(1).csv")

df = pd.read_csv(file_path)
df['Fiscal Quarter End Date'] = pd.to_datetime(df['Fiscal Quarter End Date'])
df = df.sort_values('Fiscal Quarter End Date')
df.set_index('Fiscal Quarter End Date', inplace=True)

# Explicitly set quarterly frequency.
# Why is this important? Without a recognized frequency, statsmodels can't
# infer seasonal periodicity (s=4 for quarterly). asfreq('Q') tells it
# the series has 4 observations per year. If you see weird forecasts,
# check that asfreq() didn't create NaN rows for missing quarters.
ts_data = df['Quarterly Revenue (Millions USD)'].asfreq('Q')

# =============================================================================
# 2. SARIMA Model Specification
# =============================================================================
# order=(1,1,1): ARIMA(1,1,1) — one AR term, one difference, one MA term
# seasonal_order=(1,1,1,4): Seasonal AR=1, Seasonal diff=1, Seasonal MA=1, period=4
#
# Why (1,1,1)?
#   - d=1: first-difference makes quarterly revenue stationary (most revenue series
#          are I(1) — integrated of order 1)
#   - p=1, q=1: parsimonious — minimal parameters for a quarterly series with
#                only ~60-80 observations
#
# Why seasonal D=1?
#   - Quarterly data has intra-year patterns: Q2 tends to be weakest, Q4 strongest
#     (holiday season). Seasonal differencing removes this repeating pattern so
#     the ARIMA core can focus on the trend and irregular components.
#
# enforce_stationarity=False, enforce_invertibility=False:
#   We already differenced (d=1, D=1), so the resulting series is stationary by
#   construction. Enforcing these constraints would just cause estimation problems.
# Trend='c': include a constant (drift) term in the ARIMA — helps capture
#            the long-run growth trend without an explicit intercept trick.
model = SARIMAX(ts_data,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 4),
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit(disp=False)  # disp=False: no iterative optimization printout

# =============================================================================
# 3. Forecast — Next 12 Quarters
# =============================================================================
# h = 12: forecast 12 steps ahead = 3 years of quarterly data.
# With ~3 years of data, we're pushing the envelope a bit — confidence
# intervals will widen meaningfully. Real decision-making should use the
# central estimate but track actuals vs. forecast closely.
forecast_steps = 12
forecast_res = results.get_forecast(steps=forecast_steps)

# Build a proper datetime index for the forecast periods.
# We start from the last observed date and add 12 more quarters.
# [1:] skips the first element which equals the last observed date.
forecast_index = pd.date_range(ts_data.index[-1], periods=forecast_steps + 1, freq='Q')[1:]

forecast_values = forecast_res.predicted_mean
forecast_ci = forecast_res.conf_int()  # 95% confidence interval by default

# =============================================================================
# 4. Visualization
# =============================================================================
plt.figure(figsize=(15, 7))

# Historical series — solid blue line
plt.plot(ts_data, label='Historical Revenue', color='blue', linewidth=2)

# Forecast — red dashed line
plt.plot(forecast_index, forecast_values, label='3-Year Forecast', color='red', linestyle='--')

# Confidence interval — shaded area
# alpha=0.2 makes the shading semi-transparent so gridlines show through.
# The interval gets wider as we forecast further out — this is correct behavior.
plt.fill_between(forecast_index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1],
                 color='pink', alpha=0.2, label='95% Confidence Interval')

plt.title('P&G 3-Year Quarterly Revenue Forecast', fontsize=14)
plt.ylabel('Revenue (Millions USD)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.show()