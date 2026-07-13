"""
Time Series Analysis — P&G Quarterly Revenue
===============================================
A comprehensive time series analysis pipeline for P&G quarterly revenue, covering:
  1. Data loading and frequency alignment
  2. Stationarity testing (ADF — but see ADF-Test.py for the full version)
  3. SARIMA model estimation
  4. Residual diagnostics

This is a streamlined version of the full analysis. For detailed ADF testing
across multiple differencing levels, see ADF-Test.py.

Note on P&G's fiscal year:
  - P&G's fiscal year ends June 30. So "FY2024" runs from July 1, 2023 to June 30, 2024.
  - When using calendar-quarter dates in the CSV, be aware that the alignment
    between calendar quarters and fiscal quarters may shift around June/July.

Input file: quarterly_revenue(1).csv on the desktop
Dependencies: pandas, numpy, matplotlib, statsmodels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# =============================================================================
# 1. Load Data from Desktop
# =============================================================================
# The CSV should have two columns:
#   - Fiscal Quarter End Date  (e.g., "2018-03-31")
#   - Quarterly Revenue (Millions USD)  (e.g., 16543.2)
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
file_path = os.path.join(desktop_path, "quarterly_revenue(1).csv")

# Read, parse dates, sort, set index — the standard time-series pipeline
df = pd.read_csv(file_path)
df['Fiscal Quarter End Date'] = pd.to_datetime(df['Fiscal Quarter End Date'])
df = df.sort_values('Fiscal Quarter End Date')
df.set_index('Fiscal Quarter End Date', inplace=True)

# The target variable: quarterly revenue in millions USD
# .asfreq('Q') explicitly declares quarterly frequency.
# If the date range has gaps (missing quarters), this will create NaN rows —
# check for those if you get weird results in the ADF test.
ts_data = df['Quarterly Revenue (Millions USD)'].asfreq('Q')

# =============================================================================
# 2. Quick ADF Stationarity Check
# =============================================================================
# The Augmented Dickey-Fuller test checks for a unit root.
# H0: series has a unit root (non-stationary)
# We want to reject H0 (p < 0.05) to confirm stationarity.
adf_result = adfuller(ts_data.dropna())
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value:       {adf_result[1]:.4f}")
print("Stationary (p<0.05):", "Yes" if adf_result[1] < 0.05 else "No — differencing needed")

# =============================================================================
# 3. Fit SARIMA Model
# =============================================================================
# order=(1,1,1): ARIMA with p=1 (one lag of autoregression),
#                           d=1 (first difference for stationarity),
#                           q=1 (one moving average lag)
# seasonal_order=(1,1,1,4): Seasonal AR=1, Seasonal Diff=1, Seasonal MA=1, Period=4
# enforce_stationarity=False: We're differencing, so stationarity is already imposed
model = SARIMAX(ts_data,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 4),
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit(disp=False)

# Print model summary: coefficients, standard errors, AIC, diagnostic stats
print("\n" + "=" * 50)
print("SARIMA Model Summary")
print("=" * 50)
print(results.summary())

# =============================================================================
# 4. Forecast — Next 8 Quarters
# =============================================================================
# h=8: forecast 8 quarters ahead (~2 years)
# For longer horizons, use the 10-year script.
forecast = results.get_forecast(steps=8)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# =============================================================================
# 5. Visualization
# =============================================================================
plt.figure(figsize=(14, 6))

# Historical data — blue line
plt.plot(ts_data, label='Historical Revenue', color='blue', linewidth=2)

# Forecast — red dashed line
plt.plot(forecast_mean.index, forecast_mean.values,
         label='Forecast', color='red', linestyle='--', linewidth=2)

# 95% confidence interval — shaded band
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1],
                 color='pink', alpha=0.3, label='95% CI')

plt.title('P&G Quarterly Revenue — SARIMA Forecast (8 Quarters)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Revenue (Millions USD)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# =============================================================================
# 6. ACF and PACF Plots — Model Diagnostic
# =============================================================================
# ACF (Autocorrelation Function): shows correlation between y_t and y_{t-k}
# PACF (Partial ACF): correlation with lag k after removing intermediate lags
# If the model is well-specified, ACF/PACF of residuals should have no
# significant spikes beyond the blue shaded 95% bands.
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# ACF — use differenced data since that's what the model sees
plot_acf(ts_data.diff().dropna(), lags=12, ax=axes[0], alpha=0.05)
axes[0].set_title('ACF of Differenced Revenue')

# PACF
plot_pacf(ts_data.diff().dropna(), lags=12, ax=axes[1], alpha=0.05)
axes[1].set_title('PACF of Differenced Revenue')

plt.tight_layout()
plt.show()