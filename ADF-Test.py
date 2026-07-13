"""
ADF Stationarity Test for P&G Quarterly Revenue
=================================================
Performs the Augmented Dickey-Fuller (ADF) unit-root test on P&G quarterly
revenue at multiple differencing levels, then fits a SARIMA model.

ADF test H0: series has a unit root (non-stationary).
p < 0.05 rejects H0 — confirms stationarity.

Tested differencing levels:
  1. Original data
  2. First-order differencing (d=1)
  3. First + Seasonal differencing (d=1, D=1, s=4) — removes quarterly seasonality

After testing, the script fits a SARIMA model using the recommended orders
and runs full model diagnostics: residual plots, Ljung-Box test, and PACF.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

# ==================== 1. Data Reading and Preprocessing ====================
# The CSV is assumed to have two columns:
#   - Fiscal Quarter End Date  (e.g., 2018-03-31)
#   - Quarterly Revenue (Millions USD)
path = os.path.join(os.path.expanduser("~"), "Desktop", "quarterly_revenue(1).csv")
df = pd.read_csv(path)

# Parse dates, sort chronologically, set as the DataFrame index
df["Fiscal Quarter End Date"] = pd.to_datetime(df["Fiscal Quarter End Date"])
df = df.sort_values("Fiscal Quarter End Date").set_index("Fiscal Quarter End Date")

# Explicitly set quarterly frequency using Business Quarter-End (BQ).
# BQ anchors to the last business day of the quarter, matching P&G reporting
# (typically the last trading day of Mar/Jun/Sep/Dec).
# If asfreq() creates NaN rows, your date range has gaps - recheck the CSV.
ts_data = df["Quarterly Revenue (Millions USD)"].asfreq("BQ")

# ==================== 2. ADF Test Function ====================
def adf_table(series_dict):
    """
    Run ADF tests on multiple series and return a formatted summary table.

    The ADF test regresses Delta-y_t on y_{t-1} plus lagged differences.
    H0: the series has a unit root (non-stationary).
    Rejecting H0 (p < 0.05) means the series is stationary.

    Args:
        series_dict: dict of {name: pd.Series} to test.
    Returns:
        DataFrame with Test Statistic, p-value, critical value at 5%,
        and a Yes/No stationarity verdict at the 5% level.
    """
    rows = []
    for name, series in series_dict.items():
        clean = series.dropna()
        if len(clean) < 3:
            rows.append({"Series": name,
                         "Test Statistic": "N/A",
                         "p-value": "N/A",
                         "Critical Value (5%)": "N/A",
                         "Stationary (p<0.05)": "Insufficient Data"})
            continue
        # autolag="AIC" lets the test pick the optimal number of lagged differences
        result = adfuller(clean, autolag="AIC")
        rows.append({"Series": name,
                     "Test Statistic": f"{result[0]:.4f}",
                     "p-value": f"{result[1]:.4f}",
                     "Critical Value (5%)": f"{result[4]["5%"]:.4f}",
                     "Stationary (p<0.05)": "Yes" if result[1] <= 0.05 else "No"})
    return pd.DataFrame(rows).set_index("Series")

# ==================== 3. Build the Three Differenced Series ====================
orig = ts_data
first_diff = ts_data.diff(1)           # d=1: first-order differencing removes linear drift
seasonal_diff = ts_data.diff(1).diff(4) # d=1, D=1: also removes quarterly seasonality

test_series = {
    "Original Data": orig,
    "First-order Differencing": first_diff,
    "First + Seasonal Diff (4)": seasonal_diff
}

# ==================== 4. Run ADF Tests ====================
print("
" + "=" * 70)
print("ADF Unit Root Test Results")
print("=" * 70)
adf_result = adf_table(test_series)
print(adf_result.to_string())
print("=" * 70 + "
")

# Typical result for P&G quarterly revenue:
#   Original:            p ~ 0.5 -> Non-stationary (random walk with drift)
#   First diff:          p ~ 0.1 -> Still borderline, depends on period
#   First + Seasonal:    p < 0.05 -> Stationary -> recommend d=1, D=1

# ==================== 5. Fit SARIMA Model ====================
# Based on the ADF results, d=1 and D=1 are needed for stationarity.
# order=(1,1,0): ARIMA(1,1,0) - one AR lag, first diff, no MA term.
# seasonal_order=(0,1,1,4): seasonal diff (D=1), seasonal MA Q=1, period s=4.
#
# enforce_stationarity=False: we already differenced, so this constraint is redundant.
# initialization="approximate_diffuse": better starting values when d+D > 0.
model = SARIMAX(ts_data,
                order=(1, 1, 0),
                seasonal_order=(0, 1, 1, 4),
                enforce_stationarity=False,
                enforce_invertibility=False,
                initialization="approximate_diffuse")
results = model.fit(disp=False)
print(results.summary())

# ==================== 6. Model Diagnostics ====================
print("
--- Model Diagnosis Report ---")

# Standardized residuals, histogram, Q-Q, residual ACF
# A well-specified model shows: white-noise residuals, normal Q-Q, no ACF spikes.
results.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.show()

# Ljung-Box test: H0 = residuals are independently distributed (no autocorrelation).
# p < 0.05 rejects H0 -> autocorrelation remains -> model needs more terms.
try:
    lb = acorr_ljungbox(results.resid, lags=[4], return_df=True, model_df=2)
    pval = lb["lb_pvalue"].values[0]
    print(f"Ljung-Box Test (lag=4) p-value: {pval:.4f}")
    if pval > 0.05:
        print("Residuals have no significant autocorrelation -- model fit is good.")
    else:
        print("Residuals still have autocorrelation -- may need to adjust model orders.")
except Exception as e:
    print(f"Ljung-Box Test failed: {e}")

# ==================== 7. PACF Plot (Post-2018 Subset) ====================
# PACF shows the direct correlation between y_t and y_{t-k} after removing
# intermediate lags. Useful for validating or choosing the AR order (p).
# Plotted only for post-2018 data to show the current structural regime.
subset = ts_data["2018":].dropna()
if len(subset) > 10:
    safe_lags = min(8, len(subset) // 3)
    if safe_lags > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_pacf(subset.diff().dropna(), lags=safe_lags, ax=ax)
        ax.set_title(f"PACF of differenced series (post-2018, lags={safe_lags})")
        plt.tight_layout()
        plt.show()
else:
    print("Insufficient data after 2018 -- skipping PACF plot.")
