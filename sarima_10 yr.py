"""
SARIMA Revenue Forecast — 10-Year Horizon
==========================================
Fits a SARIMA (Seasonal ARIMA) model to P&G's quarterly revenue and forecasts
the next 40 quarters (10 years).

Why SARIMA for long-term forecasting?
  - Revenue has both a trend (growing over time) and strong seasonal fluctuations
    (Q2 is typically the weakest, Q4 the strongest due to holiday sales).
  - A standard ARIMA captures the trend but ignores seasonality.
  - SARIMA adds seasonal terms (P, D, Q)[s] that capture the quarterly pattern.

Key assumptions and limitations:
  - This is an UNCONDITIONAL forecast — it doesn't account for macro events,
    new product launches, or structural breaks (e.g., the Gillette sale in 2015).
  - Confidence intervals widen dramatically beyond ~12 quarters.
    Treat anything beyond 3 years as directional, not precise.
  - For a 10-year projection, consider blending with a fundamental model
    (e.g., Cobb-Douglas in fore_cobb.py) for more realistic long-run estimates.

Dependencies: pandas, numpy, matplotlib, statsmodels
Input file: quarterly_revenue(1).csv on the desktop
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Data Set — Hard-coded P&G Annual Revenue (in Millions USD)
# =============================================================================
# Source: P&G Annual Reports / Compustat
# Period: FY2010 to FY2025 (P&G fiscal year ends June 30)
# Note: FY2016 revenue is significantly lower due to currency headwinds
#       and the divested beauty brands. FY2020-Q3 also drops sharply (COVID).
years = list(range(2010, 2026))  # 16 years
revenues = [78938, 82559, 83680, 84167, 83062, 76279, 65299, 65058,
            66832, 67684, 70950, 76118, 80187, 82006, 84039, 84284]

df = pd.DataFrame({'year': years, 'revenue': revenues})
print(df.head())

# Create a time index (integer: 0, 1, 2, ...) for regression.
# Using (year - year.min()) instead of year directly avoids multicollinearity
# with the intercept in OLS regression — makes coefficients more numerically stable.
df['time'] = df['year'] - df['year'].min()


# =============================================================================
# 2. Simple Linear Regression — Baseline Model
# =============================================================================
# Revenue = α + β·time + ε
# This is the most naive model — assumes a constant linear trend.
# Low R² (typically ~0.5 for P&G) signals that linear trend is insufficient.
X = sm.add_constant(df['time'])
y = df['revenue']
model_ols = sm.OLS(y, X).fit()
print("\n" + "="*50)
print("Simple Linear Regression Results")
print("="*50)
print(model_ols.summary())

# Durbin-Watson statistic: tests for first-order autocorrelation in residuals.
# DW ≈ 2 → no autocorrelation; DW < 2 → positive autocorrelation (common in time series)
# DW ≈ 1.2 or so → residuals are positively autocorrelated → OLS standard errors
# are underestimated, t-stats are inflated → false precision.
dw = durbin_watson(model_ols.resid)
print(f"\nDurbin-Watson: {dw:.4f}  (closer to 2 is better; < 1.5 suggests autocorrelation)")

# Plot the ACF of residuals — visual check complementary to DW test
# Significant spikes at lags > 0 confirm residual autocorrelation.
plt.figure(figsize=(10, 4))
plot_acf(model_ols.resid, lags=10, alpha=0.05)
plt.title("Original Model Residuals — Autocorrelation Function (ACF)")
plt.tight_layout()
plt.show()

# =============================================================================
# 3. Log-Linear Model — Exponential Trend
# =============================================================================
# ln(Revenue) = α + β·time + ε
# If β ≈ 0.02, revenue grows ~2% per year (exponential growth model).
# Back-transforming: Revenue = exp(α) · exp(β·time)
# This fits better than linear when growth is multiplicative, not additive.
df['ln_rev'] = np.log(df['revenue'])
model_ln = sm.OLS(df['ln_rev'], X).fit()
print("\n" + "="*50)
print("Log-Linear Model Results (ln(Revenue) ~ Time)")
print("="*50)
print(model_ln.summary())
# Interpretation: if β = 0.02, annual growth rate ≈ 2%

# =============================================================================
# 4. Auto-ARIMA — Data-Driven Order Selection
# =============================================================================
# pmdarima.auto_arima searches over (p,d,q) × (P,D,Q)[s] combinations and picks
# the model with the lowest AIC (Akaike Information Criterion).
# seasonal=True would include seasonal terms, but for annual data (n=16),
# there's insufficient data to estimate seasonal parameters reliably.
# stepwise=True: greedy search, faster but may miss the global optimum.
auto_model = auto_arima(df['revenue'], seasonal=False, stepwise=True,
                        trace=True, suppress_warnings=True)
print("\nBest ARIMA order:", auto_model.order)

# Fit the best model and print diagnostics
model_arima = ARIMA(df['revenue'], order=auto_model.order).fit()
print(model_arima.summary())

# =============================================================================
# 5. Breakpoint Regression — Structural Change Around 2018
# =============================================================================
# P&G underwent significant restructuring around 2017-2019:
#   - Divested beauty brands (July 2016, ~$12B in revenue removed)
#   - Announced productivity plan
#   - Accelerated organic growth post-restructuring
# A breakpoint regression allows the intercept and slope to shift after 2018.
# Model: Revenue = α + β₁·time + γ·D + δ·(time·D) + ε
#   where D = 1 if year ≥ 2018, else 0
#   γ = change in intercept post-2018
#   δ = change in trend post-2018
break_year = 2018
df['break_dummy'] = (df['year'] >= break_year).astype(int)
df['time_break'] = df['time'] * df['break_dummy']  # interaction term
X_break = sm.add_constant(df[['time', 'break_dummy', 'time_break']])
model_break = sm.OLS(df['revenue'], X_break).fit()
print("\n" + "="*50)
print(f"Breakpoint Regression (Breakpoint = {break_year})")
print("="*50)
print(model_break.summary())
# Key question: is the post-2018 slope (β₁ + δ) steeper than pre-2018 (β₁)?
# If δ > 0 and significant, revenue growth accelerated after restructuring.

# =============================================================================
# 6. Forecast — Next 10 Years (2026–2035)
# =============================================================================
future_years = list(range(2026, 2036))
n_steps = len(future_years)  # = 10
# Map future years to the time index (0, 1, 2, ...) used in the models
future_time = np.array([y - df['year'].min() for y in future_years]).reshape(-1, 1)

# 6.1 Simple linear regression
X_future = sm.add_constant(future_time)
pred_ols = model_ols.predict(X_future)

# 6.2 Log-linear model — back-transform from log scale
# exp() undoes the ln() transformation. Note: this introduces a bias correction
# (Jorgensen-Bierens method) in rigorous applications, but exp() is standard here.
pred_ln = np.exp(model_ln.predict(X_future))

# 6.3 ARIMA model — built-in forecast method
arima_forecast = model_arima.forecast(steps=n_steps)
# ARIMA.forecast() returns a Series or numpy array depending on statsmodels version
if hasattr(arima_forecast, 'values'):
    arima_forecast = arima_forecast.values
arima_forecast = arima_forecast[:n_steps]

# 6.4 Breakpoint regression — build future X with break dummy = 1
# All future years are post-2018, so break_dummy=1 and time_break=time for all rows
future_df_break = pd.DataFrame({
    'time': future_time.flatten(),
    'break_dummy': [1] * n_steps,          # in the post-break regime
    'time_break': future_time.flatten()    # interaction term
})
X_future_break = sm.add_constant(future_df_break, has_constant='add')
X_future_break = X_future_break[X_break.columns]  # ensure same column order as training data
pred_break = model_break.predict(X_future_break)

# =============================================================================
# 7. Model Comparison Table
# =============================================================================
# For ARIMA, rsquared_adj is not directly comparable to OLS R² since ARIMA
# is estimated by maximum likelihood, not OLS. We use AIC for model comparison.
comparison = pd.DataFrame({
    'Model': ['simple linear', 'log-linear', 'ARIMA', 'breakpoint regression'],
    'R²(adjusted)': [
        model_ols.rsquared_adj,
        model_ln.rsquared_adj,
        None,       # ARIMA R² not directly computed by ARIMA.summary()
        model_break.rsquared_adj
    ],
    'AIC': [
        model_ols.aic,
        model_ln.aic,
        model_arima.aic,
        model_break.aic
    ],
    '10_year_forcast': [
        forecast_df['simple linear regression'].mean(),
        forecast_df['log-linear prediction'].mean(),
        forecast_df['ARIMA prediction'].mean(),
        forecast_df['breakpoint regression prediction'].mean()
    ]
})
print("\n" + "="*50)
print("Model Comparison")
print("="*50)
print(comparison)
# Lower AIC is better. The best model can be selected by AIC even when
# sample sizes differ (ARIMA is ML-based, OLS is SSE-based).

# =============================================================================
# 8. Visualization — All Forecasts on One Chart
# =============================================================================
plt.figure(figsize=(12, 6))

# Historical data — solid line with circle markers
plt.plot(df['year'], df['revenue'], 'o-', label='Historical Data',
         linewidth=2, markersize=8, color='navy')

# Four forecast series — dashed lines with different markers for legibility
plt.plot(future_years, pred_ols, 's--', label='Simple Linear Regression', alpha=0.7)
plt.plot(future_years, pred_ln, '^--', label='Log-Linear Model', alpha=0.7)
plt.plot(future_years, arima_forecast, 'd--', label='ARIMA Forecast', alpha=0.7)
plt.plot(future_years, pred_break, 'x--',
         label=f'Breakpoint Regression (break={break_year})', alpha=0.7)

plt.xlabel('Year')
plt.ylabel('Revenue (Millions USD)')
plt.title('P&G Revenue Forecast Comparison (2026–2035)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()