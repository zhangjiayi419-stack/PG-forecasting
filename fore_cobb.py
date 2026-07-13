"""
Cobb-Douglas Production Function Forecast (2026–2035)
======================================================
Uses the Cobb-Douglas production function to project P&G's revenue for the next
10 years, calibrated to 2025 actuals and driven by labor/capital growth rates
computed from historical data.

What is the Cobb-Douglas production function?
  - A classic economic model of how inputs (labor L and capital K) generate output (Y):
      Y = A · L^α · K^β
    where:
      A = Total Factor Productivity (TFP) — the "residuals" not explained by L or K
      α = labor elasticity: % change in Y for a 1% change in L (holding K constant)
      β = capital elasticity: % change in Y for a 1% change in K (holding L constant)
      α + β = returns to scale:
          α+β = 1 → constant returns to scale (doubling inputs doubles output)
          α+β > 1 → increasing returns (scale economies)
          α+β < 1 → decreasing returns (scale diseconomies)

Why use this for revenue forecasting?
  - Cobb-Douglas is a fundamental, theory-driven model — unlike black-box ML,
    its parameters have economic meaning and are interpretable.
  - Once we have α, β from historical regression (cobb.py) and the growth
    rates of L and K (cagr.py), we can project future revenue growth as:
      g_Y = α · g_L + β · g_K
    where g_i is the growth rate of each input.

The calibration approach in this script:
  - Use 2025 actual revenue as the anchor point (instead of relying on the
    model's fitted value, which may drift from actuals).
  - Project growth factors forward using CAGR of L and K, then apply those
    growth factors to the 2025 actual revenue.
  - This hybrid approach respects both the model's structural logic and
    the most recent data point.

Inputs:
  - 2025 actual revenue (hard-coded from P&G annual report)
  - Labor and Capital growth rates (computed in CAGR.py)
  - Cobb-Douglas elasticities α, β (hard-coded from the regression output in cobb.py)

Output: Projected revenue for 2026–2035, visualized as a line chart.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. Anchor Point — 2025 Actual Revenue
# =============================================================================
# P&G FY2025 revenue: $84.284 billion (from P&G Annual Report, FY ends June 2025).
# Convert to billions for cleaner chart labels.
revenue_2025_actual = 84284000000 / 1e9  # → 84.28 Billion USD

# =============================================================================
# 2. Recover Labor and Capital Levels from Log Values
# =============================================================================
# These log values come from the Cobb-Douglas regression output in cobb.py.
# The regression was run in log-space: ln(Y) = ln(A) + α·ln(L) + β·ln(K)
# Here we back-transform the estimated ln(L) and ln(K) to get the levels for 2010 and 2025.
L_2010, L_2025 = np.exp(22.8765), np.exp(22.8827)
K_2010, K_2025 = np.exp(23.6804), np.exp(23.9350)

# =============================================================================
# 3. Compute Annual Growth Rates (CAGR) for Labor and Capital
# =============================================================================
# CAGR = (End/Start)^(1/15) - 1  over the 2010–2025 period
# These growth rates are assumed to persist into the forecast period.
# A more sophisticated model would project g_L and g_K over time, but
# assuming constant rates is a reasonable baseline for a 10-year horizon.
years_diff = 2025 - 2010
cagr_L = (L_2025 / L_2010) ** (1 / years_diff) - 1
cagr_K = (K_2025 / K_2020) ** (1 / years_diff) - 1

# =============================================================================
# 4. Forecast Revenue (2026–2035) Using Cobb-Douglas Growth Identity
# =============================================================================
# Cobb-Douglas implies that output growth is a weighted average of input growths:
#   %ΔY = α · %ΔL + β · %ΔK
#
# In growth-factor terms (compounding):
#   Y_t = Y_2025 · (1+g_L)^t^α · (1+g_K)^t^β
#
# Where the growth factors are raised to the power of their elasticity.
# This accounts for diminishing/amplifying returns from each input.

forecast_years = np.arange(2026, 2036)  # 10 years: 2026 through 2035
projected_revenue = []

# Elasticity coefficients — directly from the Cobb-Douglas regression (cobb.py).
# α = 0.7226: a 1% increase in labor raises revenue by ~0.72%
# β = 0.8720: a 1% increase in capital raises revenue by ~0.87%
beta_L = 0.7226
beta_K = 0.8720

for i in range(1, 11):
    # i = 1, 2, ..., 10: number of years from 2025
    # Growth factor for each input raised to its elasticity power.
    # The exponents α, β reflect diminishing returns — if β=0.5, doubling K
    # only raises output by √2 ≈ 1.41x, not 2x.
    growth_factor_L = (1 + cagr_L) ** i
    growth_factor_K = (1 + cagr_K) ** i

    # Combine the two growth factors using Cobb-Douglas multiplication.
    # Revenue growth ratio = L_ratio^α × K_ratio^β
    revenue_growth_ratio = (growth_factor_L ** beta_L) * (growth_factor_K ** beta_K)

    # Apply the growth ratio to the 2025 base
    projected_revenue.append(revenue_2025_actual * revenue_growth_ratio)

# =============================================================================
# 5. Visualization
# =============================================================================
plt.figure(figsize=(12, 6))

# Main forecast line
plt.plot(forecast_years, projected_revenue,
         marker='s', color='#004c97', linewidth=2,
         label='Cobb-Douglas Calibrated Forecast')

# Annotate each data point with the dollar amount
# ha='center', va='bottom' places the text just above the marker.
for x, y in zip(forecast_years, projected_revenue):
    plt.text(x, y + 0.5, f'${y:.2f}B',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Set y-axis range to zoom in on the relevant range (~$84B–$105B).
# Without this, matplotlib auto-scales to $0–$110B and the growth looks flat.
plt.ylim(80, 105)

plt.title('P&G Revenue Forecast (2026–2035) — Calibrated to 2025 Actuals', fontsize=14)
plt.ylabel('Revenue (Billions USD)')
plt.grid(True, alpha=0.3)
plt.show()