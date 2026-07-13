"""
Unilever CAPM Beta Regression
===============================
Simple OLS regression to estimate Unilever's beta using monthly stock returns
and market excess returns from WRDS.

This is a streamlined version of the CAPM calculation in Unilever_Financial_Analysis.py.
The difference: this script uses sklearn's LinearRegression (vs. numpy manual calculation)
and produces a publication-quality scatter plot with the regression line.

Why a separate visualization script?
  - Sometimes you need a clean chart for a report or presentation.
  - This shows the actual data points (monthly observations) rather than just
    the summary statistics.
  - The scatter plot helps you visually assess:
      - Are there outliers pulling the beta estimate?
      - Is the relationship roughly linear, or are there non-linearities?
      - Are the residuals roughly homoscedastic (equal variance)?

What the chart shows:
  - X-axis: Market Excess Return (Rm - Rf) — how the market performed each month
  - Y-axis: Unilever Excess Return (Ri - Rf) — how Unilever performed relative to T-Bills
  - Red regression line: fitted CAPM: E(Ri - Rf) = α + β × (Rm - Rf)
  - β (beta): slope of the line — how sensitive Unilever is to market moves
    - β = 1: Unilever moves 1-for-1 with the market
    - β = 0.7: Unilever moves 0.7% for every 1% market move (less risky)
    - β = 1.3: Unilever moves 1.3% for every 1% market move (more risky)
  - R²: proportion of Unilever's variance explained by the market

Data: Unilever_Monthly_Returns_CRSP.csv and P&G_Fama_French_Factors.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =============================================================================
# 1. Load Data
# =============================================================================
unilever = pd.read_csv('D:/PG/Unilever_Monthly_Returns_CRSP.csv', parse_dates=['Date'])
ff = pd.read_csv('D:/PG/P&G_Fama_French_Factors.csv', parse_dates=['Date'])

# =============================================================================
# 2. Date Alignment
# =============================================================================
# Important data alignment issue:
#   - CRSP monthly returns are dated on the last trading day of the month
#     (e.g., January return is dated 2020-01-31)
#   - Fama-French factors are dated on the first of the month
#     (e.g., January factor is dated 2020-01-01)
#   - This creates a 1-month offset if we merge on Date directly.
#
# Fix: shift the Fama-French date to month-end (add one month and use MonthEnd(0))
#      so that 2020-01-01 becomes 2020-01-31, matching CRSP's convention.
ff['Date'] = ff['Date'] + pd.offsets.MonthEnd(0)

# Now merge on the aligned dates
merged = pd.merge(unilever, ff, on='Date', how='inner')
print(f"Merged rows: {len(merged)}")

# =============================================================================
# 3. Compute Excess Returns
# =============================================================================
# Unilever excess return: what Unilever returned minus the risk-free rate
# Market excess return: what the market returned minus the risk-free rate
# In CAPM theory, these two should be linearly related.
merged['Unilever_Excess'] = merged['Monthly_Return'] - merged['Risk_Free_Rate']
merged['Market_Excess']   = merged['Market_Excess_Return']

# Drop any rows with NaN (e.g., months where CRSP has no return data)
merged = merged.dropna()
print(f"After dropna: {len(merged)}")

# =============================================================================
# 4. OLS Regression via sklearn
# =============================================================================
# sklearn's LinearRegression is equivalent to OLS in this context (minimize SSE).
# We regress Unilever Excess Return (y) on Market Excess Return (x).
X = merged['Market_Excess'].values.reshape(-1, 1)
y = merged['Unilever_Excess'].values

reg = LinearRegression()
reg.fit(X, y)

# Beta = slope, Alpha = intercept
alpha = reg.intercept_
beta  = reg.coef_[0]

# Predicted values and R²
y_pred = reg.predict(X)
r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2))

print(f"Beta: {beta:.4f}")
print(f"Alpha (monthly): {alpha:.6f}")
print(f"R-squared: {r2:.4f}")

# =============================================================================
# 5. Publication-Quality Scatter Plot
# =============================================================================
plt.figure(figsize=(10, 6))

# Scatter plot of monthly observations
plt.scatter(X, y, alpha=0.6,
            label='Monthly Observations',
            color='steelblue', s=30)

# Regression line
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = reg.predict(x_line)
plt.plot(x_line, y_line, color='crimson', linewidth=2,
         label=f'Regression: y = {beta:.4f}x + {alpha:.4f}\n$\\beta$ = {beta:.4f}, $R^2$ = {r2:.4f}')

plt.xlabel('Market Excess Return', fontsize=12)
plt.ylabel('Unilever Excess Return', fontsize=12)
plt.title('Unilever CAPM Beta Regression\n(Market Excess Return vs Unilever Excess Return)', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save to the Graphs subfolder for organized output
os.makedirs('D:/PG/Graphs', exist_ok=True)
plt.savefig('D:/PG/Graphs/Unilever_Beta_Regression.png', dpi=150)
plt.show()
print("Saved: D:/PG/Graphs/Unilever_Beta_Regression.png")