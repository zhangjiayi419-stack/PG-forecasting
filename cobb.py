"""
Cobb-Douglas Production Function — P&G Regression
===================================================
Estimates the Cobb-Douglas production function for P&G using OLS regression
on annual data (2010–2025):

    ln(Revenue) = ln(A) + α·ln(Labor) + β·ln(Capital) + ε

Where:
  - Revenue: Total revenue in millions USD
  - Labor (L): Number of employees (in thousands)
  - Capital (K): Property, Plant & Equipment in millions USD
  - A: Intercept term = Total Factor Productivity (TFP)
  - α, β: Output elasticities of labor and capital (estimated via OLS)

Why log-linear form?
  - Cobb-Douglas is multiplicative: Y = A·L^α·K^β
  - Taking logs: ln(Y) = ln(A) + α·ln(L) + β·ln(K) + ε
  - This turns it into a LINEAR regression, which OLS can estimate directly.
  - The coefficients α and β are now directly interpretable as elasticities:
      α = ∂ln(Y)/∂ln(L) = %ΔY / %ΔL  (holding K constant)
      β = ∂ln(Y)/∂ln(K) = %ΔY / %ΔK  (holding L constant)

Expected results for P&G:
  - α ≈ 0.7–0.8: Labor is the dominant driver of revenue
  - β ≈ 0.8–0.9: Capital is also significant (P&G is capital-intensive)
  - α + β ≈ 1.5–1.7: Increasing returns to scale — P&G benefits from scale
  - TFP (A): should be a positive constant; explains the residual growth

Data source: P&G_Annual_Compustat_Data.csv on the desktop
  (fetched via WRDS Compustat in the data pipeline)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# =============================================================================
# 1. Load Data
# =============================================================================
# File path — update this if your data is in a different location.
file_path = r"D:\PG\Financial_data\PG_financial_data(raw).csv"

try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully!")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    print("Check the path and try again.")
    raise

# =============================================================================
# 2. Align Column Names
# =============================================================================
# The raw Compustat CSV may have messy column names.
# We map them to the standardized names used in the code.
# If your CSV has different column names, update the mapping below.
# Example of what the raw file might look like:
#   Revenue_Million_USD | PPE_Million_USD | Employees_Thousands
df.rename(columns={
    "Your_Revenue_Col": "Revenue_Million_USD",
    "Your_PPE_Col": "PPE_Million_USD",
    "Your_Employee_Col": "Employees_Thousands"
}, inplace=True, errors='ignore')  # errors='ignore' skips missing columns gracefully

print("\nData Preview:")
print(df.head())

# =============================================================================
# 3. Log-Transform All Variables
# =============================================================================
# Cobb-Douglas requires the log-linear form. We compute natural logs.
# Note: Adding a small constant (e.g., +1) before logging protects against
# log(0) if any of the values are zero or missing.
df['ln_Revenue'] = np.log(df['Revenue_Million_USD'])
df['ln_Labor']   = np.log(df['Employees_Thousands'])
df['ln_Capital'] = np.log(df['PPE_Million_USD'])

# =============================================================================
# 4. OLS Regression: ln(Revenue) ~ α + β₁·ln(Labor) + β₂·ln(Capital)
# =============================================================================
# X: Labor and Capital (both log-transformed)
# sm.add_constant(X): adds the intercept term (ln(A) in the Cobb-Douglas model)
X = df[['ln_Labor', 'ln_Capital']]
X = sm.add_constant(X)  # Adds a column of 1s → ln(A) estimate
y = df['ln_Revenue']

# Fit via Ordinary Least Squares (OLS)
model = sm.OLS(y, X).fit()

# =============================================================================
# 5. Print Full Regression Output
# =============================================================================
print("\n" + "=" * 70)
print("COBB-DOUGLAS PRODUCTION FUNCTION — OLS RESULTS")
print("=" * 70)
print(model.summary())

# =============================================================================
# 6. Extract Key Parameters
# =============================================================================
# α = coefficient on ln(Labor) = labor elasticity
# β = coefficient on ln(Capital) = capital elasticity
alpha = model.params['ln_Labor']
beta  = model.params['ln_Capital']

# A = exp(intercept) = Total Factor Productivity
# The intercept is ln(A), so A = exp(ln(A))
A = np.exp(model.params['const'])

# Returns to Scale: α + β
#   = 1: constant (doubling inputs → doubling output)
#   > 1: increasing (scale economies — relevant for P&G's brand portfolio)
#   < 1: decreasing (scale diseconomies)
returns_scale = alpha + beta

print("\n" + "=" * 50)
print("KEY INTERPRETATION")
print("=" * 50)
print(f"Labor Elasticity (α):        {alpha:.4f}")
print(f"Capital Elasticity (β):       {beta:.4f}")
print(f"Returns to Scale (α + β):     {returns_scale:.4f}")
print(f"Total Factor Productivity (A): {A:.4f}")
print("=" * 50)
print()
print("Interpretation:")
print(f"  A 1% increase in labor raises revenue by {alpha:.2f}% (holding capital fixed)")
print(f"  A 1% increase in capital raises revenue by {beta:.2f}% (holding labor fixed)")
print(f"  α + β = {returns_scale:.2f} → {'increasing' if returns_scale > 1 else 'constant' if returns_scale == 1 else 'decreasing'} returns to scale")
print()

# =============================================================================
# 7. Model Fit Plot
# =============================================================================
# Plot actual ln(Revenue) vs. fitted ln(Revenue).
# If the model fits well, points should cluster around the 45° line.
plt.figure(figsize=(10, 5))
plt.scatter(y, model.fittedvalues, color='blue', alpha=0.7,
            label='Yearly observations')

# 45-degree reference line — perfect prediction would put all points here
plt.plot([y.min(), y.max()], [y.min(), y.max()],
         'r--', lw=2, label='Perfect fit (45° line)')

plt.xlabel('Actual Log Revenue')
plt.ylabel('Predicted Log Revenue')
plt.title('Cobb-Douglas Model Fit')
plt.grid(True)
plt.legend()
plt.show()