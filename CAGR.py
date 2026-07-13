"""
CAGR Calculator — Labor and Capital Growth Rates
==================================================
Calculates the Compound Annual Growth Rate (CAGR) for P&G's labor input (employees)
and capital stock (Property, Plant & Equipment) over the period 2010–2025.

What is CAGR?
  - The constant annual growth rate that would take the starting value to the
    ending value over n years.
  - Formula: CAGR = (End / Start)^(1/n) - 1
  - More meaningful than a simple average growth rate because it accounts for
    compounding, not just linear accumulation.

Why this matters for P&G analysis:
  - Labor and capital growth rates are inputs to the Cobb-Douglas production function.
  - Knowing these rates allows us to project future inputs and generate more
    realistic long-run revenue forecasts (see fore_cobb.py).
  - The inputs here are in LOG-SPACE (already log-transformed from the regression
    in cobb.py). We exponentiate them back to levels before computing CAGR.

Input: Log-transformed values from EViews regression output (cobb.py):
  - ln_L_2010, ln_L_2025: log of labor (employees × average compensation)
  - ln_K_2010, ln_K_2025: log of capital (PPE in USD millions)

Output: Annualized growth rates for labor and capital inputs.
"""

import numpy as np

# =============================================================================
# 1. Input: Log Values from EViews / Cobb-Douglas Regression
# =============================================================================
# These come from the log-space coefficients from the Cobb-Douglas regression
# in cobb.py. They're expressed as natural logs (ln), NOT base-10.
# We exponentiate them back to levels before calculating CAGR.
#
# L = Labor input (approximated by total employee compensation or headcount proxy)
# K = Capital input (PPE — Property, Plant & Equipment)

# Log of labor, 2010 and 2025
ln_L_2010 = 22.8765345
ln_L_2025 = 22.8827412

# Log of capital (PPE), 2010 and 2025
ln_K_2010 = 23.6804652
ln_K_2025 = 23.9349962

# =============================================================================
# 2. Back-Transform from Log Space and Calculate CAGR
# =============================================================================
# Exponentiate the log values to get actual levels.
L_start, L_end = np.exp(ln_L_2010), np.exp(ln_L_2025)
K_start, K_end = np.exp(ln_K_2010), np.exp(ln_K_2025)

# Number of years between the two data points
n_years = 2025 - 2010  # = 15 years

# CAGR formula: (End/Start)^(1/n) - 1
# This gives the average annual growth rate assuming compounding.
cagr_L = (L_end / L_start)**(1 / n_years) - 1
cagr_K = (K_end / K_start)**(1 / n_years) - 1

# =============================================================================
# 3. Print Results
# =============================================================================
# The output tells us whether P&G's inputs have been growing or shrinking,
# and at what rate. These rates feed directly into the Cobb-Douglas forecast.
print("-" * 30)
print(f"Strategic Growth Analysis (2010–2025):")
print("-" * 30)
print(f"Labor (Salary) CAGR:  {cagr_L:.4%}")
print(f"Capital (PPE) CAGR:   {cagr_K:.4%}")
print("-" * 30)

# Interpretation:
# - A positive labor CAGR means P&G is spending more on compensation (either more
#   employees, higher wages, or both) — note this is affected by inflation.
# - A positive capital CAGR means P&G is investing in more fixed assets
#   (factories, equipment, etc.).
# - These rates, combined with Cobb-Douglas elasticities (α, β from cobb.py),
#   give us the projected revenue growth rate: g_Y = α·g_L + β·g_K