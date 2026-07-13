"""
P&G Financial Ratio Analysis & CAPM Model
==========================================
Computes 15+ financial ratios from P&G's annual Compustat data and runs a
Capital Asset Pricing Model (CAPM) regression to estimate P&G's beta.

This is the main analytical script that generates two key output files:
  - P&G_Financial_Ratios.csv: Annual ratio time series (profitability, leverage, coverage)
  - P&G_CAPM_Results.csv: CAPM regression results (beta, alpha, R²)

What does CAPM tell us?
  - Beta (β) measures systematic risk — how much P&G's returns co-move with the market.
    β > 1 → more volatile than the market (defensive but not a "risk-free" haven)
    β < 1 → less volatile (a "widow and orphan" stock)
  - Alpha (α) is the excess return not explained by market exposure.
    A significant positive alpha means P&G outperformed what CAPM would predict,
    given its risk level. A significant negative alpha means underperformance.
  - Expected Return = Rf + β × (Rm - Rf)
    This is the "fair" return an investor should expect given the risk they bear.

Data Sources (all from WRDS):
  1. Compustat Annual (funda): Balance sheet + income statement items
  2. CRSP Monthly Returns: P&G stock returns (permno 18163)
  3. Fama-French Factors: Market excess return + risk-free rate

================================================================================
DATA CAVEATS — READ THIS BEFORE INTERPRETING RESULTS
================================================================================
1. P&G Fiscal Year: P&G's fiscal year ends June 30.
   FY2010 = July 1, 2009 to June 30, 2010.
   Compustat uses the fiscal year end date, so FY2010 data corresponds to
   calendar year 2009/2010 depending on the field. Match on fyear carefully.

2. CRSP Publishing Lag: CRSP data has a 2–3 month delay.
   As of early 2026, the latest available CRSP data ends 2024-12.
   2025 monthly returns must be re-queried from WRDS when available.
   Check: if len(returns) < 180 for 2010–2024, you're missing data.

3. Risk-Free Rate (2010–2015):
   The early-period "zeros" in the Fama-French file are NOT data errors.
   They reflect the near-zero Federal Reserve policy after the 2008 crisis.
   T-Bill rates were ~0% from 2009–2015. The actual rates were:
     2010–2012: ~0.00%
     2013–2015: ~0.01–0.05%
     2016–2019: ~0.5–2.0%
     2020–2025: ~3.0–5.0%

================================================================================
CAPM THEORY
================================================================================
Model:   Ri - Rf = α + β × (Rm - Rf) + ε

Variables in the regression:
  - Excess_Return  = Monthly_Stock_Return - Risk_Free_Rate
  - Market_Excess_Return = Monthly_Market_Return - Risk_Free_Rate

In our implementation:
  - We directly compute beta as: Cov(Ri, Rm) / Var(Rm)
  - Then derive alpha from the regression identity
  - This gives the same result as OLS but is more explicit

Key output interpretation:
  - β ≈ 0.4–0.6: P&G is a low-beta consumer staples stock, less sensitive to
                  market swings than the average S&P 500 constituent
  - α not significantly different from 0: Returns are fully explained by market
    exposure — P&G doesn't have unexplained alpha or beta on average
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = 'D:/PG'    # Where the WRDS CSV files live
OUTPUT_DIR = 'D:/PG'  # Where to save the output files

# =============================================================================
# LOAD WRDS DATA
# =============================================================================
print("Loading data from WRDS...")

# Annual Compustat: balance sheet + income statement
# Columns: Global_Company_Key, Ticker, Fiscal_Year, Stock_Price_Close_Fiscal_Year,
#           Shares_Outstanding, Total_Assets, Total_Liabilities, Revenue,
#           EBITDA, EBIT, Net_Income, Operating_Income, Dividends
funda = pd.read_csv(f'{DATA_DIR}/P&G_Annual_Compustat_Data.csv')

# Monthly stock returns from CRSP
returns = pd.read_csv(f'{DATA_DIR}/P&G_Monthly_Returns_CRSP.csv')

# Fama-French monthly factors (shared benchmark for both P&G and Unilever)
ff = pd.read_csv(f'{DATA_DIR}/P&G_Fama_French_Factors.csv')

# =============================================================================
# DATA QUALITY CHECKS
# =============================================================================
print("\nData Availability Check:")
print(f"  P&G Monthly Returns: {returns['Date'].min()} to {returns['Date'].max()}")
print(f"                       ({len(returns)} observations)")
print(f"  FF Factors:          {ff['Date'].min()} to {ff['Date'].max()}")
print(f"  Annual Compustat:   FY{min(funda['Fiscal_Year'])} to FY{max(funda['Fiscal_Year'])}")

print("\n  NOTE: CRSP data ends 2024-12-31 due to WRDS publishing lag.")
print("        Re-query CRSP for 2025 data when available.")

print("\n  Risk-Free Rate Source: 1-month US Treasury Bill rate")
print("  Early zeros (2010-2015) are real T-Bill rates during near-zero rate policy.")

# =============================================================================
# CALCULATE FINANCIAL RATIOS
# =============================================================================
print("\nCalculating financial ratios...")

df = funda.copy()

# ----- Per Share Metrics -----
# Market Cap = Stock Price × Shares Outstanding (both in the same units)
df['Market_Capitalization_USD'] = df['Stock_Price_Close_Fiscal_Year'] * df['Shares_Outstanding']

# Book Value Per Share = (Total Assets - Total Liabilities) / Shares
df['Book_Value_Per_Share_USD'] = (df['Total_Assets'] - df['Total_Liabilities']) / df['Shares_Outstanding']

# Dividends Per Share
df['Dividends_Per_Share_USD'] = df['Dividends'] / df['Shares_Outstanding']

# =============================================================================
# PROFITABILITY RATIOS — How well P&G turns revenue into profit
# =============================================================================
# Net Profit Margin = Net Income / Revenue
# Measures bottom-line efficiency after all expenses.
df['Net_Profit_Margin'] = (df['Net_Income'] / df['Revenue'] * 100).round(2)

# EBIT Margin = EBIT / Revenue
# Operating profitability, before interest and taxes.
df['EBIT_Margin'] = (df['EBIT'] / df['Revenue'] * 100).round(2)

# EBITDA Margin = EBITDA / Revenue
# Operating cash profitability — strips out depreciation/amortization.
df['EBITDA_Margin'] = (df['EBITDA'] / df['Revenue'] * 100).round(2)

# Return On Assets (ROA) = EBIT / Total Assets
# How efficiently assets generate operating profit.
df['Return_On_Assets_ROA'] = (df['EBIT'] / df['Total_Assets'] * 100).round(2)

# Return On Equity (ROE) = Net Income / Shareholders' Equity
# How efficiently equity generates profit for shareholders.
# Shareholders' Equity = Total Assets - Total Liabilities
df['Return_On_Equity_ROE'] = (df['Net_Income'] / (df['Total_Assets'] - df['Total_Liabilities']) * 100).round(2)

# Cash Flow Margin = EBITDA / Revenue
# Similar to EBIT margin but more cash-focused (less affected by D&A policy).
df['Cash_Flow_Margin'] = (df['EBITDA'] / df['Revenue'] * 100).round(2)

# =============================================================================
# ACTIVITY RATIOS — How efficiently assets are used
# =============================================================================
# Asset Turnover = Revenue / Total Assets
# How many dollars of revenue each dollar of assets produces.
df['Asset_Turnover'] = (df['Revenue'] / df['Total_Assets']).round(4)

# Net Income To Total Assets
# Another measure of asset efficiency, this time on a net income basis.
df['Net_Income_To_Total_Assets'] = (df['Net_Income'] / df['Total_Assets']).round(4)

# =============================================================================
# COVERAGE RATIOS — Ability to service debt
# =============================================================================
# Debt Ratio = Total Liabilities / Total Assets
# What fraction of assets are financed by debt. P&G typically ~50–55%.
df['Debt_Ratio'] = (df['Total_Liabilities'] / df['Total_Assets']).round(4)

# Cash Debt Coverage = EBITDA / Total Liabilities
# How many times EBITDA covers total debt — higher is safer.
df['Cash_Debt_Coverage'] = (df['EBITDA'] / df['Total_Liabilities']).round(4)

# Market To Debt = Market Cap / Total Liabilities
# How the market values P&G relative to its debt obligations.
# > 1 means market value exceeds book debt; < 1 signals potential distress.
df['Market_To_Debt'] = (df['Market_Capitalization_USD'] / df['Total_Liabilities']).round(4)

# =============================================================================
# LIQUIDITY / VALUATION RATIOS
# =============================================================================
# Market To Assets = Market Cap / Total Assets
# The market's valuation of P&G relative to its asset base.
df['Market_To_Assets'] = (df['Market_Capitalization_USD'] / df['Total_Assets']).round(4)

# Liabilities To Assets = Debt Ratio (redundant but explicit)
df['Liabilities_To_Assets'] = (df['Total_Liabilities'] / df['Total_Assets']).round(4)

# Dividend Yield = DPS / Stock Price
# Annual cash return to shareholders as a % of share price.
df['Dividend_Yield'] = (df['Dividends_Per_Share_USD'] / df['Stock_Price_Close_Fiscal_Year'] * 100).round(2)

# Earnings Yield = Net Income / Market Cap
# Inverse of the P/E ratio — earnings as a % of market value.
# Useful for comparing to bond yields (E/P vs. bond yield).
df['Earnings_Yield'] = (df['Net_Income'] / df['Market_Capitalization_USD'] * 100).round(2)

# =============================================================================
# SAVE FINANCIAL RATIOS
# =============================================================================
ratio_columns = [
    'Fiscal_Year',
    # Profitability
    'Net_Profit_Margin', 'EBIT_Margin', 'EBITDA_Margin',
    'Return_On_Assets_ROA', 'Return_On_Equity_ROE', 'Cash_Flow_Margin',
    # Activity
    'Asset_Turnover', 'Net_Income_To_Total_Assets',
    # Coverage
    'Debt_Ratio', 'Cash_Debt_Coverage', 'Market_To_Debt',
    # Liquidity / Valuation
    'Market_To_Assets', 'Liabilities_To_Assets', 'Dividend_Yield', 'Earnings_Yield',
    # Raw numbers for reference
    'Market_Capitalization_USD', 'Book_Value_Per_Share_USD', 'Dividends_Per_Share_USD',
    'Revenue', 'EBIT', 'EBITDA', 'Net_Income',
    'Total_Assets', 'Total_Liabilities', 'Stock_Price_Close_Fiscal_Year', 'Shares_Outstanding'
]

ratios_df = df[ratio_columns].copy()
ratios_df.to_csv(f'{OUTPUT_DIR}/P&G_Financial_Ratios.csv', index=False)
print(f"Saved: {OUTPUT_DIR}/P&G_Financial_Ratios.csv")

# =============================================================================
# CAPM REGRESSION
# =============================================================================
print("\nRunning CAPM regression...")

# Convert Date columns to datetime (may be string in the CSV)
returns['Date'] = pd.to_datetime(returns['Date'])
ff['Date'] = pd.to_datetime(ff['Date'])

# Create year-month period for merging (CRSP reports on month-end dates)
returns['Year_Month'] = returns['Date'].dt.to_period('M')
ff['Year_Month'] = ff['Date'].dt.to_period('M')

# Merge on Year_Month — inner join keeps only matching months
# Fama-French factors provide: Market_Excess_Return (Rm - Rf) and Risk_Free_Rate (Rf)
capm_data = returns.merge(ff[['Year_Month', 'Market_Excess_Return', 'Risk_Free_Rate']],
                           on='Year_Month', how='inner').dropna()

# Excess return = stock return minus risk-free rate
# In CAPM regression: (Ri - Rf) = α + β × (Rm - Rf)
capm_data['Excess_Return'] = capm_data['Monthly_Return'] - capm_data['Risk_Free_Rate']

# Extract arrays for calculation
x = capm_data['Market_Excess_Return'].values   # Independent variable: market excess return
y = capm_data['Excess_Return'].values           # Dependent variable: stock excess return

# =============================================================================
# OLS via Closed-Form (same result as statsmodels OLS)
# =============================================================================
x_mean = np.mean(x)
y_mean = np.mean(y)

# Beta = Cov(Ri, Rm) / Var(Rm) — by definition in CAPM theory
# Covariance measures how P&G moves with the market; variance measures
# total market risk. Their ratio is the proportion of P&G's risk that's
# "systematic" (market-wide) rather than idiosyncratic (P&G-specific).
beta = np.cov(x, y)[0, 1] / np.var(x)

# Alpha = Mean(Excess_Return) - Beta × Mean(Market_Excess_Return)
# The intercept — average return not explained by market exposure.
alpha = y_mean - beta * x_mean

# Fitted values and R²
y_pred = alpha + beta * x
ss_res = np.sum((y - y_pred) ** 2)   # Residual sum of squares
ss_tot = np.sum((y - y_mean) ** 2)    # Total sum of squares
r_squared = 1 - (ss_res / ss_tot)

# Standard Error of Beta
n = len(x)
se_beta = np.sqrt(ss_res / (n - 2)) / np.sqrt(np.sum((x - x_mean) ** 2))

# t-statistic for Beta
t_statistic_beta = beta / se_beta

# Annualize the rates (monthly → annual)
# Compound annualization: (1 + r_monthly)^12 - 1 ≈ 12 × r_monthly for small rates
risk_free_annual = (1 + capm_data['Risk_Free_Rate'].mean()) ** 12 - 1
market_excess_annual = capm_data['Market_Excess_Return'].mean() * 12

# Expected Annual Return via CAPM: E(R) = Rf + β × (Rm - Rf)
expected_return_annual = risk_free_annual + beta * market_excess_annual

# =============================================================================
# SAVE CAPM RESULTS
# =============================================================================
capm_results = pd.DataFrame({
    'Metric': [
        'Beta',
        'Alpha_Monthly',
        'Alpha_Annual_Percent',
        'R_Squared',
        'Number_of_Monthly_Observations',
        'Data_Start_Date',
        'Data_End_Date',
        'Risk_Free_Rate_Monthly',
        'Risk_Free_Rate_Annual_Percent',
        'Average_Market_Excess_Return_Monthly',
        'Average_Market_Excess_Return_Annual_Percent',
        'Expected_Annual_Return_Percent',
        't_statistic_for_Beta'
    ],
    'Value': [
        f'{beta:.4f}',
        f'{alpha:.6f}',
        f'{alpha * 12 * 100:.2f}',    # Annualize alpha: multiply monthly by 12
        f'{r_squared:.4f}',
        str(n),
        str(capm_data['Date'].min().date()),
        str(capm_data['Date'].max().date()),
        f'{capm_data["Risk_Free_Rate"].mean():.6f}',
        f'{risk_free_annual * 100:.2f}',
        f'{capm_data["Market_Excess_Return"].mean():.6f}',
        f'{market_excess_annual * 100:.2f}',
        f'{expected_return_annual * 100:.2f}',
        f'{t_statistic_beta:.4f}'
    ]
})

capm_results.to_csv(f'{OUTPUT_DIR}/P&G_CAPM_Results.csv', index=False)
print(f"Saved: {OUTPUT_DIR}/P&G_CAPM_Results.csv")

# =============================================================================
# PRINT SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("CAPM MODEL RESULTS")
print("=" * 70)
print(f"  Beta:                              {beta:.4f}")
print(f"  Alpha (annual):                    {alpha * 12 * 100:.2f}%")
print(f"  R-squared:                         {r_squared:.4f}")
print(f"  t-statistic (Beta):                {t_statistic_beta:.4f}")
print(f"  Risk-free rate (annual):            {risk_free_annual * 100:.2f}%")
print(f"  Expected Annual Return (CAPM):      {expected_return_annual * 100:.2f}%")
print(f"  Number of observations:             {n}")
print(f"  Date range:                         {capm_data['Date'].min().date()} to {capm_data['Date'].max().date()}")
print()
print("CAPM Formula: E(R) = Rf + Beta × (Rm - Rf)")
print(f"              = {risk_free_annual*100:.2f}% + {beta:.2f} × ({market_excess_annual*100:.2f}%)")
print("=" * 70)
print("\nAnalysis Complete!")
print("\nNOTE: To update with 2025 monthly returns, re-run the CRSP query in WRDS")
print("      and replace P&G_Monthly_Returns_CRSP.csv before re-running this script.")