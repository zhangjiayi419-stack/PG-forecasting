"""
Unilever Financial Ratio Analysis & CAPM Model
=================================================
Computes financial ratios and CAPM beta for Unilever (ticker: UL, NYSE), using
data fetched by fetch_unilever.py from WRDS.

This is the Unilever counterpart to PG_Financial_Analysis.py. The methodology
is identical, but note these key differences between the two companies:

  P&G (PG) vs. Unilever (UL):
  - Fiscal year: P&G ends June 30; Unilever ends December 31 (calendar year).
    When analyzing both companies together, align by calendar year or fiscal Q4.
  - Beta: Unilever tends to have a higher beta (~0.7–0.9) than P&G (~0.4–0.6),
    reflecting its greater exposure to emerging markets and FX volatility.
  - Currency: Both report in USD (UL converts from GBP/EUR), so FX fluctuations
    affect the revenue series even if the underlying business is stable.

Data sources (from fetch_unilever.py):
  - Unilever_Annual_Compustat_Data.csv  → financial ratios
  - Unilever_Monthly_Returns_CRSP.csv   → monthly stock returns
  - P&G_Fama_French_Factors.csv        → market benchmark (shared with P&G)

Output files:
  - Unilever_Financial_Ratios.csv
  - Unilever_CAPM_Results.csv
"""

import pandas as pd
import numpy as np

DATA_DIR = 'D:/PG'
OUTPUT_DIR = 'D:/PG'

print("Loading data from WRDS...")

funda  = pd.read_csv(f'{DATA_DIR}/Unilever_Annual_Compustat_Data.csv')
returns = pd.read_csv(f'{DATA_DIR}/Unilever_Monthly_Returns_CRSP.csv')
ff     = pd.read_csv(f'{DATA_DIR}/P&G_Fama_French_Factors.csv')  # Same FF factors for both companies

print("\nData Availability:")
print(f"  Monthly Returns: {returns['date'].min()} to {returns['date'].max()} ({len(returns)} obs)")
print(f"  FF Factors:     {ff['Date'].min()} to {ff['Date'].max()}")
print(f"  Annual Data:    FY{funda['fyear'].min()} to FY{funda['fyear'].max()}")

print("\nCalculating ratios...")

df = funda.copy()

# Per-share metrics
df['Market_Capitalization_USD'] = df['prcc_f'] * df['csho'] / 1000    # Market cap in USD millions
df['Book_Value_Per_Share_USD']  = (df['at'] - df['lt']) / df['csho']  # Book value per share
df['Dividends_Per_Share_USD']   = df['dvt'] / df['csho']              # DPS

# PROFITABILITY RATIOS
df['Net_Profit_Margin']      = (df['ni']   / df['revt'] * 100).round(2)
df['EBIT_Margin']            = (df['ebit'] / df['revt'] * 100).round(2)
df['EBITDA_Margin']          = (df['ebitda'] / df['revt'] * 100).round(2)
df['Return_On_Assets_ROA']   = (df['ebit'] / df['at'] * 100).round(2)
df['Return_On_Equity_ROE']   = (df['ni']   / (df['at'] - df['lt']) * 100).round(2)
df['Cash_Flow_Margin']       = (df['ebitda'] / df['revt'] * 100).round(2)

# ACTIVITY RATIOS
df['Asset_Turnover']             = (df['revt'] / df['at']).round(4)
df['Net_Income_To_Total_Assets'] = (df['ni']   / df['at']).round(4)

# COVERAGE RATIOS
df['Debt_Ratio']       = (df['lt'] / df['at']).round(4)
df['Cash_Debt_Coverage'] = (df['ebitda'] / df['lt']).round(4)
df['Market_To_Debt']   = (df['Market_Capitalization_USD'] * 1000 / df['lt']).round(4)

# LIQUIDITY / VALUATION
df['Market_To_Assets']   = (df['Market_Capitalization_USD'] * 1000 / df['at']).round(4)
df['Liabilities_To_Assets'] = (df['lt'] / df['at']).round(4)
df['Dividend_Yield']    = (df['Dividends_Per_Share_USD'] / df['prcc_f'] * 100).round(2)
df['Earnings_Yield']    = (df['ni'] / (df['Market_Capitalization_USD'] * 1000) * 100).round(2)

# =============================================================================
# SAVE FINANCIAL RATIOS
# =============================================================================
ratio_cols = ['fyear',
              'Net_Profit_Margin', 'EBIT_Margin', 'EBITDA_Margin',
              'Return_On_Assets_ROA', 'Return_On_Equity_ROE', 'Cash_Flow_Margin',
              'Asset_Turnover', 'Net_Income_To_Total_Assets',
              'Debt_Ratio', 'Cash_Debt_Coverage', 'Market_To_Debt', 'Market_To_Assets',
              'Liabilities_To_Assets', 'Dividend_Yield', 'Earnings_Yield',
              'Market_Capitalization_USD', 'Book_Value_Per_Share_USD', 'Dividends_Per_Share_USD',
              'revt', 'ebit', 'ebitda', 'ni', 'at', 'lt', 'prcc_f', 'csho']

ratios_df = df[ratio_cols].copy()
ratios_df.columns = ['Fiscal_Year'] + ratio_cols[1:]
ratios_df.to_csv(f'{OUTPUT_DIR}/Unilever_Financial_Ratios.csv', index=False)
print(f"Saved: {OUTPUT_DIR}/Unilever_Financial_Ratios.csv")

# =============================================================================
# CAPM REGRESSION
# =============================================================================
print("\nRunning CAPM regression...")
returns['date'] = pd.to_datetime(returns['date'])
ff['Date']      = pd.to_datetime(ff['Date'])
returns['Year_Month'] = returns['date'].dt.to_period('M')
ff['Year_Month']      = ff['Date'].dt.to_period('M')

# Inner merge on year-month — only keep months where both return and factor data exist
capm = returns.merge(ff[['Year_Month', 'Market_Excess_Return', 'Risk_Free_Rate']],
                     on='Year_Month', how='inner').dropna()
capm['Excess_Return'] = capm['ret'] - capm['Risk_Free_Rate']

x = capm['Market_Excess_Return'].values
y = capm['Excess_Return'].values

# Beta = Cov(Ri, Rm) / Var(Rm)
beta = np.cov(x, y)[0, 1] / np.var(x)

# Alpha
alpha = np.mean(y) - beta * np.mean(x)

# Fitted values and R²
y_pred = alpha + beta * x
r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

# Standard error of beta
n = len(x)
se_b = np.sqrt(np.sum((y - y_pred) ** 2) / (n - 2)) / np.sqrt(np.sum((x - np.mean(x)) ** 2))
t_b = beta / se_b

# Annualized rates
rf_ann  = (1 + capm['Risk_Free_Rate'].mean()) ** 12 - 1
mkt_ann = capm['Market_Excess_Return'].mean() * 12
exp_ret = rf_ann + beta * mkt_ann

# Save results
capm_results = pd.DataFrame({
    'Metric': ['Beta', 'Alpha_Monthly', 'Alpha_Annual_Percent', 'R_Squared',
               'Number_of_Monthly_Observations', 'Data_Start_Date', 'Data_End_Date',
               'Risk_Free_Rate_Monthly', 'Risk_Free_Rate_Annual_Percent',
               'Average_Market_Excess_Return_Monthly', 'Average_Market_Excess_Return_Annual_Percent',
               'Expected_Annual_Return_Percent', 't_statistic_for_Beta'],
    'Value': [f'{beta:.4f}', f'{alpha:.6f}', f'{alpha * 12 * 100:.2f}', f'{r2:.4f}',
              str(n), str(capm['date'].min().date()), str(capm['date'].max().date()),
              f'{capm["Risk_Free_Rate"].mean():.6f}', f'{rf_ann * 100:.2f}',
              f'{capm["Market_Excess_Return"].mean():.6f}', f'{mkt_ann * 100:.2f}',
              f'{exp_ret * 100:.2f}', f'{t_b:.4f}']
})
capm_results.to_csv(f'{OUTPUT_DIR}/Unilever_CAPM_Results.csv', index=False)
print(f"Saved: {OUTPUT_DIR}/Unilever_CAPM_Results.csv")

# =============================================================================
# PRINT SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("UNILEVER CAPM RESULTS")
print("=" * 60)
print(f"  Beta:                     {beta:.4f}")
print(f"  Alpha (annual):            {alpha * 12 * 100:.2f}%")
print(f"  R-squared:                {r2:.4f}")
print(f"  Expected Return (CAPM):    {exp_ret * 100:.2f}%")
print(f"  Observations:              {n}")
print(f"  Date Range:                {capm['date'].min().date()} to {capm['date'].max().date()}")
print()
print("Formula: E(R) = Rf + Beta × (Rm - Rf)")
print(f"       = {rf_ann * 100:.2f}% + {beta:.2f} × ({mkt_ann * 100:.2f}%)")
print("=" * 60)
print("\nDone!")