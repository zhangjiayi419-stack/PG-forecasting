"""
Fetch Unilever Data from WRDS
===============================
Downloads raw financial data for Unilever (ticker: UL) from WRDS.
Must be run BEFORE Unilever_Financial_Analysis.py, which consumes these files.

This script is a WRDS SQL query wrapper — it connects to WRDS using the
credentials stored in the connect_wrds module (part of the wrds-fetcher skill),
executes three queries, and saves the results as CSV files.

Why separate data fetching from analysis?
  - WRDS is a restricted resource (subscription-based). Keeping fetch separate
    from analysis allows the analysis to be re-run without re-querying WRDS.
  - The CSV files become the "raw data" layer in the analysis pipeline.
  - If WRDS access is revoked or the subscription lapses, the analysis can
    still be reproduced using the cached CSVs.

Three datasets fetched:
  1. Compustat Annual (funda): Income statement + balance sheet for Unilever
     - Used for financial ratio calculation
  2. CRSP Monthly Returns: Unilever monthly stock returns (permno 71654)
     - Used for CAPM beta estimation
  3. Fama-French Monthly Factors: Market return, SMB, HML, risk-free rate
     - Used as the benchmark in CAPM (shared with P&G analysis)

WRDS identifiers for Unilever:
  - Compustat Ticker: UL (Unilever PLC, NYSE)
  - CRSP Permno: 71654 (Unilever PLC, NYSE)
  - gvkey: 110033 (Unilever PLC, global key)
  - Note: Unilever also has a London-listed share (ULVR) — this script uses NYSE

Note on Unilever's fiscal year:
  - Unilever reports on a calendar year basis (year-end = December 31)
  - Contrast with P&G which ends June 30 — this affects how we align the data

Dependencies:
  - wrds-fetcher skill: from connect_wrds import wrds_query
  - Valid WRDS credentials (set up via: wrds setup-wizard or web registration)
"""

import sys
# Add the wrds-fetcher skill to the path so we can import connect_wrds
sys.path.insert(0, r'C:\Users\leonl\.claude\plugins\marketplaces\anthropic-agent-skills\skills\wrds-fetcher\scripts')
from connect_wrds import wrds_query

DATA_DIR = 'D:/PG'

print("Fetching Unilever data from WRDS...")
print()

# =============================================================================
# 1. Annual Compustat — Income Statement + Balance Sheet
# =============================================================================
print("1. Fetching Compustat Annual Data...")
# Query explanation:
#   gvkey: company identifier (Unilever = 110033)
#   tic: ticker = 'UL' (NYSE-listed Unilever)
#   fyear: fiscal year >= 2010 and <= 2025
#   indfmt='INDL': Industrial format (vs. Financial Services, etc.)
#   datafmt='STD': Standard format (restated, not per-share)
#   popsrc='D': Population includes domestic companies only
#   consol='C': Consolidated statements (vs. segment-level)
# Fields:
#   prcc_f: Price at fiscal year close (used for market cap)
#   csho: Common shares outstanding
#   at: Total assets
#   lt: Total liabilities
#   revt: Total revenue
#   ebitda, ebit, ni: Earnings metrics
#   oibdp: Operating income before depreciation
#   dvt: Dividends (total)
funda = wrds_query("""
    SELECT gvkey, tic, fyear, prcc_f, csho, at, lt, revt, ebitda, ebit, ni, oibdp, dvt
    FROM comp.funda
    WHERE tic='UL' AND fyear >= 2010 AND fyear <= 2025
    AND indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
    ORDER BY fyear
""")

# Rename columns to match what our analysis script expects
funda.columns = ['Global_Company_Key', 'Ticker', 'Fiscal_Year',
                 'Stock_Price_Close_Fiscal_Year', 'Shares_Outstanding',
                 'Total_Assets', 'Total_Liabilities', 'Revenue',
                 'EBITDA', 'EBIT', 'Net_Income', 'Operating_Income', 'Dividends']

funda.to_csv(f'{DATA_DIR}/Unilever_Annual_Compustat_Data.csv', index=False)
print(f"   Saved: {DATA_DIR}/Unilever_Annual_Compustat_Data.csv ({len(funda)} rows)")
print()

# =============================================================================
# 2. Monthly Returns from CRSP
# =============================================================================
print("2. Fetching Monthly Returns from CRSP...")
# CRSP Monthly Stock File (msf): one row per permno per month
#   date: month-end date
#   ret: raw monthly return (including dividends, splits adjusted)
# permno 71654 = Unilever PLC (NYSE)
# We ask for all dates >= 2010-01-01 and let the merge with FF factors
# further filter the range.
returns = wrds_query("""
    SELECT date, ret FROM crsp.msf
    WHERE permno=71654 AND date >= '2010-01-01'
    ORDER BY date
""")

returns.columns = ['Date', 'Monthly_Return']
returns.to_csv(f'{DATA_DIR}/Unilever_Monthly_Returns_CRSP.csv', index=False)
print(f"   Saved: {DATA_DIR}/Unilever_Monthly_Returns_CRSP.csv ({len(returns)} rows)")
print()

# =============================================================================
# 3. Fama-French Factors (Shared Benchmark)
# =============================================================================
print("3. Fetching Fama-French Factors...")
# ff.factors_monthly: monthly market return and risk factors from Ken French's library
#   mktrf: Market excess return (Mkt - RF)
#   smb: Small Minus Big (size factor)
#   hml: High Minus Low (value factor)
#   rf: Risk-free rate (1-month T-Bill)
# These are the same factors used for P&G's CAPM, so we share the file.
ff = wrds_query("""
    SELECT date, mktrf, smb, hml, rf
    FROM ff.factors_monthly
    WHERE date >= '2010-01-01' AND date <= '2025-12-31'
    ORDER BY date
""")

ff.columns = ['Date', 'Market_Excess_Return', 'Small_Minus_Big',
              'High_Minus_Low', 'Risk_Free_Rate']

# Reuse the same filename as P&G — the analysis scripts look for this file
# for both companies (they're merged on date, so the same file works for both).
ff.to_csv(f'{DATA_DIR}/P&G_Fama_French_Factors.csv', index=False)
print(f"   Saved: {DATA_DIR}/P&G_Fama_French_Factors.csv ({len(ff)} rows)")
print()

# =============================================================================
# Summary
# =============================================================================
print("=" * 60)
print("All data fetched successfully!")
print("Next step: python D:/PG/scripts/Unilever_Financial_Analysis.py")
print("=" * 60)