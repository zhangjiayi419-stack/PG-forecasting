"""
P&G Financial Ratio Trend Analysis — Single Company
====================================================
Computes a comprehensive set of financial ratios from P&G's annual Compustat
data and produces a professional 4-panel chart showing key profitability and
leverage trends over time.

Why this script vs. PG_Financial_Analysis.py?
  - PG_Financial_Analysis.py computes all ratios and saves them to CSV.
  - This script focuses specifically on visual analysis, plotting 4 key ratios
    (profit margin, ROA, ROE, debt-to-assets) over time for a clean presentation.
  - The charts are saved to the Desktop in high resolution (300 DPI).

Scope: Uses data from FY2018 onwards (post the major restructuring and
       Gillette divestiture) to focus on the current business regime.

What each ratio tells you:
  - Profit Margin: Net income as % of revenue. P&G's target is ~18–20%.
    COVID (FY2020) caused a dip; margin has since recovered.
  - ROA: How efficiently P&G uses all assets to generate profit.
    P&G's ROA is typically 8–12% — reasonable for a consumer staples giant.
  - ROE: Return to shareholders. P&G has historically maintained ~25–30% ROE,
    driven by financial leverage (debt) amplifying equity returns.
  - Debt to Assets: ~50–55% for P&G — a deliberate capital structure choice.
    High leverage is sustainable for P&G given its stable cash flows.

Input file: P&G_Annual_Compustat_Data.csv on the desktop
Output:    PG_Full_Ratio_Analysis.png on the desktop (300 DPI)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ===================== 1. Load Data =====================
# P&G Annual Compustat data from WRDS (same source as PG_Financial_Analysis.py)
desktop_path = str(Path.home() / "Desktop")
annual_data = pd.read_csv(f"{desktop_path}/P&G_Annual_Compustat_Data.csv")

# Clean column names: strip whitespace and convert to lowercase
# WRDS sometimes adds trailing spaces to column names — this catches those edge cases.
annual_data.columns = annual_data.columns.str.strip().str.lower()

# Define column names based on the actual Compustat column names in the raw CSV.
# These map to what the raw file actually contains after loading.
year_col  = "fiscal_year"
rev       = "revenue"
ni        = "net_income"
ta        = "total_assets"
tl        = "total_liabilities"
shares    = "shares_outstanding"
dividends = "dividends"

# Optional columns — if present in the file, we'll compute additional ratios.
# If absent, we simply skip those ratios (no error, just fewer outputs).
cash_col           = "cash_and_cash_equivalents" if "cash_and_cash_equivalents" in annual_data.columns else None
current_assets_col = None   # Current assets column (if present)
current_liab_col   = None   # Current liabilities column (if present)
cogs_col           = None   # Cost of goods sold (if present)
inv_col            = None   # Inventory (if present)
ar_col             = None   # Accounts receivable (if present)
interest_exp_col   = None   # Interest expense (if present)
op_cash_col        = None   # Operating cash flow (if present)

# ===================== 2. Filter to Post-2018 Data =====================
# P&G underwent major restructuring 2015–2017 (Gillette divestiture + productivity plan).
# Using data from FY2018 onwards gives us a consistent "current regime."
df_pg = annual_data[annual_data[year_col] >= 2018].copy().sort_values(year_col).reset_index(drop=True)

# Compute average balances (required for ROA/ROE by accounting best practice).
# Using average of current and prior year instead of end-of-period balances
# smooths out one-time fluctuations and is more comparable to income statement items
# (which are flows over the period, not point-in-time balances).
df_pg['avg_total_assets'] = df_pg[ta].rolling(2).mean()
df_pg['total_equity'] = df_pg[ta] - df_pg[tl]   # Book value of equity
df_pg['avg_equity'] = df_pg['total_equity'].rolling(2).mean()

# ===================== 3. Compute Financial Ratios =====================
# PROFITABILITY RATIOS
# Profit Margin = Net Income / Revenue
df_pg['profit_margin'] = (df_pg[ni] / df_pg[rev]) * 100

# Return on Assets (ROA) = Net Income / Average Total Assets
df_pg['roa'] = (df_pg[ni] / df_pg['avg_total_assets']) * 100

# Return on Equity (ROE) = Net Income / Average Equity
df_pg['roe'] = (df_pg[ni] / df_pg['avg_equity']) * 100

# EPS = Net Income / Shares Outstanding
df_pg['eps'] = df_pg[ni] / df_pg[shares]

# Payout Ratio = Dividends / Net Income (what % of earnings are returned as dividends)
df_pg['payout_ratio'] = (df_pg[dividends] / df_pg[ni]) * 100

# LEVERAGE / COVERAGE RATIOS
# Debt to Assets = Total Liabilities / Total Assets
df_pg['debt_to_assets'] = (df_pg[tl] / df_pg[ta]) * 100

# Debt to Equity = Total Liabilities / Total Equity
df_pg['debt_to_equity'] = (df_pg[tl] / df_pg['total_equity']) * 100

# Book Value Per Share = Total Equity / Shares Outstanding
df_pg['book_value_per_share'] = df_pg['total_equity'] / df_pg[shares]

# OPTIONAL RATIOS (only if source columns are available)
# Current Ratio = Current Assets / Current Liabilities
if current_assets_col and current_liab_col:
    df_pg['current_ratio'] = df_pg[current_assets_col] / df_pg[current_liab_col]
    quick_assets = df_pg[cash_col] + df_pg.get('short_term_investment', 0) + df_pg.get(ar_col, 0)
    df_pg['quick_ratio'] = quick_assets / df_pg[current_liab_col]

# Asset Turnover = Revenue / Average Total Assets
df_pg['asset_turnover'] = df_pg[rev] / df_pg['avg_total_assets']

# Interest Coverage = EBIT / Interest Expense
if interest_exp_col:
    ebit = df_pg[ni] + df_pg[interest_exp_col] + df_pg.get('income_tax', 0)
    df_pg['times_interest_earned'] = ebit / df_pg[interest_exp_col]

# Cash Debt Coverage = Operating Cash Flow / Average Total Liabilities
if op_cash_col:
    df_pg['avg_total_liab'] = df_pg[tl].rolling(2).mean()
    df_pg['cash_debt_coverage'] = df_pg[op_cash_col] / df_pg['avg_total_liab']

# ===================== 4. Professional 4-Panel Chart =====================
# Use seaborn for cleaner default aesthetics
sns.set_theme(style="whitegrid", font="Arial")
plt.rcParams['axes.unicode_minus'] = False  # Show minus signs correctly, not box chars

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
yr = df_pg[year_col]

# Panel 1: Profit Margin
ax = axes[0, 0]
ax.plot(yr, df_pg['profit_margin'], marker='o', c='#2ca02c', lw=2.5, label='P&G')
ax.set_title("Profit Margin on Sales", fontsize=14)
ax.set_ylabel("Percentage (%)", fontsize=11)
ax.legend()
# Note: Y-axis starts at 0 for percentage charts — makes visual comparison fair
ax.set_ylim(bottom=0)

# Panel 2: Return on Assets
ax = axes[0, 1]
ax.plot(yr, df_pg['roa'], marker='o', c='#9467bd', lw=2.5, label='P&G')
ax.set_title("Return on Assets (ROA)", fontsize=14)
ax.set_ylabel("Percentage (%)", fontsize=11)
ax.legend()
ax.set_ylim(bottom=0)

# Panel 3: Return on Equity
ax = axes[1, 0]
ax.plot(yr, df_pg['roe'], marker='o', c='#d62728', lw=2.5, label='P&G')
ax.set_title("Return on Equity (ROE)", fontsize=14)
ax.set_ylabel("Percentage (%)", fontsize=11)
ax.legend()
ax.set_ylim(bottom=0)

# Panel 4: Debt to Assets
ax = axes[1, 1]
ax.plot(yr, df_pg['debt_to_assets'], marker='o', c='#17becf', lw=2.5, label='P&G')
ax.set_title("Debt to Assets Ratio", fontsize=14)
ax.set_ylabel("Percentage (%)", fontsize=11)
ax.legend()
ax.set_ylim(bottom=0)

# Main title and layout
fig.suptitle("P&G Full Financial Ratio Trend Analysis (2018–Present)", fontsize=18, y=1.02)
plt.tight_layout()

# Save in high resolution for reports/presentations
plt.savefig(f"{desktop_path}/PG_Full_Ratio_Analysis.png", dpi=300, bbox_inches='tight')

print("Saved: PG_Full_Ratio_Analysis.png on desktop (300 DPI)")