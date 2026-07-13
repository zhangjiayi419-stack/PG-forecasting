"""
Download P&G 10-K Annual Filings from SEC Edgar
================================================
Scrapes all P&G (ticker: PG) 10-K annual filings from SEC EDGAR for 2010–2025
and saves them as HTML files to D:\PG\10K_filings.

What is a 10-K?
  - The annual counterpart to the 10-Q. Contains audited financial statements,
    full-year MD&A, risk factors, and disclosure about executive compensation.
  - P&G's fiscal year ends June 30, so FY2024 10-K was filed around August 2024.
  - The 10-K covers the ENTIRE fiscal year — it's more comprehensive than 10-Q.

Difference from download_pg_10q.py:
  - Downloads 10-K (annual) instead of 10-Q (quarterly)
  - Output goes to D:\PG\10K_filings\ (a separate subfolder)
  - Date range: 2010–2025 (longer history than the 10-Q script)
  - Tracks "skipped" count for files that already exist

Output: D:\PG\10K_filings\PG_10K_{report_date}_{accession_no}.html
"""

import os
import time
from datetime import datetime
from edgar import Company, set_identity

# =============================================================================
# Setup
# =============================================================================
set_identity("Claude Code claude@anthropic.com")

# Use a dedicated subfolder for 10-K filings to keep things organized.
# This allows the 10-Qs and 10-Ks to live separately.
output_dir = r"D:\PG\10K_filings"
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# Fetch All 10-K Filings
# =============================================================================
company = Company("PG")

print("Fetching all P&G 10-K filings...")
all_filings = company.get_filings(form="10-K")
print(f"Total 10-K filings found: {len(all_filings)}")

# =============================================================================
# Filter to 2010–2025
# =============================================================================
target_years = set(range(2010, 2026))  # 2010 through 2025
filtered_filings = []

print("\nFiltering for years 2010–2025...")
for filing in all_filings:
    try:
        filing_year = filing.filing_date.year
        if filing_year in target_years:
            filtered_filings.append(filing)
    except Exception as e:
        print(f"  Error parsing filing {filing.accession_no}: {e}")

print(f"Filtered to {len(filtered_filings)} filings")

# =============================================================================
# Download Each Filing
# =============================================================================
print("\nDownloading filings...")
downloaded = 0
skipped = 0      # Already on disk
failed = []

for filing in filtered_filings:
    try:
        # Handle different date types (datetime vs. date vs. None)
        report_date = filing.report_date if hasattr(filing, 'report_date') and filing.report_date else 'unknown'
        if isinstance(report_date, datetime):
            report_date_str = report_date.strftime('%Y-%m-%d')
        else:
            report_date_str = str(report_date)

        acc = filing.accession_no.replace('-', '_')
        filename = f"PG_10K_{report_date_str}_{acc}.html"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"  [EXISTS] {filename}")
            skipped += 1
            continue

        print(f"  [DOWNLOAD] {filename}")
        html = filing.html()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        downloaded += 1
        print(f"    Saved: {filename}")
        time.sleep(0.5)

    except Exception as e:
        print(f"  [ERROR] {filing.accession_no}: {e}")
        failed.append((filing.accession_no, str(e)))
        time.sleep(0.5)

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'='*60}")
print("Complete!")
print(f"Downloaded: {downloaded} new files")
print(f"Skipped (already exist): {skipped}")
print(f"Failed: {len(failed)}")
print(f"Location: {output_dir}")
if failed:
    print("\nFailed filings:")
    for acc, err in failed[:10]:
        print(f"  {acc}: {err}")
print('='*60)