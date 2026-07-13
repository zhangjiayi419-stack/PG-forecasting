"""
Download P&G 10-Q Quarterly Filings from SEC Edgar
==================================================
Scrapes all P&G (ticker: PG) 10-Q filings from SEC EDGAR for the years 2018–2025
and saves them as HTML files to D:\PG.

What is a 10-Q?
  - SEC filing submitted quarterly (hence "Q") by all publicly traded companies.
  - Contains unaudited financial statements, MD&A (Management Discussion & Analysis),
    and risk disclosures.
  - Contrast with 10-K: filed annually, includes audited statements.
  - P&G files 10-Qs for Q1, Q2, Q3 of each fiscal year (not Q4 — the 10-K
    covers the full year).

Why download the HTML directly instead of XBRL / raw data?
  - HTML retains the original document structure as filed, including footnotes
    and cross-references that structured datasets often strip out.
  - Good for text analysis (NLP on MD&A), visual parsing, or archival purposes.
  - The filing index page on SEC.gov is the entry point; the primary HTML document
    contains the full text.

SEC Rate Limiting:
  - The SEC asks that you sleep ≥ 0.5 seconds between requests.
  - Uses the edgar Python package (sec-api/edgar-py) which handles redirects,
    filing parsing, and rate limiting gracefully.

Output: D:\PG\PG_10Q_{report_date}_{accession_no}.html
"""

import os
import time
from datetime import datetime
from edgar import Company, set_identity

# =============================================================================
# Setup — SEC Identity and Output Directory
# =============================================================================
# SEC requires an identity string (email) in all requests per their robots policy.
# This identifies you as a responsible API user. Use a real email.
# The edgar package appends this to the User-Agent header automatically.
set_identity("Claude Code claude@anthropic.com")

# Create the output directory if it doesn't exist.
# r"D:\PG" uses a raw string to avoid backslash-escape issues on Windows.
output_dir = r"D:\PG"
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# Connect to P&G's EDGAR Company object
# =============================================================================
# The edgar package caches EDGAR company data. Company("PG") returns a proxy
# that lets us query filings by form type.
company = Company("PG")

# =============================================================================
# Fetch All 10-Q Filings
# =============================================================================
print("Fetching all P&G 10-Q filings...")
all_filings = company.get_filings(form="10-Q")
print(f"Total 10-Q filings found: {len(all_filings)}")

# =============================================================================
# Filter to Target Years (2018–2025)
# =============================================================================
# EDGAR may return filings going back decades. We only want recent ones.
# Using a set for O(1) membership tests instead of a list.
target_years = set(range(2018, 2026))  # 2018, 2019, ..., 2025
filtered_filings = []

print("\nFiltering for years 2018–2025...")
for filing in all_filings:
    try:
        # filing.filing_date is a datetime.date object (NOT a string).
        # Accessing .year directly on it is much faster than string parsing.
        filing_year = filing.filing_date.year
        if filing_year in target_years:
            filtered_filings.append(filing)
    except Exception as e:
        # Skip any malformed filing that doesn't have a valid filing_date.
        # This is rare but happens with certain legacy EDGAR entries.
        print(f"  Error parsing filing {filing.accession_no}: {e}")

print(f"Filtered to {len(filtered_filings)} filings")

# =============================================================================
# Download Each Filing
# =============================================================================
print("\nDownloading filings...")
downloaded = 0
failed = []

for filing in filtered_filings:
    try:
        # Get the report date (the period covered by the filing).
        # filing.report_date may be a datetime, a date, or missing — handle all cases.
        report_date = filing.report_date if hasattr(filing, 'report_date') and filing.report_date else 'unknown'
        if isinstance(report_date, datetime):
            report_date_str = report_date.strftime('%Y-%m-%d')
        else:
            report_date_str = str(report_date)

        # accession_no is the unique EDGAR identifier.
        # We remove hyphens to avoid filesystem issues and ensure clean filenames.
        acc = filing.accession_no.replace('-', '_')
        filename = f"PG_10Q_{report_date_str}_{acc}.html"
        filepath = os.path.join(output_dir, filename)

        # Skip if already downloaded — allows re-running this script without
        # re-fetching everything. Deletes the file to re-download.
        if os.path.exists(filepath):
            print(f"  [EXISTS] {filename}")
            continue

        print(f"  [DOWNLOAD] {filename}")

        # filing.html() fetches the primary document (the actual filing text).
        # This is the document in /Archives/edgar/data/{cik}/{folder}/{filename}.htm
        html = filing.html()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        downloaded += 1
        print(f"    Saved: {filename}")

        # SEC rate limit: ≥0.5s between requests. Be respectful — EDGAR
        # will block IP addresses that hammer the server.
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
print(f"Failed:     {len(failed)}")
print(f"Location:   {output_dir}")
if failed:
    print("\nFailed filings:")
    for acc, err in failed[:10]:  # Print up to 10 failures
        print(f"  {acc}: {err}")
print('='*60)