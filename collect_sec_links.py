"""
Collect SEC Filing Links for P&G and Unilever (2010–2025)
=========================================================
Scrapes the SEC EDGAR filing index to collect URLs for all 10-K and 10-Q
filings for P&G (PG) and Unilever (UL) from 2010 to 2025.

Why collect links instead of downloading the files directly?
  - Sometimes you want the URLs for other purposes (archival, audit trail,
    re-downloading later, cross-referencing with other datasets).
  - This script is much faster than downloading full HTML — it only fetches
    the filing index pages, not the documents themselves.
  - The output file is used as input by download_sec_html.py.

Output format: D:\PG\financial_statements_links.txt
  Each entry is 5 fields separated by " | ":
    [Ticker] | [Filing Date] | [Form Type] | [Accession No] | [SEC URL]

Note on accession numbers:
  - The accession number is a unique identifier per filing within EDGAR.
  - It's formatted as CIK-COUNTIFIER in SEC URLs (e.g., 0000080424-23-000012).
  - We remove hyphens in filenames for filesystem compatibility.

Dual-company analysis:
  - P&G (PG): 10-K + 10-Q filings — US domestic company
  - Unilever (UL): 10-K (for NYSE-listed shares) filings — cross-listed company
    Note: Unilever files 20-F with the SEC (its annual filing for foreign private
    issuers), but this script focuses on 10-K/10-Q for simplicity.

Rate limiting:
  - EDGAR asks for 0.5s between requests.
  - This script is lightweight (index pages only), so the rate is manageable.
"""

import os
from edgar import Company, set_identity

set_identity("Claude Code claude@anthropic.com")

output_path = r"D:\PG\financial_statements_links.txt"

# Companies to process: (display name, Company object)
companies = [("P&G", Company("PG")), ("Unilever", Company("UL"))]

forms = ["10-K", "10-Q"]  # Annual and quarterly US filings
years = range(2010, 2026)  # 2010 through 2025 inclusive

lines = []  # Accumulate all lines, write once at the end

for name, company in companies:
    # Section separator
    lines.append(f"\n{'='*60}\n{name} ({'PG' if name=='P&G' else 'UL'})\n{'='*60}\n")

    for form in forms:
        lines.append(f"\n--- {form} ---\n")
        for year in years:
            try:
                # company.get_filings() returns all filings for the given form in the given year.
                # Note: EDGAR's year filter is based on the filing date, not the fiscal period.
                filings = company.get_filings(form=form, year=year)
                for f in filings:
                    # filing_date is a datetime.date object
                    date_str = str(f.filing_date)[:10]   # "YYYY-MM-DD" format
                    # f.path is the relative URL: /Archives/edgar/data/...
                    url = f"https://www.sec.gov{f.path}"
                    lines.append(f"{date_str}  {form}  {f.accession_no}\n  {url}\n")
            except Exception as e:
                # Skip years with errors — EDGAR sometimes returns empty for older years
                lines.append(f"  [ERROR year {year} {form}: {e}]\n")

# Write all lines to file in one operation
with open(output_path, "w", encoding="utf-8") as f:
    f.writelines(lines)

# Count URLs written (each URL starts with "https://www.sec.gov")
total = len([l for l in lines if l.startswith("https")])
print(f"Done. Wrote {total} filing links to {output_path}")