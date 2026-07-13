"""
Download SEC Filings (HTML) for P&G and Unilever
================================================
Downloads the primary document HTML for each filing listed in
financial_statements_links.txt. The filing index page is fetched first,
and then the actual primary document (the 10-K/10-Q/20-F) is extracted.

Why two-step fetching?
  - SEC EDGAR's direct URL for the primary filing document is not always
    predictable from the accession number alone — it depends on the filing
    structure within the SEC archive.
  - The index page (accession-no-index.htm) has a table linking to all
    documents in the filing package. We find the correct link in that table.
  - This is more robust than guessing the document URL format.

Three filing types handled:
  - 10-K: P&G and Unilever annual filings (US GAAP)
  - 10-Q: P&G quarterly filings
  - 20-F: Unilever annual filing (IFRS, as a foreign private issuer)

File naming convention:
  {Ticker}_{Form}_{FilingDate}_{CIK}_{AccessionNo}.html
  Example: PG_10K_2010-06-30_0000080424_0000080424-10-000012.html

Output directories:
  D:\PG\10K_filings\   — 10-K annual filings (PG and UL)
  D:\PG\10Q_filings\   — 10-Q quarterly filings (PG only)
  D:\PG\20F_filings\   — 20-F annual filings (UL only)
"""

import os
import time
import re
import urllib.request

BASE = r"D:\PG"

# Create output directories
os.makedirs(BASE + r"\10K_filings", exist_ok=True)
os.makedirs(BASE + r"\10Q_filings", exist_ok=True)
os.makedirs(BASE + r"\20F_filings", exist_ok=True)

# HTTP headers — SEC requires a User-Agent identifying you
# Using a descriptive string helps if you ever need to contact SEC for issues.
headers = {"User-Agent": "Investor claude@anthropic.com", "Accept": "text/html,application/xhtml+xml"}

# =============================================================================
# Helper: Fetch a URL with error handling
# =============================================================================
def fetch(url, timeout=15):
    """
    Fetch a URL and return (html_bytes, status_or_error_message).

    Returns (None, error_string) on failure instead of raising —
    this lets us handle failures gracefully and continue the batch.
    """
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(), r.status
    except Exception as e:
        return None, str(e)

# =============================================================================
# Helper: Extract the primary document URL from an EDGAR index page
# =============================================================================
def get_doc_url(cik, accession, form):
    """
    Fetch an EDGAR filing index page and extract the link to the primary document.

    The index page is at:
      https://www.sec.gov/Archives/edgar/data/{CIK}/{folder}/{accession}-index.htm

    We then parse the HTML table looking for a row matching the form type,
    and extract the link to the .htm or .html document (the primary filing text).
    SEC filings are sometimes split across multiple documents; we grab the first
    .htm[l] link in the matching row, which is typically the main filing.

    Returns: (doc_url, filename) or (None, None) if not found.
    """
    folder = accession.replace("-", "")  # Folder name = accession without hyphens
    idx_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{accession}-index.htm"

    html, status = fetch(idx_url)
    if not html:
        return None, None

    html_str = html.decode("utf-8", errors="ignore")

    # Parse the HTML table for this filing form
    # Each <tr> in the documents table contains the form type and a link
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html_str, re.DOTALL)
    for row in rows:
        # The row should contain the form type (10-K, 10-Q, 20-F)
        if form_tag in row:
            # Find all /Archives links in this row
            links = re.findall(r'href="(/Archives/edgar/data/[^"]+)"', row)
            for link in links:
                # Skip XBRL (.xml) and other non-HTML documents
                # The primary filing is always an .htm or .html file
                if link.endswith(".htm") or link.endswith(".html"):
                    return "https://www.sec.gov" + link, link.split("/")[-1]
    return None, None

# =============================================================================
# Main Download Loop
# =============================================================================
def run():
    # Parse the link file — fields are separated by " | "
    lines = open(r"D:\PG\financial_statements_links.txt", "r").read().strip().split("\n")

    downloaded = 0
    skipped = 0
    failed = []

    for line in lines:
        parts = line.split(" | ")
        if len(parts) < 5:
            continue  # Skip blank lines and section headers

        _, filing_date, form, accession, old_url = parts
        cik = "0000080424" if "PG" in line else "0000217410"  # P&G vs. Unilever CIK
        folder = "10K_filings" if form == "10-K" else ("10Q_filings" if form == "10-Q" else "20F_filings")

        # Build local filename: avoid filesystem issues with special characters
        local_name = f"{'PG' if 'PG' in line else 'UL'}_{form.replace('-','')}_{filing_date}_{cik}_{accession}.html"
        out_path = os.path.join(BASE, folder, local_name)

        if os.path.exists(out_path):
            skipped += 1
            continue  # Already downloaded

        # Step 1: Find the primary document URL from the index page
        doc_url, fname = get_doc_url(cik, accession, form)
        if not doc_url:
            failed.append(f"{'PG' if 'PG' in line else 'UL'} {form} {accession} => index fetch failed")
            time.sleep(0.3)
            continue

        # Step 2: Download the primary document
        body, status = fetch(doc_url)
        if body and len(body) > 500:  # Sanity check: real HTML filings are at least 500 bytes
            with open(out_path, "wb") as f:
                f.write(body)
            print(f"[OK] {local_name}")
            print(f"     {doc_url}")
            downloaded += 1
        else:
            failed.append(f"{'PG' if 'PG' in line else 'UL'} {form} {accession} => HTTP {status}")
            print(f"[FAIL] {local_name} ({status})")

        time.sleep(0.4)  # SEC rate limit

    # =============================================================================
    # Summary
    # =============================================================================
    print(f"\nDone. Downloaded={downloaded} skipped={skipped} failed={len(failed)}")
    if failed:
        print("\nFailed filings (showing up to 10):")
        for f in failed[:10]:
            print(f"  {f}")

if __name__ == "__main__":
    run()