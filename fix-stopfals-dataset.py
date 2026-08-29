import json
import re
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests


def build_http_session() -> cffi_requests.Session:
    session = cffi_requests.Session(impersonate="chrome124")
    session.headers.update({
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://stopfals.md/ro/campania/articole-fact-checking",
        "X-Requested-With": "XMLHttpRequest",
    })
    return session


# -------------------------------------------------------------
# Strategy 1: Fetch Date Mapping from the API
# -------------------------------------------------------------
def try_fetch_api_dates(session: cffi_requests.Session) -> dict:
    url_to_date = {}
    base_url = "https://stopfals.md"
    
    # Test candidate API URL structures
    candidate_bases = [
        f"{base_url}/ro/articles/4",
        f"{base_url}/articles/4",
        f"{base_url}/ro/api/articles/4",
        f"{base_url}/api/articles/4",
    ]

    working_base = None
    total_pages = 1

    for cand in candidate_bases:
        test_url = f"{cand}?page=1&lang=ro"
        try:
            r = session.get(test_url, timeout=5)
            if r.status_code == 200 and "pagination" in r.text:
                working_base = cand
                payload = r.json()
                total_pages = payload.get("pagination", {}).get("pageCount", 1)
                print(f"[+] Connected to working API endpoint: {cand} (Total pages: {total_pages})")
                break
        except Exception:
            continue

    if not working_base:
        print("[!] Could not connect via API endpoints. Switching to direct HTML date extraction...")
        return {}

    for page in range(1, total_pages + 1):
        api_url = f"{working_base}?page={page}&lang=ro"
        try:
            r = session.get(api_url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                items = data.get("data") or data.get("articles") or data.get("results") or (data if isinstance(data, list) else [])
                
                for item in items:
                    if isinstance(item, dict):
                        slug = item.get("url") or item.get("slug") or ""
                        full_url = urljoin(base_url, slug if slug.startswith("/") else f"/ro/article/{slug}")
                        real_date = item.get("prettyCreatedAt") or item.get("created_at") or item.get("date") or ""
                        
                        if full_url and real_date and not real_date.startswith("{"):
                            url_to_date[full_url] = real_date
            time.sleep(0.05)
        except Exception:
            continue

    return url_to_date


# -------------------------------------------------------------
# Strategy 2: Fast HTML Date Extraction (Fallback)
# -------------------------------------------------------------
def extract_date_from_article_html(session: cffi_requests.Session, url: str) -> str:
    try:
        r = session.get(url, timeout=10)
        if r.status_code != 200:
            return ""

        soup = BeautifulSoup(r.text, "html.parser")

        # 1. Primary target: author details or main article header
        for selector in [
            ".page--article .author-details .date",
            ".page--article .wrap--image .info .post-details .date",
            ".page--article .post-details .date",
            ".author-details span.date",
        ]:
            tag = soup.select_one(selector)
            if tag and tag.get_text(strip=True) and not tag.get_text(strip=True).startswith("{"):
                return tag.get_text(strip=True)

        # 2. Secondary target: meta tags
        meta_tag = soup.find("meta", property="article:published_time")
        if meta_tag and meta_tag.get("content") and not meta_tag.get("content").startswith("{"):
            return meta_tag["content"]

        # 3. Regex search for Romanian date pattern: DD luna YYYY
        match = re.search(r"\b\d{1,2}\s+(?:ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|septembrie|octombrie|noiembrie|decembrie)\s+\d{4}\b", r.text, re.IGNORECASE)
        if match:
            return match.group(0)

    except Exception:
        pass
    return ""


# -------------------------------------------------------------
# Main Patching Logic
# -------------------------------------------------------------
def patch_dataset(input_file: str, output_file: str):
    session = build_http_session()
    
    # Step 1: Try API first (takes ~2 seconds)
    date_lookup = try_fetch_api_dates(session)

    # Step 2: Read records
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"[*] Processing {len(records)} records from {input_file}...")

    updated_count = 0
    for idx, rec in enumerate(records, start=1):
        link = rec.get("factcheck_analysis_link", "")
        current_date = rec.get("factcheck_date", "")

        # Check if date needs fixing
        if current_date == "{article.prettyCreatedAt}" or not current_date or current_date.startswith("{"):
            # Option A: Check API lookup map
            if link in date_lookup:
                rec["factcheck_date"] = date_lookup[link]
                updated_count += 1
            else:
                # Option B: Fast HTTP request directly to article page
                real_date = extract_date_from_article_html(session, link)
                if real_date:
                    rec["factcheck_date"] = real_date
                    updated_count += 1
                time.sleep(0.05)

        if idx % 50 == 0 or idx == len(records):
            print(f"    -> Checked {idx}/{len(records)} records (Fixed so far: {updated_count})")

    # Step 3: Write out clean JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[+] Finished! Successfully fixed {updated_count} dates.")
    print(f"[+] Clean dataset written to: {output_file}")


if __name__ == "__main__":
    patch_dataset("dataset_stopfals-md.jsonl", "dataset_stopfals-md_clean.jsonl")