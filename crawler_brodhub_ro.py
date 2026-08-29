import json
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from curl_cffi import requests

BASE_URL = "https://brodhub.eu"
OUTPUT_FILE = "dataset_brodhub_eu.json"

# Template SolrSearch XWiki pentru paginare
SEARCH_URL_TEMPLATE = (
    "https://brodhub.eu/XWiki/SearchGrid/SolrSearch"
    "?f_class=XWiki.FactCheck.FactCheckClass"
    "&hidden=false"
    "&f_locale=ro"
    "&sort=property.XWiki.FactCheck.FactCheckClass.pdate_sortDate"
    "&sortOrder=desc"
    "&f_wiki=xwiki"
    "&rows=30"
    "&r=1"
    "&f_type=DOCUMENT"
    "&text=*"
    "&searchGridId=factcheck-searchgrid"
    "&itemDisplayer=XWiki.FactCheck.FactCheckSearchGridItemDisplayer"
    "&firstIndex={first_index}"
)

REQUEST_TIMEOUT = 25
RETRY_HTTP_CODES = {403, 429, 500, 502, 503, 504}


def build_http_client():
    session = requests.Session(impersonate="chrome124")
    session.headers.update({
        "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://brodhub.eu/ro/metodologie/",
    })
    return session


def get_with_retries(client, url, retries=3, sleep_seconds=2.0):
    last_response = None
    for attempt in range(1, retries + 1):
        try:
            response = client.get(url, timeout=REQUEST_TIMEOUT)
            last_response = response

            if response.status_code == 200:
                return response

            if response.status_code in RETRY_HTTP_CODES and attempt < retries:
                time.sleep(sleep_seconds * attempt)
                continue

            return response
        except requests.RequestsError:
            if attempt >= retries:
                return last_response
            time.sleep(sleep_seconds * attempt)

    return last_response


def get_all_factcheck_links(client, total_pages=43):
    all_links = []
    seen = set()

    for page_num in range(1, total_pages + 1):
        first_index = (page_num - 1) * 30
        page_url = SEARCH_URL_TEMPLATE.format(first_index=first_index)
        print(f"[+] Colectare link-uri Solr (Pagina {page_num}/{total_pages} - firstIndex={first_index})...")

        resp = get_with_retries(client, page_url)
        if not resp or resp.status_code != 200:
            print(f"[-] Pagina {page_num} nu a putut fi încărcată.")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        page_links = 0

        # Caută link-urile din containerul de rezultate
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            
            # Selectează URL-urile către articolele de fact-checking
            if "/factchecks/" in href and not href.startswith("#"):
                full_url = urljoin(BASE_URL, href)
                if full_url not in seen:
                    seen.add(full_url)
                    all_links.append(full_url)
                    page_links += 1

        print(f"    -> găsite {page_links} articole noi (Total acumulat: {len(all_links)})")
        time.sleep(0.4)

    return all_links


def find_claim_review_in_json_ld(soup):
    for script_tag in soup.find_all("script", type="application/ld+json"):
        raw_text = script_tag.get_text(strip=True)
        if not raw_text:
            continue

        try:
            data = json.loads(raw_text)
            nodes = data.get("@graph", [data]) if isinstance(data, dict) else (data if isinstance(data, list) else [data])

            for node in nodes:
                if isinstance(node, dict):
                    node_type = node.get("@type")
                    if node_type == "ClaimReview" or (isinstance(node_type, list) and "ClaimReview" in node_type):
                        return node
        except json.JSONDecodeError:
            continue

    return None


def extract_record_from_claim(claim_json, article_url, analysis_text):
    item_rev = claim_json.get("itemReviewed", {})
    rating = claim_json.get("reviewRating", {})

    verdict_raw = rating.get("alternateName") or rating.get("ratingValue") or "neconcludent"
    verdict = str(verdict_raw).strip().lower()

    statement_originator = "Unknown"
    if isinstance(item_rev, dict):
        author_data = item_rev.get("author", {})
        if isinstance(author_data, dict):
            statement_originator = author_data.get("name", "Unknown")
        elif isinstance(author_data, str):
            statement_originator = author_data

    statement_source = "social_media"
    if isinstance(item_rev, dict) and "appearance" in item_rev:
        appearance = item_rev["appearance"]
        if isinstance(appearance, dict) and "url" in appearance:
            statement_source = appearance["url"]

    author_obj = claim_json.get("author", {})
    if isinstance(author_obj, dict):
        factchecker = author_obj.get("name", "BROD Hub")
    elif isinstance(author_obj, str):
        factchecker = author_obj
    else:
        factchecker = "BROD Hub"

    return {
        "source_country": "Romania / Bulgaria",
        "news_source": "brodhub.eu",
        "verdict": verdict,
        "statement": claim_json.get("claimReviewed", ""),
        "statement_originator": statement_originator,
        "statement_date": item_rev.get("datePublished", "") if isinstance(item_rev, dict) else "",
        "statement_source": statement_source,
        "factchecker": factchecker,
        "factcheck_date": claim_json.get("datePublished", ""),
        "factcheck_analysis_link": article_url,
        "analysis_text": analysis_text,
        "raw_claim_review": claim_json,
    }


def parse_brod_article(client, article_url):
    try:
        resp = get_with_retries(client, article_url)
        if not resp or resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        claim_data = find_claim_review_in_json_ld(soup)

        if not claim_data:
            return None

        # Preluare conținut text
        content_div = soup.select_one(".entry-content, .post-content, #xwikicontent, article")
        analysis_text = ""
        if content_div:
            for junk in content_div.select("script, style, .sharedaddy, .ads"):
                junk.decompose()
            analysis_text = content_div.get_text(separator="\n", strip=True)

        return extract_record_from_claim(claim_data, article_url, analysis_text)

    except Exception as e:
        print(f"[-] Eroare la parsare {article_url}: {e}")
        return None


def main():
    client = build_http_client()

    # 1. Colectare link-uri din toate cele 43 de pagini Solr
    links = get_all_factcheck_links(client, total_pages=43)
    print(f"\n[+] Total link-uri colectate: {len(links)}. Începe extragerea...")

    # 2. Parsare articole
    dataset = []
    for idx, link in enumerate(links, 1):
        print(f"[{idx}/{len(links)}] Parsare: {link}")
        record = parse_brod_article(client, link)
        if record:
            dataset.append(record)
        else:
            print("    -> ClaimReview JSON-LD nu a fost găsit.")
        time.sleep(0.3)

    # 3. Salvare dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"\n[✓] Gata! Au fost salvate {len(dataset)} înregistrări în '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()