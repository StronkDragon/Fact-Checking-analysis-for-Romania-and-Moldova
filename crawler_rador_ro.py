import json
import re
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from curl_cffi import requests

BASE_URL = "https://www.rador.ro"
AJAX_URL = "https://www.rador.ro/wp-admin/admin-ajax.php?td_theme_name=Newspaper&v=7.4"
CATEGORY_PAGE_URL = "https://www.rador.ro/romania-check/articole-romania-check/"
OUTPUT_FILE = "dataset_rador_ro.json"

REQUEST_TIMEOUT = 25
RETRY_HTTP_CODES = {403, 429, 500, 502, 503, 504}


def build_http_client():
    session = requests.Session(impersonate="chrome124")
    session.headers.update({
        "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": CATEGORY_PAGE_URL,
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
        except requests.RequestsError as exc:
            if attempt >= retries:
                return last_response
            time.sleep(sleep_seconds * attempt)
    return last_response


def extract_links_from_html(html_text):
    links = set()
    soup = BeautifulSoup(html_text, "html.parser")
    # Caută link-urile din blocurile de module Newspaper
    for h3 in soup.select("h3.entry-title a, .td-module-title a"):
        href = h3.get("href", "").strip()
        if href and "rador.ro" in href and "/romania-check/" not in href:
            links.add(href)
    return links


def get_all_rador_links(client):
    """
    Descarcă prima pagină și apoi interoghează AJAX-ul temei Newspaper
    până sunt colectate toate cele ~86 de articole.
    """
    seen_links = set()
    all_links = []

    print(f"[+] Descărcare pagina principală: {CATEGORY_PAGE_URL}")
    resp = get_with_retries(client, CATEGORY_PAGE_URL)
    if not resp or resp.status_code != 200:
        print("[-] Nu s-a putut descărca pagina principală.")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    initial_links = extract_links_from_html(resp.text)
    for lnk in initial_links:
        if lnk not in seen_links:
            seen_links.add(lnk)
            all_links.append(lnk)

    print(f"    -> {len(all_links)} articole găsite inițial.")

    # Extragem parametrii pentru AJAX din scriptul paginii (tdBlock atts)
    block_match = re.search(r"block_td_uid_[^.]+\.atts\s*=\s*'([^']+)';", resp.text)
    block_uid_match = re.search(r'block_td_uid_[^.]+\.id\s*=\s*"([^"]+)";', resp.text)
    max_pages_match = re.search(r"block_td_uid_[^.]+\.max_num_pages\s*=\s*\"?(\d+)\"?", resp.text)

    if block_match and block_uid_match:
        block_atts = block_match.group(1)
        block_uid = block_uid_match.group(1)
        max_pages = int(max_pages_match.group(1)) if max_pages_match else 9

        print(f"[+] Paginare AJAX detectată ({max_pages} pagini în total). Colectare...")

        for current_page in range(2, max_pages + 1):
            payload = {
                "action": "td_ajax_block",
                "td_atts": block_atts,
                "td_block_id": block_uid,
                "td_column_number": "3",
                "td_current_page": str(current_page),
                "block_type": "td_block_15",
            }
            try:
                ajax_resp = client.post(AJAX_URL, data=payload, timeout=REQUEST_TIMEOUT)
                if ajax_resp.status_code == 200:
                    data = ajax_resp.json()
                    html_chunk = data.get("td_data", "")
                    chunk_links = extract_links_from_html(html_chunk)
                    new_found = 0
                    for lnk in chunk_links:
                        if lnk not in seen_links:
                            seen_links.add(lnk)
                            all_links.append(lnk)
                            new_found += 1
                    print(f"    -> Pagina AJAX {current_page}/{max_pages}: +{new_found} articole (Total: {len(all_links)})")
                else:
                    print(f"    [-] Eroare status AJAX {ajax_resp.status_code} la pagina {current_page}")
            except Exception as e:
                print(f"    [-] Eroare AJAX la pagina {current_page}: {e}")
            time.sleep(0.6)

    return all_links


def determine_verdict_and_statement(raw_title):
    title = raw_title.strip()
    verdict = "neconcludent"
    statement = title

    lower = title.lower()
    if re.match(r"^nu\b", lower):
        verdict = "fals"
        statement = re.sub(r"^nu\s*[,:\-–—\.\s]\s*", "", title, flags=re.IGNORECASE).strip()
    elif re.match(r"^da\b", lower):
        verdict = "adevarat"
        statement = re.sub(r"^da\s*[,:\-–—\.\s]\s*", "", title, flags=re.IGNORECASE).strip()
    elif "fals" in lower.split()[:2]:
        verdict = "fals"
        statement = re.sub(r"^fals\s*[,:\|\-–—\.\s]\s*", "", title, flags=re.IGNORECASE).strip()
    elif "adevărat" in lower.split()[:2] or "adevarat" in lower.split()[:2]:
        verdict = "adevarat"
        statement = re.sub(r"^adev[aă]rat\s*[,:\|\-–—\.\s]\s*", "", title, flags=re.IGNORECASE).strip()

    return verdict, statement


def parse_article_page(client, article_url):
    try:
        resp = get_with_retries(client, article_url)
        if not resp or resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # 1. Titlu & Afirmație / Verdict
        title_el = soup.select_one("h1.entry-title, h1")
        raw_title = title_el.get_text(strip=True) if title_el else ""
        verdict, statement = determine_verdict_and_statement(raw_title)

        # 2. Dată articol
        time_el = soup.select_one("time.entry-date, time.published, time")
        factcheck_date = time_el.get("datetime") if (time_el and time_el.get("datetime")) else (time_el.get_text(strip=True) if time_el else "")

        # 3. Factchecker
        author_el = soup.select_one(".td-post-author-name a, .author a, .entry-author a")
        factchecker = author_el.get_text(strip=True) if author_el else "Rador Romania Check"

        # 4. Text Corp / Analiză
        content_div = soup.select_one(".td-post-content, .entry-content, article .post-content")
        analysis_text = ""
        statement_originator = "Necunoscut"
        statement_source = "online"

        if content_div:
            for junk in content_div.select("script, style, .adsbygoogle, .sharedaddy"):
                junk.decompose()
            analysis_text = content_div.get_text(separator="\n", strip=True)

            lower_body = analysis_text.lower()
            if "tiktok" in lower_body:
                statement_source = "TikTok"
            elif "facebook" in lower_body:
                statement_source = "Facebook"
            elif "telegram" in lower_body:
                statement_source = "Telegram"
            elif "whatsapp" in lower_body:
                statement_source = "WhatsApp"
            elif "rețelele sociale" in lower_body or "retelele de socializare" in lower_body:
                statement_source = "social_media"

            # Detectare autor de la final ("Autor: Nume")
            author_match = re.search(r"Autor:\s*([A-ZĂÂÎȘȚa-zăâîșț\s\-]+)$", analysis_text, re.MULTILINE)
            if author_match:
                factchecker = author_match.group(1).strip()

        return {
            "source_country": "Romania",
            "news_source": "rador.ro",
            "verdict": verdict,
            "statement": statement,
            "raw_title": raw_title,
            "statement_originator": statement_originator,
            "statement_date": "",
            "statement_source": statement_source,
            "factchecker": factchecker,
            "factcheck_date": factcheck_date,
            "factcheck_analysis_link": article_url,
            "analysis_text": analysis_text
        }

    except Exception as e:
        print(f"[-] Eroare la parsare {article_url}: {e}")
        return None


def main():
    client = build_http_client()
    links = get_all_rador_links(client)
    print(f"\n[+] Total link-uri de extras: {len(links)}. Începe parsarea articolelor...")

    dataset = []
    for idx, link in enumerate(links, 1):
        print(f"[{idx}/{len(links)}] Parsare: {link}")
        record = parse_article_page(client, link)
        if record:
            dataset.append(record)
        time.sleep(0.4)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"\n[✓] Gata! Au fost salvate {len(dataset)} înregistrări în '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()