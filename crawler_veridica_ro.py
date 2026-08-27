import json
import re
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from curl_cffi import requests

BASE_URL = "https://www.veridica.ro"
LIST_URL_TEMPLATE = "https://www.veridica.ro/fake-news-dezinformare-propaganda?page={page}"
OUTPUT_FILE = "dataset_veridica_ro.json"

REQUEST_TIMEOUT = 25
RETRY_HTTP_CODES = {403, 429, 500, 502, 503, 504}


def build_http_client():
    session = requests.Session(impersonate="chrome124")
    session.headers.update({
        "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.veridica.ro/",
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
                wait = sleep_seconds * attempt
                print(f"[!] Status {response.status_code} pentru {url}. Reîncerc în {wait:.1f}s...")
                time.sleep(wait)
                continue

            return response
        except requests.RequestsError as exc:
            if attempt >= retries:
                print(f"[-] Eroare request pentru {url}: {exc}")
                return last_response

            wait = sleep_seconds * attempt
            print(f"[!] Eroare temporară pentru {url}: {exc}. Reîncerc în {wait:.1f}s...")
            time.sleep(wait)

    return last_response


def extract_article_links_from_page(html_text):
    links = set()
    soup = BeautifulSoup(html_text, "html.parser")

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        # Veridica plasează articolele în categoriile tematice
        if "/fake-news-dezinformare-propaganda/" in href and not href.endswith("?page="):
            full_url = urljoin(BASE_URL, href)
            # Excludem link-urile de paginare/filtre
            if "?page=" not in full_url:
                links.add(full_url)

    return links


def get_all_article_links(client, max_pages=5):
    all_links = []
    seen = set()

    for page in range(1, max_pages + 1):
        url = LIST_URL_TEMPLATE.format(page=page)
        print(f"[+] Colectare link-uri de pe pagina {page}: {url}")

        resp = get_with_retries(client, url)
        if not resp or resp.status_code != 200:
            print(f"[-] Pagina {page} nu a putut fi încărcată.")
            break

        links_found = extract_article_links_from_page(resp.text)
        new_links = [lnk for lnk in links_found if lnk not in seen]

        if not new_links:
            print(f"    -> Niciun articol nou pe pagina {page}. Oprire.")
            break

        for lnk in new_links:
            seen.add(lnk)
            all_links.append(lnk)

        print(f"    -> găsite {len(new_links)} articole noi (Total: {len(all_links)})")
        time.sleep(1.0)

    return all_links


def parse_structured_body(body_text):
    """
    Extrage secțiunile specifice din formatul standard al articolelor Veridica.
    """
    sections = {
        "stire": None,
        "naratiuni": None,
        "obiective": None,
        "de_ce_sunt_false": None,
        "context": None,
    }

    stire_m = re.search(r"ȘTIRE:(.*?)(?=NARAȚIUNI:|$)", body_text, re.DOTALL | re.IGNORECASE)
    naratiuni_m = re.search(r"NARAȚIUNI:(.*?)(?=OBIECTIVE:|$)", body_text, re.DOTALL | re.IGNORECASE)
    obiective_m = re.search(r"OBIECTIVE:(.*?)(?=DE CE SUNT FALSE NARAȚIUNILE:|$)", body_text, re.DOTALL | re.IGNORECASE)
    false_m = re.search(r"DE CE SUNT FALSE NARAȚIUNILE:(.*?)(?=CONTEXT:|$)", body_text, re.DOTALL | re.IGNORECASE)
    context_m = re.search(r"CONTEXT:(.*?)$", body_text, re.DOTALL | re.IGNORECASE)

    if stire_m: sections["stire"] = stire_m.group(1).strip()
    if naratiuni_m: sections["naratiuni"] = naratiuni_m.group(1).strip()
    if obiective_m: sections["obiective"] = obiective_m.group(1).strip()
    if false_m: sections["de_ce_sunt_false"] = false_m.group(1).strip()
    if context_m: sections["context"] = context_m.group(1).strip()

    return sections


def parse_veridica_article(client, article_url):
    try:
        resp = get_with_retries(client, article_url)
        if not resp or resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        json_ld_tags = soup.find_all("script", type="application/ld+json")

        for tag in json_ld_tags:
            raw_text = tag.get_text(strip=True)
            if not raw_text:
                continue

            try:
                data = json.loads(raw_text)
                if data.get("@type") == "NewsArticle":
                    # Adăugăm URL-ul sursă
                    data["source_url"] = article_url
                    
                    # Extragem componentele structurate din corpul textului
                    body = data.get("articleBody", "")
                    data["extracted_sections"] = parse_structured_body(body)
                    
                    return data
            except json.JSONDecodeError:
                continue

    except Exception as e:
        print(f"[-] Eroare la parsare {article_url}: {e}")

    return None


def main():
    client = build_http_client()
    
    # Numărul de pagini din listă pe care vrei să le parcurgi
    links = get_all_article_links(client, max_pages=39)
    print(f"\n[+] Colectate {len(links)} articole. Începe extragerea...")

    dataset = []
    for idx, link in enumerate(links, 1):
        print(f"[{idx}/{len(links)}] Parsare: {link}")
        record = parse_veridica_article(client, link)
        if record:
            dataset.append(record)
        else:
            print("    -> NewsArticle JSON-LD negăsit")
        time.sleep(0.5)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"\n[✓] Gata! Au fost salvate {len(dataset)} înregistrări în '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
