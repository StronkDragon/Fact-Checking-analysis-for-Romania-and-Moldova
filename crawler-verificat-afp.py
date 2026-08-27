import json
import re
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from curl_cffi import requests

BASE_URL = "https://verificat.afp.com"
LIST_URL_TEMPLATE = "https://verificat.afp.com/list/Romania?page={page}"
OUTPUT_FILE = "dataset_verificat_afp.json"

REQUEST_TIMEOUT = 25
RETRY_HTTP_CODES = {403, 429, 500, 502, 503, 504}


def build_http_client():
    # impersonate="chrome124" emulează complet handshake-ul TLS/JA3 și HTTP/2 al Chrome
    session = requests.Session(impersonate="chrome124")
    session.headers.update({
        "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://verificat.afp.com/",
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


def extract_doc_links(html_text):
    links = set()
    soup = BeautifulSoup(html_text, "html.parser")

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if "/doc.afp.com" in href:
            links.add(urljoin(BASE_URL, href))

    regex_links = re.findall(r"(?:https?://verificat\.afp\.com)?(/doc\.afp\.com[^\"'\s<]+)", html_text)
    for partial in regex_links:
        links.add(urljoin(BASE_URL, partial))

    return links


def get_article_links(client, start_page=0, end_page=4):
    article_links = set()
    for page in range(start_page, end_page + 1):
        url = LIST_URL_TEMPLATE.format(page=page)
        print(f"[+] Colectare link-uri de pe pagina {page}: {url}")

        resp = get_with_retries(client, url)
        if not resp or resp.status_code != 200:
            status = resp.status_code if resp else "Fără răspuns"
            print(f"[-] Eroare {status} la {url}")
            continue

        links = extract_doc_links(resp.text)
        print(f"    -> găsite {len(links)} link-uri pe pagina {page}")
        article_links.update(links)
        time.sleep(1.0)

    return list(article_links)


def iter_jsonld_objects(json_payload):
    if isinstance(json_payload, dict):
        if "@graph" in json_payload and isinstance(json_payload["@graph"], list):
            for node in json_payload["@graph"]:
                if isinstance(node, dict):
                    yield node
        yield json_payload
    elif isinstance(json_payload, list):
        for item in json_payload:
            if isinstance(item, dict):
                yield item


def parse_claim_review(client, article_url):
    try:
        resp = get_with_retries(client, article_url)
        if not resp or resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        json_ld_tags = soup.find_all("script", type="application/ld+json")

        for tag in json_ld_tags:
            raw_jsonld = tag.get_text(strip=True)
            if not raw_jsonld:
                continue

            try:
                data = json.loads(raw_jsonld)
                for item in iter_jsonld_objects(data):
                    if item.get("@type") == "ClaimReview":
                        item["source_url"] = article_url
                        return item
            except json.JSONDecodeError:
                continue

    except Exception as e:
        print(f"[-] Eroare la parsarea articolului {article_url}: {e}")

    return None


def main():
    client = build_http_client()
    links = get_article_links(client, start_page=0, end_page=4)
    print(f"\n[+] Au fost găsite {len(links)} articole unice. Începe extragerea...")

    dataset = []
    for idx, link in enumerate(links, 1):
        print(f"[{idx}/{len(links)}] Parsare: {link}")
        record = parse_claim_review(client, link)
        if record:
            dataset.append(record)
        else:
            print("    -> ClaimReview negăsit")
        time.sleep(0.5)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"\n[✓] Gata! Au fost salvate {len(dataset)} înregistrări în '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()