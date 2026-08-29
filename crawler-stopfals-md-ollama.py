import json
import re
import time
from typing import List, Literal, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests
from ollama import chat
from pydantic import BaseModel, Field



# -------------------------------------------------------------
# 1. Pydantic Schemas with Strict Verdict Taxonomy
# -------------------------------------------------------------
VerdictType = Literal["True", "False", "Mostly True", "Mostly False", "Mixture"]

class FactCheckRecord(BaseModel):
    source_country: str = "Moldova"
    news_source: str = "Stopfals.md"
    verdict: VerdictType = Field(description="Verdict label: True, False, Mostly True, Mostly False, or Mixture")
    statement: str = Field(description="The exact statement or claim being fact-checked")
    statement_originator: str = Field(description="Who made the statement, or 'Unknown'")
    statement_date: str = Field(default="", description="Date of the statement, or empty string")
    statement_source: str = Field(default="social_media", description="Medium: social_media, tv, press, web, etc.")
    factchecker: str = Field(default="Stopfals Team", description="Author or editorial team")
    factcheck_date: str = Field(description="Publication date of the fact check")
    factcheck_analysis_link: str = Field(description="Full article URL")
    analysis_text: str = Field(description="Concise summary of the fact-check reasoning")


class LLMSemanticFields(BaseModel):
    verdict: VerdictType = Field(description="Strictly classify into one of: True, False, Mostly True, Mostly False, Mixture")
    statement: str = Field(description="The core claim or narrative verified in the text")
    statement_originator: str = Field(default="Unknown", description="Person/entity that originated the claim")
    statement_date: str = Field(default="", description="Date of the statement if stated, or empty")
    statement_source: str = Field(default="social_media", description="Channel: social_media, tv, press, web, etc.")
    analysis_summary: str = Field(description="Max 2 sentences explaining why this verdict was assigned")


# -------------------------------------------------------------
# 2. HTTP Client Configuration
# -------------------------------------------------------------
def build_http_client() -> cffi_requests.Session:
    session = cffi_requests.Session(impersonate="chrome124")
    session.headers.update({
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://stopfals.md/ro/campania/articole-fact-checking",
        "X-Requested-With": "XMLHttpRequest",
    })
    return session


# -------------------------------------------------------------
# 3. Direct API Link Discovery via `/articles/4`
# -------------------------------------------------------------
def get_all_stopfals_links(session: cffi_requests.Session, category_id: int = 4, lang: str = "ro") -> List[str]:
    base_url = "https://stopfals.md"
    discovered_links = set()

    # Step 1: Initial request to discover total pageCount
    initial_api_url = f"{base_url}/articles/{category_id}?page=1&lang={lang}"
    print(f"[*] Fetching pagination info from: {initial_api_url}")

    try:
        res = session.get(initial_api_url, timeout=15)
        if res.status_code != 200:
            print(f"[!] Failed to fetch initial API page: HTTP {res.status_code}")
            return []

        payload = res.json()
        pagination = payload.get("pagination", {})
        total_pages = pagination.get("pageCount", 1)
        total_rows = pagination.get("rowCount", 0)
        print(f"[+] Total articles: {total_rows} across {total_pages} API pages.\n")

    except Exception as e:
        print(f"[!] Error querying API metadata: {e}")
        return []

    # Step 2: Iterate across all API pages
    for page in range(1, total_pages + 1):
        api_url = f"{base_url}/articles/{category_id}?page={page}&lang={lang}"
        print(f"[*] Querying API [Page {page}/{total_pages}]: {api_url}")

        try:
            res = session.get(api_url, timeout=15)
            if res.status_code != 200:
                print(f"[!] Warning: HTTP {res.status_code} on page {page}")
                continue

            data = res.json()
            page_text = json.dumps(data)

            # Extract full article URLs or slugs from the JSON tree
            matches = re.findall(r'/(?:ro|ru)/article/[a-zA-Z0-9\-]+', page_text)
            for match in matches:
                clean_url = urljoin(base_url, match)
                discovered_links.add(clean_url)

            time.sleep(0.15)  # Lightweight rate-limiting

        except Exception as e:
            print(f"[!] Error on page {page}: {e}")
            continue

    return sorted(list(discovered_links))


# -------------------------------------------------------------
# 4. Processing & Ollama LLM Extraction
# -------------------------------------------------------------
def process_article(session: cffi_requests.Session, url: str, model_name: str = "qwen2.5:3b") -> Optional[FactCheckRecord]:
    try:
        res = session.get(url, timeout=15)
        if res.status_code != 200:
            return None

        soup = BeautifulSoup(res.text, "html.parser")

        # HTML Metadata: Published Date
        pub_tag = (
            soup.find("meta", property="article:published_time")
            or soup.find("time")
            or soup.select_one(".post-details .date, .date, .article-date, .published")
        )
        factcheck_date = (
            pub_tag.get("content") or pub_tag.get("datetime") or pub_tag.get_text(strip=True)
            if pub_tag
            else ""
        )

        # HTML Metadata: Author
        author_tag = (
            soup.find("meta", property="author")
            or soup.select_one(".author-details .author-name, .author, .article-author")
        )
        factchecker = (
            author_tag.get("content") or author_tag.get_text(strip=True)
            if author_tag
            else "Stopfals Team"
        )

        # Article text body
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 35]
        raw_text = "\n".join(paragraphs)

        if not raw_text:
            return None

        prompt = f"""Extract fact-checking data from this article.
Strictly classify the verdict into one of: True, False, Mostly True, Mostly False, Mixture.
Keep the analysis_summary to a maximum of 2 sentences.

Article text:
{raw_text[:2500]}"""

        response = chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a fact-checking dataset parser. Extract verdict, statement, claimant, and analysis into strict JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            format=LLMSemanticFields.model_json_schema(),
            options={"temperature": 0.0, "num_predict": 512},
        )

        semantic_data = LLMSemanticFields.model_validate_json(response.message.content)

        return FactCheckRecord(
            source_country="Moldova",
            news_source="Stopfals.md",
            verdict=semantic_data.verdict,
            statement=semantic_data.statement,
            statement_originator=semantic_data.statement_originator,
            statement_date=semantic_data.statement_date,
            statement_source=semantic_data.statement_source,
            factchecker=factchecker,
            factcheck_date=factcheck_date,
            factcheck_analysis_link=url,
            analysis_text=semantic_data.analysis_summary,
        )

    except Exception as e:
        print(f"[!] Error processing {url}: {e}")
        return None



def build_date_lookup(category_id: int = 4, lang: str = "ro") -> dict:
    base_url = "https://stopfals.md"
    session = cffi_requests.Session(impersonate="chrome124")
    url_to_date = {}

    # 1. Get total page count
    first_page = session.get(f"{base_url}/articles/{category_id}?page=1&lang={lang}").json()
    total_pages = first_page.get("pagination", {}).get("pageCount", 1)

    print(f"[*] Building date map across {total_pages} API pages...")

    # 2. Map every article URL to its real date
    for page in range(1, total_pages + 1):
        res = session.get(f"{base_url}/articles/{category_id}?page={page}&lang={lang}")
        if res.status_code == 200:
            data = res.json()
            # Stopfals returns article objects under 'data' or 'items' or directly as a list
            items = data.get("data") or data.get("articles") or data.get("results") or (data if isinstance(data, list) else [])
            
            for item in items:
                if isinstance(item, dict):
                    # Match URL slug
                    slug = item.get("slug") or item.get("url") or ""
                    full_link = urljoin(base_url, slug if slug.startswith("/") else f"/ro/article/{slug}")
                    
                    # Extract date field
                    real_date = item.get("prettyCreatedAt") or item.get("created_at") or item.get("published_at") or item.get("date") or ""
                    if full_link and real_date:
                        url_to_date[full_link] = real_date
                        
        time.sleep(0.05)

    print(f"[+] Mapped {len(url_to_date)} dates from API.")
    return url_to_date


def patch_jsonl_dates(input_file: str, output_file: str, date_lookup: dict):
    updated = 0
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            link = record.get("factcheck_analysis_link", "")

            # If the date is the broken template, replace it from our lookup table
            if record.get("factcheck_date") == "{article.prettyCreatedAt}" or not record.get("factcheck_date"):
                if link in date_lookup:
                    record["factcheck_date"] = date_lookup[link]
                    updated += 1
                else:
                    # Partial matching by article ID/slug if full URL differs
                    for api_url, d in date_lookup.items():
                        if link.split("-")[-1] in api_url:
                            record["factcheck_date"] = d
                            updated += 1
                            break

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[+] Successfully fixed dates for {updated} records in {output_file}!")

# Example usage:
# lookup = build_date_lookup(category_id=4, lang="ro")
# patch_jsonl_dates("dataset_stopfals-md.jsonl", "dataset_stopfals-md_fixed.jsonl", lookup)

# -------------------------------------------------------------
# 5. Main Execution
# -------------------------------------------------------------
if __name__ == "__main__":
    session = build_http_client()

    # 1. Fetch all links directly from the API endpoint
    links = get_all_stopfals_links(session, category_id=4, lang="ro")
    print(f"\n[+] Successfully gathered {len(links)} article links.\n")

    output_path = "dataset_stopfals-md.jsonl"

    # 2. Extract structured entries and append to JSONL
    for idx, link in enumerate(links, start=1):
        t0 = time.time()
        print(f"[{idx}/{len(links)}] Extracting: {link}")
        record = process_article(session, link, model_name="qwen2.5:3b")

        if record:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
            print(f" -> Saved ({time.time() - t0:.2f}s) | Verdict: {record.verdict}")