import json
import re
import time
from typing import List, Literal, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from curl_cffi import requests
from ollama import chat
from pydantic import BaseModel, Field

BASE_URL = "https://brodhub.eu"
OUTPUT_FILE = "dataset_brodhub_eu_updated.jsonl"

REQUEST_TIMEOUT = 25
RETRY_HTTP_CODES = {403, 429, 500, 502, 503, 504}

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

# -------------------------------------------------------------
# 1. Unified Dataset Schema & Taxonomy
# -------------------------------------------------------------
VerdictType = Literal["True", "False", "Mostly True", "Mostly False", "Mixture"]

class FactCheckRecord(BaseModel):
    source_country: str = "Romania / Bulgaria"
    news_source: str = "brodhub.eu"
    verdict: VerdictType = Field(description="Normalized verdict")
    statement: str = Field(description="The exact statement or claim being analyzed")
    statement_originator: str = Field(default="Unknown", description="Who made the statement")
    statement_date: str = Field(default="", description="Date of original statement")
    statement_source: str = Field(default="social_media", description="Medium / channel")
    factchecker: str = Field(default="BROD Hub", description="Author or editorial team")
    factcheck_date: str = Field(description="Publication date of fact check")
    factcheck_analysis_link: str = Field(description="Full article URL")
    analysis_text: str = Field(description="Summary of analysis")
    extraction_method: Literal["claim_review", "ollama_fallback"] = Field(description="Source of structured metadata")


class LLMSemanticFields(BaseModel):
    verdict: VerdictType = Field(description="Strictly classify into: True, False, Mostly True, Mostly False, Mixture")
    statement: str = Field(description="The core claim verified in the article")
    statement_originator: str = Field(default="Unknown", description="Person/entity originating the claim")
    statement_date: str = Field(default="", description="Date of claim if stated")
    statement_source: str = Field(default="social_media", description="Medium: social_media, tv, press, web, etc.")
    analysis_summary: str = Field(description="Max 2 sentences explaining why verdict was assigned")


# -------------------------------------------------------------
# 2. HTTP Client Configuration
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# 3. Solr Search Link Discovery
# -------------------------------------------------------------
def get_all_factcheck_links(client, total_pages=43):
    all_links = []
    seen = set()

    for page_num in range(1, total_pages + 1):
        first_index = (page_num - 1) * 30
        page_url = SEARCH_URL_TEMPLATE.format(first_index=first_index)
        print(f"[*] Fetching links (Page {page_num}/{total_pages} - firstIndex={first_index})...")

        resp = get_with_retries(client, page_url)
        if not resp or resp.status_code != 200:
            print(f"[!] Page {page_num} could not be loaded.")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        page_links = 0

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if "/factchecks/" in href and not href.startswith("#"):
                full_url = urljoin(BASE_URL, href)
                if full_url not in seen:
                    seen.add(full_url)
                    all_links.append(full_url)
                    page_links += 1

        print(f"  -> Found {page_links} new articles (Total unique: {len(all_links)})")
        time.sleep(0.3)

    return all_links


# -------------------------------------------------------------
# 4. JSON-LD Extraction & Fallback Normalization
# -------------------------------------------------------------
def find_claim_review_in_json_ld(soup: BeautifulSoup) -> Optional[dict]:
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


def normalize_verdict(raw_verdict: str) -> VerdictType:
    v = raw_verdict.strip().lower()
    if any(k in v for k in ["fals", "false", "fake", "neadevărat"]):
        return "False"
    if any(k in v for k in ["adevărat", "adevarat", "true", "corect"]):
        return "True"
    if any(k in v for k in ["parțial", "partial", "mostly false", "trunchiat", "înșelător", "inselator"]):
        return "Mostly False"
    if any(k in v for k in ["mostly true", "predominant adevărat"]):
        return "Mostly True"
    return "Mixture"


def extract_record_from_claim(claim_json: dict, article_url: str, raw_text: str) -> FactCheckRecord:
    item_rev = claim_json.get("itemReviewed", {})
    rating = claim_json.get("reviewRating", {})

    verdict_raw = rating.get("alternateName") or rating.get("ratingValue") or "Mixture"
    verdict = normalize_verdict(str(verdict_raw))

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

    return FactCheckRecord(
        source_country="Romania / Bulgaria",
        news_source="brodhub.eu",
        verdict=verdict,
        statement=claim_json.get("claimReviewed", ""),
        statement_originator=statement_originator,
        statement_date=item_rev.get("datePublished", "") if isinstance(item_rev, dict) else "",
        statement_source=statement_source,
        factchecker=factchecker,
        factcheck_date=claim_json.get("datePublished", ""),
        factcheck_analysis_link=article_url,
        analysis_text=raw_text[:600].strip(),
        extraction_method="claim_review",
    )


# -------------------------------------------------------------
# 5. Fallback Parser (Ollama)
# -------------------------------------------------------------
def parse_with_ollama(raw_text: str, article_url: str, soup: BeautifulSoup, model_name: str = "qwen2.5:3b") -> Optional[FactCheckRecord]:
    meta_date = (
        soup.find("meta", property="article:published_time")
        or soup.find("meta", property="og:published_time")
        or soup.find("time")
    )
    factcheck_date = (
        meta_date.get("content") or meta_date.get("datetime") or meta_date.get_text(strip=True)
        if meta_date else ""
    )

    prompt = f"""Extract fact-checking data from this article.
Strictly classify the verdict into one of: True, False, Mostly True, Mostly False, Mixture.
Keep analysis_summary to a maximum of 2 sentences.

Article text:
{raw_text[:12000]}"""

    try:
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
            options={
                "temperature": 0.0,
                "num_ctx": 8192,
                "num_predict": 512,
            },
        )
        semantic_data = LLMSemanticFields.model_validate_json(response.message.content)

        return FactCheckRecord(
            source_country="Romania / Bulgaria",
            news_source="brodhub.eu",
            verdict=semantic_data.verdict,
            statement=semantic_data.statement,
            statement_originator=semantic_data.statement_originator,
            statement_date=semantic_data.statement_date,
            statement_source=semantic_data.statement_source,
            factchecker="BROD Hub",
            factcheck_date=factcheck_date,
            factcheck_analysis_link=article_url,
            analysis_text=semantic_data.analysis_summary,
            extraction_method="ollama_fallback",
        )
    except Exception as e:
        print(f"[!] Ollama fallback failed for {article_url}: {e}")
        return None


# -------------------------------------------------------------
# 6. Hybrid Article Handler
# -------------------------------------------------------------
def process_article(client: requests.Session, url: str) -> Optional[FactCheckRecord]:
    resp = get_with_retries(client, url)
    if not resp or resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Clean non-content elements
    content_div = soup.select_one(".entry-content, .post-content, #xwikicontent, article") or soup
    for junk in content_div.select("script, style, .sharedaddy, .ads, nav, footer, header"):
        junk.decompose()

    paragraphs = [p.get_text(strip=True) for p in content_div.find_all("p") if len(p.get_text(strip=True)) > 30]
    raw_text = "\n".join(paragraphs)

    # 1. Prefer native ClaimReview metadata
    claim_data = find_claim_review_in_json_ld(soup)
    if claim_data and claim_data.get("claimReviewed"):
        return extract_record_from_claim(claim_data, url, raw_text)

    # 2. Fall back to Ollama if ClaimReview is missing or empty
    if raw_text:
        return parse_with_ollama(raw_text, url, soup)

    return None


# -------------------------------------------------------------
# 7. Main Execution Pipeline
# -------------------------------------------------------------
def main():
    client = build_http_client()
    links = get_all_factcheck_links(client, total_pages=43)
    print(f"\n[+] Collected {len(links)} links. Starting processing...\n")

    for idx, link in enumerate(links, 1):
        t0 = time.time()
        print(f"[{idx}/{len(links)}] Fetching: {link}")
        record = process_article(client, link)

        if record:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
            print(f" -> Saved via [{record.extraction_method}] ({time.time() - t0:.2f}s) | Verdict: {record.verdict}")
        else:
            print(f" -> Failed to process article.")

        time.sleep(0.15)


if __name__ == "__main__":
    main()