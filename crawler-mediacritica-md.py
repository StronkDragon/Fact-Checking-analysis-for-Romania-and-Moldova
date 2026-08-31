import json
import re
import time
from typing import List, Literal, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests
from ollama import chat
from pydantic import BaseModel, Field

# -------------------------------------------------------------
# 1. Pydantic Schemas
# -------------------------------------------------------------
VerdictType = Literal["True", "False", "Mostly True", "Mostly False", "Mixture"]

class FactCheckRecord(BaseModel):
    source_country: str = "Moldova"
    news_source: str = "Mediacritica.md"
    verdict: VerdictType = Field(description="Verdict label: True, False, Mostly True, Mostly False, or Mixture")
    statement: str = Field(description="The core claim, bias, or media narrative being analyzed")
    statement_originator: str = Field(default="Unknown", description="Who made the claim or which outlet published the narrative")
    statement_date: str = Field(default="", description="Date of the analyzed statement/article, or empty string")
    statement_source: str = Field(default="press", description="Medium: press, tv, social_media, web, etc.")
    factchecker: str = Field(default="Mediacritica Team", description="Author or editorial team")
    factcheck_date: str = Field(description="Publication date of the analysis")
    factcheck_analysis_link: str = Field(description="Full article URL")
    analysis_text: str = Field(description="Concise summary of the reasoning")


class LLMSemanticFields(BaseModel):
    verdict: VerdictType = Field(description="Strictly classify into one of: True, False, Mostly True, Mostly False, Mixture")
    statement: str = Field(description="Core claim or media manipulation discussed in the text")
    statement_originator: str = Field(default="Unknown", description="Person or media outlet that originated the claim")
    statement_date: str = Field(default="", description="Date of the claim/event if stated, or empty")
    statement_source: str = Field(default="press", description="Channel: press, tv, social_media, web, etc.")
    analysis_summary: str = Field(description="Max 2 sentences explaining why this verdict was assigned")


# -------------------------------------------------------------
# 2. HTTP Client Configuration
# -------------------------------------------------------------
def build_http_client() -> cffi_requests.Session:
    session = cffi_requests.Session(impersonate="chrome124")
    session.headers.update({
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
    })
    return session


# -------------------------------------------------------------
# 3. Targeted Link Discovery (Exact 6 Feed Articles per Page)
# -------------------------------------------------------------
def get_all_mediacritica_links(
    session: cffi_requests.Session, 
    start_page: int = 1, 
    end_page: int = 51
) -> List[str]:
    base_url = "https://mediacritica.md/categorie/media-in-vizor"
    discovered_links = []
    seen = set()

    for page in range(start_page, end_page + 1):
        target_url = f"{base_url}/page/{page}/" if page > 1 else f"{base_url}/"
        print(f"[*] Crawling page [{page}/{end_page}]: {target_url}")

        try:
            res = session.get(target_url, timeout=15)
            if res.status_code == 404:
                print(f"[+] Reached end of category (HTTP 404 at page {page}). Stopping.")
                break
            if res.status_code != 200:
                print(f"[!] Warning: HTTP {res.status_code} on page {page}")
                continue

            soup = BeautifulSoup(res.text, "html.parser")

            # Remove sidebar, top menus, and footers to eliminate non-feed links
            for unwanted in soup.select("aside, .sidebar, header, nav, footer, .widget, .elementor-location-header, .elementor-location-footer"):
                unwanted.decompose()

            # Target the main article loop cards specifically
            page_links = []
            
            # 1. Target article title headers directly (e.g. h2/h3 a, .entry-title a)
            title_anchors = soup.select(
                "main h2 a, main h3 a, .content h2 a, .content h3 a, "
                "article h2 a, article h3 a, .entry-title a, .post-title a, .card-title a"
            )

            # 2. Fallback to general cards inside main container if selectors differ
            if not title_anchors:
                title_anchors = soup.select("main article a, #primary article a, .site-main article a")

            for a in title_anchors:
                href = a.get("href", "").split("?")[0].rstrip("/")
                if not href:
                    continue
                full_url = urljoin("https://mediacritica.md", href) + "/"
                parsed = urlparse(full_url)

                # Ensure it belongs to the domain and isn't a taxonomy or structural link
                if (
                    parsed.netloc.endswith("mediacritica.md")
                    and not any(x in parsed.path for x in ["/categorie/", "/category/", "/tag/", "/page/", "/author/", "/autor/"])
                    and full_url not in seen
                    and full_url not in page_links
                ):
                    page_links.append(full_url)

            # Take the top 6 feed items for this page
            page_links = page_links[:6]
            print(f"  -> Extracted {len(page_links)} main feed articles from page {page}")

            for link in page_links:
                seen.add(link)
                discovered_links.append(link)

            time.sleep(0.3)

        except Exception as e:
            print(f"[!] Error fetching page {page}: {e}")
            continue

    return discovered_links


# -------------------------------------------------------------
# 4. Publication Date Extraction
# -------------------------------------------------------------
def extract_date(soup: BeautifulSoup) -> str:
    # 1. OpenGraph / Schema metadata
    meta_date = (
        soup.find("meta", property="article:published_time")
        or soup.find("meta", property="og:published_time")
        or soup.find("meta", attrs={"name": "publish-date"})
    )
    if meta_date and meta_date.get("content"):
        return meta_date["content"].strip()

    # 2. HTML5 <time> tag
    time_tag = soup.find("time")
    if time_tag:
        if time_tag.get("datetime"):
            return time_tag["datetime"].strip()
        if time_tag.get_text(strip=True):
            return time_tag.get_text(strip=True)

    # 3. Class-based dates
    class_date = soup.select_one(".entry-date, .post-date, .date, .published, .article-date")
    if class_date:
        return class_date.get_text(strip=True)

    return ""


# -------------------------------------------------------------
# 5. Article Processing & LLM Semantic Extraction
# -------------------------------------------------------------
def process_article(session: cffi_requests.Session, url: str, model_name: str = "qwen2.5:3b") -> Optional[FactCheckRecord]:
    try:
        res = session.get(url, timeout=15)
        if res.status_code != 200:
            return None

        soup = BeautifulSoup(res.text, "html.parser")
        factcheck_date = extract_date(soup)

        author_tag = (
            soup.find("meta", attrs={"name": "author"})
            or soup.find("meta", property="article:author")
            or soup.select_one(".author-name, .author, .byline, .entry-author")
        )
        factchecker = (
            author_tag.get("content") or author_tag.get_text(strip=True)
            if author_tag
            else "Mediacritica Team"
        )

        content_elem = soup.select_one("article, .entry-content, .post-content, main") or soup
        paragraphs = [
            p.get_text(strip=True)
            for p in content_elem.find_all("p")
            if len(p.get_text(strip=True)) > 35
        ]
        raw_text = "\n".join(paragraphs)

        if not raw_text:
            return None

        prompt = f"""Extract fact-checking and media manipulation analysis data from this article.
Strictly classify the verdict into one of: True, False, Mostly True, Mostly False, Mixture.
Keep analysis_summary to a maximum of 2 sentences.

Article text:
{raw_text[:12000]}"""

        response = chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a structured dataset parser. Extract verdict, statement/narrative, claimant, and analysis into strict JSON.",
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
            source_country="Moldova",
            news_source="Mediacritica.md",
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


# -------------------------------------------------------------
# 6. Main Execution
# -------------------------------------------------------------
if __name__ == "__main__":
    session = build_http_client()
    
    # Discovery loop from page 1 to 51
    links = get_all_mediacritica_links(session, start_page=1, end_page=51)
    print(f"\n[+] Total targeted feed articles captured: {len(links)} (Expected ~{51 * 6})\n")

    output_path = "dataset_mediacritica-md.jsonl"

    for idx, link in enumerate(links, start=1):
        t0 = time.time()
        print(f"[{idx}/{len(links)}] Parsing: {link}")
        record = process_article(session, link, model_name="qwen2.5:3b")

        if record:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
            print(f" -> Saved ({time.time() - t0:.2f}s) | Date: '{record.factcheck_date}' | Verdict: {record.verdict}")