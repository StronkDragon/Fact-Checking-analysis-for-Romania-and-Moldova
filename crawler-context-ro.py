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
    source_country: str = "Romania"
    news_source: str = "Context.ro"
    verdict: VerdictType = Field(description="Verdict label: True, False, Mostly True, Mostly False, or Mixture")
    statement: str = Field(description="The exact statement or claim being analyzed")
    statement_originator: str = Field(description="Who made the statement, or 'Unknown'")
    statement_date: str = Field(default="", description="Date of the statement, or empty string")
    statement_source: str = Field(default="social_media", description="Medium: social_media, tv, press, web, etc.")
    factchecker: str = Field(default="Context.ro Team", description="Author or editorial team")
    factcheck_date: str = Field(description="Publication date of the article")
    factcheck_analysis_link: str = Field(description="Full article URL")
    analysis_text: str = Field(description="Concise summary of the reasoning")


class LLMSemanticFields(BaseModel):
    verdict: VerdictType = Field(description="Strictly classify into one of: True, False, Mostly True, Mostly False, Mixture")
    statement: str = Field(description="The core claim, event, or narrative explained in the text")
    statement_originator: str = Field(default="Unknown", description="Person/entity that originated the claim/subject")
    statement_date: str = Field(default="", description="Date of the statement/event if mentioned, or empty")
    statement_source: str = Field(default="web", description="Channel: social_media, tv, press, web, etc.")
    analysis_summary: str = Field(description="Max 2 sentences explaining why this verdict was assigned")


# -------------------------------------------------------------
# 2. HTTP Client Configuration
# -------------------------------------------------------------
def build_http_client() -> cffi_requests.Session:
    session = cffi_requests.Session(impersonate="chrome124")
    session.headers.update({
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
    })
    return session


# -------------------------------------------------------------
# 3. Link Discovery from Category Page
# -------------------------------------------------------------
def get_all_context_links(session: cffi_requests.Session, category_url: str = "https://context.ro/category/explicative-context/") -> List[str]:
    print(f"[*] Fetching category page: {category_url}")
    res = session.get(category_url, timeout=20)
    if res.status_code != 200:
        print(f"[!] Failed to fetch category page: HTTP {res.status_code}")
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    discovered_links = set()

    # Match all article links within standard WordPress structures
    for a in soup.find_all("a", href=True):
        href = a["href"].split("?")[0].rstrip("/")
        full_url = urljoin("https://context.ro", href) + "/"
        parsed = urlparse(full_url)

        # Exclude taxonomy, static assets, and root category URLs
        if (
            parsed.netloc.endswith("context.ro")
            and not any(x in parsed.path for x in ["/category/", "/tag/", "/author/", "/page/", "/despre-noi/", "/contact/"])
            and parsed.path.strip("/") != ""
            and len(parsed.path.strip("/").split("/")) == 1  # Standard WP post slug structure: /slug/
        ):
            discovered_links.add(full_url)

    return sorted(list(discovered_links))


# -------------------------------------------------------------
# 4. Processing & LLM Semantic Extraction
# -------------------------------------------------------------
def extract_publication_date(soup: BeautifulSoup) -> str:
    """Directly extracts and sanitizes the published date from HTML metadata."""
    # 1. OpenGraph / Schema metadata
    meta_date = (
        soup.find("meta", property="article:published_time")
        or soup.find("meta", property="og:published_time")
        or soup.find("meta", attrs={"name": "publish-date"})
    )
    if meta_date and meta_date.get("content"):
        return meta_date["content"].strip()

    # 2. HTML5 <time> elements
    time_tag = soup.find("time")
    if time_tag:
        if time_tag.get("datetime"):
            return time_tag["datetime"].strip()
        if time_tag.get_text(strip=True):
            return time_tag.get_text(strip=True)

    # 3. Common WordPress date classes
    class_date = soup.select_one(".entry-date, .published, .post-date, .date")
    if class_date:
        return class_date.get_text(strip=True)

    return ""


def process_article(session: cffi_requests.Session, url: str, model_name: str = "qwen2.5:3b") -> Optional[FactCheckRecord]:
    try:
        res = session.get(url, timeout=15)
        if res.status_code != 200:
            return None

        soup = BeautifulSoup(res.text, "html.parser")

        # Extract real publication date from HTML (bypasses LLM hallucinations)
        factcheck_date = extract_publication_date(soup)

        # Extract author
        author_tag = (
            soup.find("meta", attrs={"name": "author"})
            or soup.find("meta", property="article:author")
            or soup.select_one(".author-name, .author, .byline, .entry-author")
        )
        factchecker = (
            author_tag.get("content") or author_tag.get_text(strip=True)
            if author_tag
            else "Context.ro Team"
        )

        # Extract article body text
        content_container = soup.select_one("article, .entry-content, .post-content, main") or soup
        paragraphs = [
            p.get_text(strip=True)
            for p in content_container.find_all("p")
            if len(p.get_text(strip=True)) > 40
        ]
        raw_text = "\n".join(paragraphs)

        if not raw_text:
            return None

        prompt = f"""Extract fact-checking data from this article.
Strictly classify the verdict into one of: True, False, Mostly True, Mostly False, Mixture.
Keep the analysis_summary to a maximum of 2 sentences.

Article text:
{raw_text[:3000]}"""

        response = chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a structured data extractor. Classify the main claim and summarize findings into JSON according to the schema.",
                },
                {"role": "user", "content": prompt},
            ],
            format=LLMSemanticFields.model_json_schema(),
            options={"temperature": 0.0, "num_predict": 512},
        )

        semantic_data = LLMSemanticFields.model_validate_json(response.message.content)

        return FactCheckRecord(
            source_country="Romania",
            news_source="Context.ro",
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
# 5. Main Execution
# -------------------------------------------------------------
if __name__ == "__main__":
    session = build_http_client()

    target_category = "https://context.ro/category/explicative-context/"
    links = get_all_context_links(session, target_category)
    print(f"\n[+] Gathered {len(links)} article links from Context.ro.\n")

    output_path = "dataset_context-ro.jsonl"

    for idx, link in enumerate(links, start=1):
        t0 = time.time()
        print(f"[{idx}/{len(links)}] Extracting: {link}")
        record = process_article(session, link, model_name="qwen2.5:3b")

        if record:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
            print(f" -> Saved ({time.time() - t0:.2f}s) | Date: '{record.factcheck_date}' | Verdict: {record.verdict}")