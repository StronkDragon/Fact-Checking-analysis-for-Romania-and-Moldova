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
    news_source: str = "Ziarul de Gardă (ZdG)"
    verdict: VerdictType = Field(description="Strict verdict label")
    statement: str = Field(description="The central claim being analyzed")
    statement_originator: str = Field(default="Unknown", description="Who made the claim")
    statement_date: str = Field(default="", description="Date when claim was made")
    statement_source: str = Field(default="social_media", description="Channel/medium of claim")
    factchecker: str = Field(default="ZdG Team", description="Author or editorial team")
    factcheck_date: str = Field(description="Article publication date")
    factcheck_analysis_link: str = Field(description="URL of the fact-check")
    analysis_text: str = Field(description="Summary of the reasoning")


class LLMSemanticFields(BaseModel):
    verdict: VerdictType = Field(description="True, False, Mostly True, Mostly False, or Mixture")
    statement: str = Field(description="Core assertion or narrative verified in the text")
    statement_originator: str = Field(default="Unknown", description="Claim originator")
    statement_date: str = Field(default="", description="Date of claim if stated")
    statement_source: str = Field(default="social_media", description="Source channel")
    analysis_summary: str = Field(description="Max 2 sentences explaining why the verdict was assigned")


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
# 3. Paginated Link Discovery (/tag/fals/page/N/)
# -------------------------------------------------------------
def get_all_zdg_links(session: cffi_requests.Session, base_tag_url: str = "https://www.zdg.md/tag/fals") -> List[str]:
    all_links = set()
    page = 1

    while True:
        target_url = f"{base_tag_url}/page/{page}/" if page > 1 else f"{base_tag_url}/"
        print(f"[*] Crawling: {target_url}")

        res = session.get(target_url, timeout=15)
        if res.status_code == 404:
            print(f"[+] Reached last page (HTTP 404 at page {page}). Discovery complete.")
            break
        elif res.status_code != 200:
            print(f"[!] Warning: HTTP {res.status_code} on page {page}. Stopping.")
            break

        soup = BeautifulSoup(res.text, "html.parser")
        page_links = set()

        # Find post anchors in listings
        for a in soup.find_all("a", href=True):
            href = a["href"].split("?")[0].rstrip("/")
            full_url = urljoin("https://www.zdg.md", href) + "/"
            parsed = urlparse(full_url)

            if (
                parsed.netloc.endswith("zdg.md")
                and not any(x in parsed.path for x in ["/tag/", "/category/", "/page/", "/autor/", "/despre-noi/", "/contact/"])
                and len(parsed.path.strip("/").split("/")) >= 1
                and any(keyword in parsed.path for keyword in ["/stiri/", "/importanta/", "/politic/", "/social/"])
            ):
                page_links.add(full_url)

        # Fallback heuristic: look inside article/post headers
        if not page_links:
            for article in soup.find_all("article"):
                a_tag = article.find("a", href=True)
                if a_tag:
                    page_links.add(urljoin("https://www.zdg.md", a_tag["href"].split("?")[0]))

        if not page_links or page_links.issubset(all_links):
            print(f"[+] No new links found on page {page}. Completed pagination.")
            break

        all_links.update(page_links)
        print(f"  -> Found {len(page_links)} links on page {page} (Total unique: {len(all_links)})")
        page += 1
        time.sleep(0.3)

    return sorted(list(all_links))


# -------------------------------------------------------------
# 4. Article Processing & LLM Semantic Extraction
# -------------------------------------------------------------
def extract_date(soup: BeautifulSoup) -> str:
    meta_date = (
        soup.find("meta", property="article:published_time")
        or soup.find("meta", property="og:published_time")
        or soup.find("meta", attrs={"name": "publish-date"})
    )
    if meta_date and meta_date.get("content"):
        return meta_date["content"].strip()

    time_tag = soup.find("time")
    if time_tag:
        return time_tag.get("datetime") or time_tag.get_text(strip=True)

    class_date = soup.select_one(".entry-date, .post-date, .date, .article-date")
    if class_date:
        return class_date.get_text(strip=True)

    return ""


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
            else "ZdG Team"
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

        # Increased to ~12000 chars (fits comfortably inside num_ctx=8192)
        prompt = f"""Extract fact-checking data from this article.
Strictly classify the verdict into one of: True, False, Mostly True, Mostly False, Mixture.
Keep analysis_summary to a maximum of 2 sentences.

Article text:
{raw_text[:12000]}"""

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
                "num_ctx": 8192,     # Allocates ~8k token context window in VRAM
                "num_predict": 512,
            },
        )

        semantic_data = LLMSemanticFields.model_validate_json(response.message.content)

        return FactCheckRecord(
            source_country="Moldova",
            news_source="Ziarul de Gardă (ZdG)",
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
    links = get_all_zdg_links(session, base_tag_url="https://www.zdg.md/tag/fals")
    print(f"\n[+] Total articles found: {len(links)}\n")

    output_path = "dataset_zdg-md.jsonl"

    for idx, link in enumerate(links, start=1):
        t0 = time.time()
        print(f"[{idx}/{len(links)}] Parsing: {link}")
        record = process_article(session, link, model_name="qwen2.5:3b")

        if record:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
            print(f" -> Saved ({time.time() - t0:.2f}s) | Date: '{record.factcheck_date}' | Verdict: {record.verdict}")