import json
import re
import time
from typing import List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests
from ollama import chat
from pydantic import BaseModel, Field


# -------------------------------------------------------------
# 1. Pydantic Schemas
# -------------------------------------------------------------
class FactCheckRecord(BaseModel):
    source_country: str = "Moldova"
    news_source: str = "Veridica.md"
    verdict: str = Field(description="Verdict label (e.g., fals, dezinformare, manipulare)")
    statement: str = Field(description="The exact statement/narrative being fact-checked")
    statement_originator: str = Field(description="Who made the statement, or 'Unknown'")
    statement_date: str = Field(default="", description="Date of the statement, or empty string")
    statement_source: str = Field(default="social_media", description="Medium: social_media, tv, press, web, etc.")
    factchecker: str = Field(default="Veridica Team", description="Author or editorial team")
    factcheck_date: str = Field(description="Publication date of the fact check")
    factcheck_analysis_link: str = Field(description="Full article URL")
    analysis_text: str = Field(description="Concise summary of the fact-check reasoning")


class LLMSemanticFields(BaseModel):
    verdict: str = Field(description="Verdict label in lowercase (e.g., fals, dezinformare, manipulare)")
    statement: str = Field(description="The exact claim or narrative verified in the text")
    statement_originator: str = Field(default="Unknown", description="Person/entity that originated the claim")
    statement_date: str = Field(default="", description="Date of the statement if mentioned, or empty")
    statement_source: str = Field(default="social_media", description="Channel: social_media, tv, press, etc.")
    analysis_summary: str = Field(description="Max 2 sentences explaining why the claim is false/true")

# -------------------------------------------------------------
# 2. HTTP Client with Chrome Fingerprint Impersonation
# -------------------------------------------------------------
def build_http_client() -> cffi_requests.Session:
    session = cffi_requests.Session(impersonate="chrome124")
    session.headers.update({
        "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.veridica.md/",
    })
    return session


# -------------------------------------------------------------
# 3. Crawler: Target `/republica-moldova/{slug}`
# -------------------------------------------------------------
def get_veridica_links(session: cffi_requests.Session, max_pages: int = 2) -> List[str]:
    base_url = "https://www.veridica.md"
    found_links = set()

    for page in range(1, max_pages + 1):
        target_url = f"{base_url}/republica-moldova?page={page}" if page > 1 else f"{base_url}/republica-moldova"
        print(f"[*] Crawling listing: {target_url}")

        try:
            res = session.get(target_url, timeout=15)
            if res.status_code != 200:
                print(f"[!] HTTP {res.status_code} on {target_url}")
                continue

            soup = BeautifulSoup(res.text, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                full_url = urljoin(base_url, href)
                parsed = urlparse(full_url)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

                # Match paths: https://www.veridica.md/republica-moldova/{article-title}
                if re.match(r"^https://www\.veridica\.md/republica-moldova/[a-zA-Z0-9\-]+$", clean_url):
                    found_links.add(clean_url)

        except Exception as e:
            print(f"[!] Error on page {page}: {e}")

    return sorted(list(found_links))


# -------------------------------------------------------------
# 4. Processing & Ollama Extraction
# -------------------------------------------------------------
def process_article(session: cffi_requests.Session, url: str, model_name: str = "qwen2.5:3b") -> Optional[FactCheckRecord]:
    try:
        res = session.get(url, timeout=15)
        if res.status_code != 200:
            return None

        soup = BeautifulSoup(res.text, "html.parser")

        # HTML Metadata
        pub_tag = soup.find("meta", property="article:published_time")
        factcheck_date = pub_tag["content"] if pub_tag else ""

        author_tag = soup.find("meta", property="author") or soup.select_one(".author, [class*='author']")
        factchecker = author_tag.get_text(strip=True) if author_tag else "Veridica Team"

        # Article text body
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
        raw_text = "\n".join(paragraphs)

        if not raw_text:
            return None

        # Slicing prompt to ~2000 chars keeps per-article inference down to ~3-6s on GTX 1660 Ti

        prompt = f"""Extract fact-checking data from this article. 
        Keep the analysis_summary strictly to a maximum of 2 sentences:

        {raw_text[:2500]}"""

        response = chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a fact-checking dataset parser. Extract verdict, statement, claimant, and analysis summary into strict JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            format=LLMSemanticFields.model_json_schema(),
            options={"temperature": 0.0, "num_predict": 1024},
        )

        semantic_data = LLMSemanticFields.model_validate_json(response.message.content)

        return FactCheckRecord(
            source_country="Moldova",
            news_source="Veridica.md",
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
    links = get_veridica_links(session, max_pages=2)
    print(f"\n[+] Total articles discovered: {len(links)}\n")

    output_path = "dataset_veridica-md.jsonl"

    for idx, link in enumerate(links, start=1):
        t0 = time.time()
        print(f"[{idx}/{len(links)}] Extracting: {link}")
        record = process_article(session, link, model_name="qwen2.5:3b")

        if record:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
            print(f" -> Saved ({time.time() - t0:.2f}s) | Verdict: {record.verdict}")