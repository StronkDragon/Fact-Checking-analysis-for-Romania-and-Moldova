# Romanian and Moldova Fact-Checking Data Crawlers

Python-based web crawlers designed to collect and structure fact-checking articles from online sources. The project combines web scraping, API-based article discovery, HTML parsing, and local Large Language Model (LLM) extraction to build a structured dataset for further data analysis and potential AI applications.

## Project Overview

The goal of this project is to collect fact-checking articles from Romanian-language online sources and transform unstructured article content into structured records. Separate crawler scripts were developed for different websites, adapting the collection and extraction process to each source.

The following example is a crawler coded for [StopFals.md](https://stopfals.md/), a fact-checking platform based in Moldova.

## Key Features

- **Article discovery:** Retrieves article links through the website's paginated article endpoint.
- **Web scraping:** Extracts article text, publication dates, and author information from HTML pages.
- **LLM-powered information extraction:** Uses a locally running LLM through Ollama to identify claims, extract verdicts, and summarize fact-checking analyses.
- **Structured data validation:** Uses Pydantic models to validate extracted records against a predefined schema.
- **Data cleaning and enrichment:** Includes utilities for recovering publication dates from API metadata and correcting incomplete dataset records.
- **Dataset generation:** Saves structured records in JSON Lines (`.jsonl`) format for subsequent processing and analysis.

## Tech Stack

- **Python** — Core implementation
- **BeautifulSoup4** — HTML parsing
- **curl-cffi** — HTTP requests
- **Ollama** — Local LLM inference
- **Qwen 2.5 (3B)** — Information extraction from article text
- **Pydantic** — Data validation and structured output
- **JSON / JSONL** — Data serialization and storage

## Extracted Data

Each processed article is represented by a structured record containing fields such as:

| Field | Description |
|---|---|
| `source_country` | Country associated with the source |
| `news_source` | Name of the fact-checking publication |
| `verdict` | Classification of the claim |
| `statement` | Claim being fact-checked |
| `statement_originator` | Person or entity associated with the claim |
| `statement_date` | Date of the original claim, when available |
| `statement_source` | Medium where the claim originated |
| `factchecker` | Author or editorial team |
| `factcheck_date` | Publication date of the fact-check |
| `factcheck_analysis_link` | URL of the original article |
| `analysis_text` | Concise summary of the fact-checking reasoning |

The verdict taxonomy used by the current crawler is `True`, `False`, `Mostly True`, `Mostly False`, and `Mixture`.

## How It Works

1. **Discover articles:** Queries the source's paginated article endpoint and collects unique article URLs.
2. **Retrieve article content:** Downloads each article and parses its HTML using BeautifulSoup.
3. **Extract metadata:** Collects available publication dates, author information, and article text.
4. **Process with an LLM:** Sends the extracted text to a local Qwen 2.5 model through Ollama to identify the claim, verdict, and supporting analysis.
5. **Validate the output:** Uses Pydantic to validate the LLM-generated structured data.
6. **Save the dataset:** Appends each successfully processed record to a JSONL file.
7. **Correct metadata:** Provides additional utilities to recover missing or incorrectly extracted publication dates using the source's API metadata.

## Getting Started

### Prerequisites

- Python 3.10 or later
- [Ollama](https://ollama.com/) installed and running
- The `qwen2.5:3b` model downloaded locally

The script discovers article URLs, processes each article, and saves successfully extracted records to:

```text
dataset_stopfals-md.jsonl
```

Each line in the output file contains one JSON object representing a fact-checking article.

## Example Output

A dataset record follows this general structure:

```json
{
  "source_country": "Moldova",
  "news_source": "Stopfals.md",
  "verdict": "False",
  "statement": "Example claim being fact-checked",
  "statement_originator": "Unknown",
  "statement_date": "",
  "statement_source": "social_media",
  "factchecker": "Stopfals Team",
  "factcheck_date": "2025-01-15",
  "factcheck_analysis_link": "https://stopfals.md/example-article",
  "analysis_text": "A concise summary of the fact-checking analysis."
}
```

*The values above are illustrative and do not represent an actual extracted article.*

## Potential Applications

The resulting dataset can support further work in:

- Exploratory data analysis and visualization
- Fact-checking claim classification
- Natural Language Processing (NLP)
- Analysis of misinformation and recurring narratives
- Development and evaluation of AI-assisted fact-checking tools

These are potential applications of the collected data, not claims that all these downstream tasks have already been implemented.

## Limitations

- LLM-extracted claims and verdicts may require manual verification against the original articles.
- Extraction quality depends on the structure and content of each source website.
- Publication metadata may be incomplete or inconsistently formatted.
- The current implementation uses a predefined verdict taxonomy, which may not cover every fact-checking methodology.

## Responsible Use

This project is intended for research, educational, and data-analysis purposes. Please respect the source websites' terms of service, robots.txt directives where applicable, copyright, and rate limits. Preserve links to original publications and verify extracted information before using it in further research.

## Author

**Marian-Alexandru Panait**

Computer Science and Engineering Student | AI & Data Analysis Enthusiast

