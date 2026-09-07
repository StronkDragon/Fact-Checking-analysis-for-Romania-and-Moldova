#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline ETL pentru unificarea tuturor seturilor de date de fact-checking din Republica Moldova.
Procesează sursele:
  - Stopfals.md (dataset_stopfals-md_clean.jsonl / dataset_stopfals-md.jsonl)
  - Ziarul de Gardă - ZdG.md (dataset_zdg-md.jsonl)
  - Mediacritica.md (dataset_mediacritica-md.jsonl)
  - Veridica.md (dataset_veridica-md.jsonl)

Include:
  1. Schemă Canonică Pydantic (Golden Schema)
  2. Normalizare exhaustivă a verdictelor (taxonomie standard)
  3. Adaptoare modulare pentru fiecare format din Moldova
  4. Extragere euristică + suport LLM via Ollama (qwen2.5:3b) cu token limit crescut (num_predict: 2048)
     și Structured JSON Schema pentru eliminarea erorilor de parsare (Unterminated string)
  5. Deduplicare inteligentă pe baza URL-ului canonic și a amprentei afirmațiilor
  6. Export în JSONL, JSON și opțional Parquet
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse

try:
    from pydantic import BaseModel, Field
except ImportError:
    print("[EROARE] Pydantic nu este instalat. Rulează: pip install pydantic")
    sys.exit(1)

# Suport opțional pentru Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Suport opțional pentru Parquet
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Configurare logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("UnifyMoldovaDatasets")


# =====================================================================
# 1. SCHEMA CANONICĂ (Golden Schema)
# =====================================================================

class VerdictEnum(str, Enum):
    TRUE = "TRUE"
    MOSTLY_TRUE = "MOSTLY_TRUE"
    MIXTURE = "MIXTURE"
    MOSTLY_FALSE = "MOSTLY_FALSE"
    FALSE = "FALSE"
    UNPROVEN = "UNPROVEN"
    OTHER = "OTHER"


class CanonicalFactCheck(BaseModel):
    """
    Schema unificată pentru toate articolele și revendicările de fact-checking.
    """
    id: str = Field(description="ID unic generat pe baza hash-ului URL-ului sau conținutului")
    source_country: str = Field(default="Moldova", description="Țara vizată")
    source_dataset: str = Field(description="Numele platformei sursă (ex: stopfals.md, zdg.md, mediacritica.md, veridica.md)")
    url: str = Field(description="Linkul canonic către analiza de fact-checking")
    title: str = Field(description="Titlul articolului sau al investigației")
    claim: str = Field(description="Afirmația / teza verificată")
    claimant: Optional[str] = Field(default=None, description="Persoana sau entitatea care a făcut afirmația (sau 'Necunoscut' / 'Surse Multiple')")
    date_published: Optional[str] = Field(default=None, description="Data publicării analizei de fact-checking")
    date_claim: Optional[str] = Field(default=None, description="Data la care s-a făcut afirmația inițială")
    claim_source: Optional[str] = Field(default=None, description="Canalul prin care a fost propagată afirmația (social_media, tv, press, web, telegram, tiktok etc.)")
    original_verdict: str = Field(description="Verdictul brut extras direct din sursă")
    normalized_verdict: VerdictEnum = Field(description="Verdictul mapat la taxonomia standard")
    factchecker: Optional[str] = Field(default=None, description="Numele autorului sau organizației de fact-checking")
    explanation_body: str = Field(description="Textul analizei, argumentarea sau rezumatul detaliat")
    tags: List[str] = Field(default_factory=list, description="Etichete tematice, categorii sau narațiuni asociate")
    extraction_method: str = Field(default="direct_mapping", description="Metoda utilizată pentru extragere")
    extra_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadate adiționale specifice sursei originale")


class LLMExtractionSchema(BaseModel):
    """
    Schema strictă de validare pentru răspunsul structurat Ollama.
    """
    claim: str = Field(description="The exact claim or narrative verified in the text")
    claimant: str = Field(default="Unknown", description="Person/entity that originated the claim")
    verdict: str = Field(description="Strict verdict: True, False, Mostly True, Mostly False, or Mixture")
    analysis_summary: str = Field(description="Max 2 sentences explaining why the verdict was assigned")


# =====================================================================
# 2. NORMALIZAREA VERDICTELOR
# =====================================================================

VERDICT_MAPPINGS: Dict[str, Dict[str, VerdictEnum]] = {
    "stopfals.md": {
        "false": VerdictEnum.FALSE,
        "fals": VerdictEnum.FALSE,
        "fals/dezinformare": VerdictEnum.FALSE,
        "dezinformare": VerdictEnum.FALSE,
        "manipulare": VerdictEnum.MOSTLY_FALSE,
        "mostly false": VerdictEnum.MOSTLY_FALSE,
        "parțial fals": VerdictEnum.MOSTLY_FALSE,
        "partial fals": VerdictEnum.MOSTLY_FALSE,
        "true": VerdictEnum.TRUE,
        "adevărat": VerdictEnum.TRUE,
        "adevarat": VerdictEnum.TRUE,
        "mostly true": VerdictEnum.MOSTLY_TRUE,
        "parțial adevărat": VerdictEnum.MOSTLY_TRUE,
        "partial adevarat": VerdictEnum.MOSTLY_TRUE,
        "mixture": VerdictEnum.MIXTURE,
        "amestec": VerdictEnum.MIXTURE,
        "satiră": VerdictEnum.OTHER,
        "satira": VerdictEnum.OTHER,
        "fără verdict": VerdictEnum.OTHER,
    },
    "zdg.md": {
        "false": VerdictEnum.FALSE,
        "fals": VerdictEnum.FALSE,
        "dezinformare": VerdictEnum.FALSE,
        "manipulare": VerdictEnum.MOSTLY_FALSE,
        "mostly false": VerdictEnum.MOSTLY_FALSE,
        "parțial fals": VerdictEnum.MOSTLY_FALSE,
        "true": VerdictEnum.TRUE,
        "adevărat": VerdictEnum.TRUE,
        "adevarat": VerdictEnum.TRUE,
        "mostly true": VerdictEnum.MOSTLY_TRUE,
        "parțial adevărat": VerdictEnum.MOSTLY_TRUE,
        "mixture": VerdictEnum.MIXTURE,
        "amestec": VerdictEnum.MIXTURE,
    },
    "mediacritica.md": {
        "false": VerdictEnum.FALSE,
        "fals": VerdictEnum.FALSE,
        "dezinformare": VerdictEnum.FALSE,
        "manipulare": VerdictEnum.MOSTLY_FALSE,
        "mostly false": VerdictEnum.MOSTLY_FALSE,
        "parțial fals": VerdictEnum.MOSTLY_FALSE,
        "true": VerdictEnum.TRUE,
        "adevărat": VerdictEnum.TRUE,
        "adevarat": VerdictEnum.TRUE,
        "mostly true": VerdictEnum.MOSTLY_TRUE,
        "parțial adevărat": VerdictEnum.MOSTLY_TRUE,
        "mixture": VerdictEnum.MIXTURE,
        "amestec": VerdictEnum.MIXTURE,
        "amestec dintre fapte și opinii": VerdictEnum.MIXTURE,
        "etichetare": VerdictEnum.MOSTLY_FALSE,
    },
    "veridica.md": {
        "false": VerdictEnum.FALSE,
        "fals": VerdictEnum.FALSE,
        "dezinformare": VerdictEnum.FALSE,
        "fake news": VerdictEnum.FALSE,
        "propaganda": VerdictEnum.FALSE,
        "propagandă": VerdictEnum.FALSE,
        "manipulare": VerdictEnum.MOSTLY_FALSE,
        "fact-checking": VerdictEnum.FALSE,  # În Veridica.md majoritatea fact-checkurilor analizează dezinformări
        "true": VerdictEnum.TRUE,
        "adevărat": VerdictEnum.TRUE,
        "mostly false": VerdictEnum.MOSTLY_FALSE,
        "mostly true": VerdictEnum.MOSTLY_TRUE,
        "mixture": VerdictEnum.MIXTURE,
    },
}

GENERAL_VERDICT_KEYWORDS: List[Tuple[re.Pattern, VerdictEnum]] = [
    (re.compile(r"\b(mostly\s*true|par[tț]ial\s*adev[aă]rat|[iî]n\s*mare\s*parte\s*adev[aă]rat)\b", re.IGNORECASE), VerdictEnum.MOSTLY_TRUE),
    (re.compile(r"\b(mostly\s*false|par[tț]ial\s*fals|[iî]n\s*mare\s*parte\s*fals|in[sș]el[aă]tor|misleading|exagerat|etichetar[ei]|manipulare)\b", re.IGNORECASE), VerdictEnum.MOSTLY_FALSE),
    (re.compile(r"\b(mixture|amestec|nuan[tț]at|trunchiat|context\s*lips[aă]|scos\s*din\s*context|scoas[aă]\s*din\s*context|f[aă]r[aă]\s*context|half\s*true)\b", re.IGNORECASE), VerdictEnum.MIXTURE),
    (re.compile(r"\b(unproven|nedovedit|neverificabil|neconfirmat|f[aă]r[aă]\s*dovezi|nesus[tț]inut)\b", re.IGNORECASE), VerdictEnum.UNPROVEN),
    (re.compile(r"\b(fals|false|fake(\s*news)?|dezinformare|minciun[aă]|propagand[aă]|fabricat|trucat|modificat|alterat)\b", re.IGNORECASE), VerdictEnum.FALSE),
    (re.compile(r"\b(true|adev[aă]rat|corect|real)\b", re.IGNORECASE), VerdictEnum.TRUE),
]


def normalize_verdict(raw_verdict: Optional[str], source_dataset: str = "") -> VerdictEnum:
    if not raw_verdict:
        return VerdictEnum.OTHER

    cleaned = raw_verdict.strip().lower()

    source_key = source_dataset.lower()
    for s_name, mapping in VERDICT_MAPPINGS.items():
        if s_name in source_key or source_key in s_name:
            if cleaned in mapping:
                return mapping[cleaned]

    for mapping in VERDICT_MAPPINGS.values():
        if cleaned in mapping:
            return mapping[cleaned]

    for pattern, verdict in GENERAL_VERDICT_KEYWORDS:
        if pattern.search(cleaned):
            return verdict

    return VerdictEnum.OTHER


# =====================================================================
# 3. UTILITARE CURĂȚARE ȘI CANONIZARE
# =====================================================================

def canonicalize_url(url: Optional[str]) -> str:
    if not url:
        return ""
    url = url.strip()
    parsed = urlparse(url)
    scheme = parsed.scheme.lower() if parsed.scheme else "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    if not path:
        path = "/"
    return urlunparse((scheme, netloc, path, "", "", ""))


def generate_record_id(source: str, url: str, claim: str) -> str:
    source_prefix = re.sub(r"[^a-zA-Z0-9]", "", source).lower()[:6]
    content_key = f"{canonicalize_url(url)}|{claim.strip().lower()}"
    hash_digest = hashlib.sha256(content_key.encode("utf-8")).hexdigest()[:12]
    return f"{source_prefix}_{hash_digest}"


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r"\r\n|\r", "\n", str(text))
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def strip_verdict_prefix_from_title(title: str) -> Tuple[str, Optional[str]]:
    pattern = r"^\s*(atentie\s*fals|atenție\s*fals|fals|adev[aă]rat|par[tț]ial\s*adev[aă]rat|par[tț]ial\s*fals|context\s*lips[aă]|trunchiat|dezinformare|fake\s*news|manipulare|propagand[aă]|nu)\s*[:\|–-]\s*"
    match = re.match(pattern, title, flags=re.IGNORECASE)
    if match:
        prefix = match.group(1).strip()
        cleaned = title[match.end():].strip()
        return cleaned, prefix
    return title, None


# =====================================================================
# 4. MODUL ENRICHMENT CU OLLAMA (Tokens mari + Format Schema Pydantic)
# =====================================================================

class OllamaExtractor:
    """
    Extrage semantic câmpurile lipsă folosind schema Pydantic și o limită generoasă de tokens (num_predict: 2048).
    Include cache persistent (.ollama_cache_md.json) pentru refolosirea extracțiilor anterioare.
    """
    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        enabled: bool = False,
        num_predict: int = 2048,
        max_chars: int = 6000,
        cache_file: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.num_predict = num_predict
        self.max_chars = max_chars
        self.enabled = enabled and OLLAMA_AVAILABLE
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, Any]] = {}

        if self.cache_file and self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(f"[Ollama] S-au încărcat {len(self.cache)} extrageri din cache ({self.cache_file.name}).")
            except Exception as e:
                logger.warning(f"[Ollama] Nu s-a putut citi fișierul cache: {e}")

        if self.enabled:
            logger.info(f"[Ollama] Extractor activat cu modelul '{self.model_name}' (num_predict={self.num_predict}, max_chars={self.max_chars})")
        elif enabled and not OLLAMA_AVAILABLE:
            logger.warning("[Ollama] Pachetul 'ollama' nu este instalat. Extragerea LLM este dezactivată.")

    def _save_cache(self):
        if self.cache_file and self.cache:
            try:
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"[Ollama] Nu s-a putut salva fișierul cache: {e}")

    def extract_fields(self, article_text: str, current_title: str = "", item_id: str = "") -> Dict[str, Any]:
        if not self.enabled or not article_text:
            return {}

        cache_key = item_id or hashlib.sha256(f"{current_title}|{article_text[:300]}".encode()).hexdigest()[:16]

        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            if cached_item and "claim" in cached_item and "verdict" in cached_item:
                return cached_item

        truncated_text = article_text[:self.max_chars]

        prompt = f"""Ești un expert în fact-checking pentru Republica Moldova.
Extrage datele structurate din analiza următoare:
Titlu: {current_title}
Text articol:
\"\"\"{truncated_text}\"\"\"
"""
        try:
            # Folosim formatul strict Pydantic JSON Schema pentru generare garantată fără erori de sintaxă
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a structured data extractor for fact-checking. Output ONLY valid JSON matching the schema.",
                    },
                    {"role": "user", "content": prompt}
                ],
                format=LLMExtractionSchema.model_json_schema(),
                options={"temperature": 0.0, "num_predict": self.num_predict}
            )
            content = response.get("message", {}).get("content", "")
            result = json.loads(content)

            self.cache[cache_key] = result
            self._save_cache()
            return result
        except Exception as e:
            logger.warning(f"[Ollama] Eroare la apelul modelului pentru '{current_title[:50]}': {e}")
            return {}


# =====================================================================
# 5. ADAPTOARE MODULARE PENTRU SURSE DIN MOLDOVA
# =====================================================================

class BaseAdapter(ABC):
    def __init__(self, ollama_extractor: Optional[OllamaExtractor] = None):
        self.ollama_extractor = ollama_extractor

    @abstractmethod
    def parse(self, raw_item: Dict[str, Any]) -> Optional[CanonicalFactCheck]:
        pass


class StopfalsAdapter(BaseAdapter):
    """
    Adaptor pentru dataset_stopfals-md_clean.jsonl și dataset_stopfals-md.jsonl
    """
    def parse(self, raw_item: Dict[str, Any]) -> Optional[CanonicalFactCheck]:
        url = raw_item.get("factcheck_analysis_link") or raw_item.get("url") or ""
        claim = clean_text(raw_item.get("statement", ""))
        raw_verdict = raw_item.get("verdict", "")

        if not claim and not url:
            return None

        title = claim
        if len(title) > 150:
            title = title[:147] + "..."

        claimant = raw_item.get("statement_originator")
        if claimant in ("Unknown", "Necunoscut", "Article", "$authors", ""):
            claimant = None

        factchecker = raw_item.get("factchecker")
        if not factchecker or factchecker in ("$authors", ""):
            factchecker = "Stopfals Team"

        return CanonicalFactCheck(
            id=generate_record_id("stopfals.md", url, claim),
            source_country="Moldova",
            source_dataset="stopfals.md",
            url=canonicalize_url(url),
            title=title,
            claim=claim,
            claimant=claimant,
            date_published=raw_item.get("factcheck_date") or None,
            date_claim=raw_item.get("statement_date") or None,
            claim_source=raw_item.get("statement_source") or "social_media",
            original_verdict=str(raw_verdict),
            normalized_verdict=normalize_verdict(raw_verdict, "stopfals.md"),
            factchecker=factchecker,
            explanation_body=clean_text(raw_item.get("analysis_text", "")),
            tags=["stopfals.md", "moldova", "fact-checking"],
            extraction_method=raw_item.get("extraction_method", "ollama_fallback"),
            extra_metadata={
                "source_country": "Moldova",
                "news_source": "Stopfals.md",
            }
        )


class ZdGAdapter(BaseAdapter):
    """
    Adaptor pentru dataset_zdg-md.jsonl (Ziarul de Gardă)
    """
    def parse(self, raw_item: Dict[str, Any]) -> Optional[CanonicalFactCheck]:
        url = raw_item.get("factcheck_analysis_link") or raw_item.get("url") or ""
        claim = clean_text(raw_item.get("statement", ""))
        raw_verdict = raw_item.get("verdict", "")

        if not claim and not url:
            return None

        # Curățare prefixe din claim / titlu
        cleaned_claim, prefix = strip_verdict_prefix_from_title(claim)
        title = claim
        if len(title) > 150:
            title = title[:147] + "..."

        claimant = raw_item.get("statement_originator")
        if claimant in ("Unknown", "Necunoscut", "Article text", ""):
            claimant = None

        return CanonicalFactCheck(
            id=generate_record_id("zdg.md", url, cleaned_claim),
            source_country="Moldova",
            source_dataset="zdg.md",
            url=canonicalize_url(url),
            title=title,
            claim=cleaned_claim,
            claimant=claimant,
            date_published=raw_item.get("factcheck_date") or None,
            date_claim=raw_item.get("statement_date") or None,
            claim_source=raw_item.get("statement_source") or "press",
            original_verdict=str(raw_verdict),
            normalized_verdict=normalize_verdict(raw_verdict, "zdg.md"),
            factchecker=raw_item.get("factchecker") or "Ziarul de Gardă",
            explanation_body=clean_text(raw_item.get("analysis_text", "")),
            tags=["zdg.md", "ziarul-de-garda", "moldova", "investigatii"],
            extraction_method="ollama_zdg_crawler",
            extra_metadata={
                "source_country": "Moldova",
                "news_source": "Ziarul de Gardă (ZdG)",
            }
        )


class MediacriticaAdapter(BaseAdapter):
    """
    Adaptor pentru dataset_mediacritica-md.jsonl
    """
    def parse(self, raw_item: Dict[str, Any]) -> Optional[CanonicalFactCheck]:
        url = raw_item.get("factcheck_analysis_link") or raw_item.get("url") or ""
        claim = clean_text(raw_item.get("statement", ""))
        raw_verdict = raw_item.get("verdict", "Mixture")

        if not claim and not url:
            return None

        title = claim
        if len(title) > 150:
            title = title[:147] + "..."

        claimant = raw_item.get("statement_originator")
        # Dacă claimant este un paragraf lung de analiză, îl curățăm
        if claimant and (len(claimant) > 100 or "The article" in claimant or "articolul" in claimant.lower()):
            claimant = None
        elif claimant in ("Unknown", "Necunoscut", ""):
            claimant = None

        claim_source = raw_item.get("statement_source")
        if claim_source and len(claim_source) > 100:
            claim_source = "press / media analysis"

        return CanonicalFactCheck(
            id=generate_record_id("mediacritica.md", url, claim),
            source_country="Moldova",
            source_dataset="mediacritica.md",
            url=canonicalize_url(url),
            title=title,
            claim=claim,
            claimant=claimant,
            date_published=raw_item.get("factcheck_date") or None,
            date_claim=raw_item.get("statement_date") or None,
            claim_source=claim_source or "press",
            original_verdict=str(raw_verdict),
            normalized_verdict=normalize_verdict(raw_verdict, "mediacritica.md"),
            factchecker=raw_item.get("factchecker") or "Mediacritica Team",
            explanation_body=clean_text(raw_item.get("analysis_text", "")),
            tags=["mediacritica.md", "moldova", "analiza-media", "deontologie"],
            extraction_method="ollama_mediacritica_crawler",
            extra_metadata={
                "source_country": "Moldova",
                "news_source": "Mediacritica.md",
            }
        )


class VeridicaMdAdapter(BaseAdapter):
    """
    Adaptor pentru dataset_veridica-md.jsonl
    """
    def parse(self, raw_item: Dict[str, Any]) -> Optional[CanonicalFactCheck]:
        url = raw_item.get("factcheck_analysis_link") or raw_item.get("url") or ""
        claim = clean_text(raw_item.get("statement", ""))
        raw_verdict = raw_item.get("verdict", "FALSE")

        if not claim and not url:
            return None

        # Dacă verdictul este generic ("Fact-checking"), deducem din URL sau claim
        if raw_verdict.lower() in ("fact-checking", "veridica", ""):
            if "dezinformare" in url.lower():
                raw_verdict = "DEZINFORMARE"
            elif "fals" in url.lower():
                raw_verdict = "FALS"
            elif "manipulare" in url.lower():
                raw_verdict = "MANIPULARE"

        title = claim
        if len(title) > 150:
            title = title[:147] + "..."

        claimant = raw_item.get("statement_originator")
        if claimant in ("NARAȚIUNILE", "Unknown", "Necunoscut", ""):
            claimant = None

        return CanonicalFactCheck(
            id=generate_record_id("veridica.md", url, claim),
            source_country="Moldova",
            source_dataset="veridica.md",
            url=canonicalize_url(url),
            title=title,
            claim=claim,
            claimant=claimant,
            date_published=raw_item.get("factcheck_date") or None,
            date_claim=raw_item.get("statement_date") or None,
            claim_source=raw_item.get("statement_source") or "social_media",
            original_verdict=str(raw_verdict),
            normalized_verdict=normalize_verdict(raw_verdict, "veridica.md"),
            factchecker=raw_item.get("factchecker") or "Veridica Team",
            explanation_body=clean_text(raw_item.get("analysis_text", "")),
            tags=["veridica.md", "moldova", "dezinformare", "propaganda"],
            extraction_method="veridica_md_crawler",
            extra_metadata={
                "source_country": "Moldova",
                "news_source": "Veridica.md",
            }
        )


# =====================================================================
# 6. PIPELINE ETL DE ÎNCĂRCARE, PROCESARE ȘI DEDUPLICARE
# =====================================================================

class MoldovaFactCheckETLPipeline:
    def __init__(
        self,
        input_dir: Path,
        use_ollama: bool = False,
        ollama_model: str = "qwen2.5:3b",
        num_predict: int = 2048,
        max_chars: int = 6000,
    ):
        self.input_dir = Path(input_dir)
        cache_file = self.input_dir / ".ollama_cache_md.json"
        self.ollama_extractor = OllamaExtractor(
            model_name=ollama_model,
            enabled=use_ollama,
            num_predict=num_predict,
            max_chars=max_chars,
            cache_file=cache_file,
        )

        # Mapare de fișiere la adaptoare corespunzătoare
        # Preferăm varianta _clean dacă există
        self.file_adapter_map: Dict[str, BaseAdapter] = {
            "dataset_stopfals-md_clean.jsonl": StopfalsAdapter(self.ollama_extractor),
            "dataset_stopfals-md.jsonl": StopfalsAdapter(self.ollama_extractor),
            "dataset_zdg-md.jsonl": ZdGAdapter(self.ollama_extractor),
            "dataset_mediacritica-md.jsonl": MediacriticaAdapter(self.ollama_extractor),
            "dataset_veridica-md.jsonl": VeridicaMdAdapter(self.ollama_extractor),
        }

    def _read_file_items(self, file_path: Path) -> Generator[Dict[str, Any], None, None]:
        filename = file_path.name.lower()
        try:
            if filename.endswith(".jsonl"):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, start=1):
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            yield json.loads(line_str)
                        except json.JSONDecodeError as err:
                            logger.warning(f"Linie JSON invalidă în {file_path.name}:{line_num} -> {err}")
            elif filename.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                yield item
                    elif isinstance(data, dict):
                        yield data
        except Exception as e:
            logger.error(f"Eroare la citirea fișierului {file_path}: {e}")

    def run(self) -> List[CanonicalFactCheck]:
        logger.info(f"==> Începe procesarea dataseturilor din Republica Moldova: {self.input_dir.resolve()}")

        seen_urls: Set[str] = set()
        seen_claims: Set[str] = set()
        canonical_records: List[CanonicalFactCheck] = []
        stats_by_source: Dict[str, int] = {}
        duplicates_dropped = 0

        # Verificăm dacă există varianta clean pentru stopfals ca să evităm procesarea redundantă
        files_to_process = list(self.file_adapter_map.items())
        has_stopfals_clean = (self.input_dir / "dataset_stopfals-md_clean.jsonl").exists()

        for filename, adapter in files_to_process:
            # Dacă avem varianta clean, omitem varianta brută
            if filename == "dataset_stopfals-md.jsonl" and has_stopfals_clean:
                logger.info("[i] Se folosește 'dataset_stopfals-md_clean.jsonl'; omit 'dataset_stopfals-md.jsonl'.")
                continue

            file_path = self.input_dir / filename
            if not file_path.exists():
                logger.warning(f"[!] Fișierul nu a fost găsit: {filename}")
                continue

            logger.info(f"[*] Procesez: {filename}...")
            count_source = 0

            for raw_item in self._read_file_items(file_path):
                try:
                    record = adapter.parse(raw_item)
                    if not record:
                        continue

                    # Deduplicare inteligentă pe URL și claim
                    clean_url = record.url.strip().lower()
                    claim_key = re.sub(r"\W+", "", record.claim.lower())[:80]

                    if clean_url and clean_url in seen_urls:
                        duplicates_dropped += 1
                        continue

                    if claim_key and len(claim_key) > 20 and claim_key in seen_claims:
                        duplicates_dropped += 1
                        continue

                    if clean_url:
                        seen_urls.add(clean_url)
                    if claim_key:
                        seen_claims.add(claim_key)

                    canonical_records.append(record)
                    count_source += 1
                    stats_by_source[record.source_dataset] = stats_by_source.get(record.source_dataset, 0) + 1

                except Exception as e:
                    logger.warning(f"Eroare la parsarea unui element din {filename}: {e}")

            logger.info(f"    -> Parsat cu succes {count_source} înregistrări unice din {filename}")

        # Statistici finale
        logger.info("=" * 60)
        logger.info(f"Total înregistrări canonice unificate (Moldova): {len(canonical_records)}")
        logger.info(f"Total duplicate eliminate: {duplicates_dropped}")
        logger.info("Distribuție pe surse:")
        for src, count in sorted(stats_by_source.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  - {src}: {count}")

        verdict_counts: Dict[str, int] = {}
        for r in canonical_records:
            v_name = r.normalized_verdict.value
            verdict_counts[v_name] = verdict_counts.get(v_name, 0) + 1

        logger.info("Distribuție pe verdicte normalizate:")
        for v_name, count in sorted(verdict_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(canonical_records) * 100) if canonical_records else 0
            logger.info(f"  - {v_name:<14}: {count:4d} ({pct:5.1f}%)")
        logger.info("=" * 60)

        return canonical_records


# =====================================================================
# 7. EXPORT DATE (JSONL, JSON, PARQUET)
# =====================================================================

def export_dataset(
    records: List[CanonicalFactCheck],
    output_dir: Path,
    base_name: str = "unified_moldova_factchecks",
    save_json: bool = True,
    save_parquet: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export JSON Lines (.jsonl)
    jsonl_path = output_dir / f"{base_name}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
    logger.info(f"[+] Salvat JSONL: {jsonl_path.resolve()} ({jsonl_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # 2. Export JSON formatat (.json)
    if save_json:
        json_path = output_dir / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in records], f, ensure_ascii=False, indent=2)
        logger.info(f"[+] Salvat JSON:  {json_path.resolve()} ({json_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # 3. Export Parquet (.parquet)
    if save_parquet:
        if PANDAS_AVAILABLE:
            parquet_path = output_dir / f"{base_name}.parquet"
            try:
                data_dicts = [r.model_dump() for r in records]
                df = pd.DataFrame(data_dicts)
                df.to_parquet(parquet_path, index=False, engine="auto")
                logger.info(f"[+] Salvat Parquet: {parquet_path.resolve()} ({parquet_path.stat().st_size / 1024 / 1024:.2f} MB)")
            except Exception as e:
                logger.warning(f"[!] Nu s-a putut salva în format Parquet: {e}")
        else:
            logger.info("[i] Pandas/PyArrow nu este instalat; omit salvarea Parquet.")


# =====================================================================
# 8. PUNCT DE INTRARE (CLI)
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de unificare și normalizare pentru dataseturile de fact-checking din Republica Moldova."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Directorul care conține fișierele brute de date (default: directorul scriptului)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Directorul unde se va salva datasetul unificat (default: directorul scriptului)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="unified_moldova_factchecks",
        help="Numele de bază al fișierului de ieșire (fără extensie, default: unified_moldova_factchecks)",
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Activează modelul Ollama pentru îmbunătățirea câmpurilor lipsă",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="qwen2.5:3b",
        help="Modelul Ollama utilizat (default: qwen2.5:3b)",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=2048,
        help="Numărul maxim de tokens generați de Ollama (default: 2048)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="Numărul maxim de caractere din articol trimise către model (default: 6000)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Dezactivează generarea fișierului .json complet",
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Dezactivează generarea fișierului .parquet",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    pipeline = MoldovaFactCheckETLPipeline(
        input_dir=input_dir,
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
        num_predict=args.num_predict,
        max_chars=args.max_chars,
    )

    records = pipeline.run()

    if records:
        export_dataset(
            records=records,
            output_dir=output_dir,
            base_name=args.output_name,
            save_json=not args.no_json,
            save_parquet=not args.no_parquet,
        )
        logger.info("[✓] Procesare finalizată cu succes!")
    else:
        logger.warning("[!] Nu a fost generată nicio înregistrare canonică. Verifică directorul de intrare.")


if __name__ == "__main__":
    main()
