#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline ETL pentru unificarea tuturor seturilor de date de fact-checking din România.
Procesează sursele:
  - Factual.ro (dataset_factual_api.json, dataset_factual_api_2.json)
  - Brodhub.eu (dataset_brodhub_eu_updated.jsonl)
  - Context.ro (dataset_context-ro.jsonl)
  - Rador.ro (dataset_rador_ro.json)
  - Veridica.ro (dataset_veridica_ro.json)
  - AFP Verificat (dataset_verificat_afp.json)

Include:
  1. Schemă Canonică Pydantic (Golden Schema)
  2. Normalizare exhaustivă a verdictelor (taxonomie standard)
  3. Adaptoare modulare pentru fiecare format de intrare
  4. Extragere euristică + suport opțional pentru îmbunătățire cu LLM via Ollama (qwen2.5:3b)
  5. Deduplicare inteligentă pe baza URL-ului canonic și a afirmațiilor
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
logger = logging.getLogger("UnifyRomaniaDatasets")


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
    source_dataset: str = Field(description="Numele platformei sursă (ex: factual.ro, context.ro, veridica.ro, brodhub.eu, rador.ro, afp)")
    url: str = Field(description="Linkul canonic către analiza de fact-checking")
    title: str = Field(description="Titlul articolului sau al investigației")
    claim: str = Field(description="Afirmația / teza verificată")
    claimant: Optional[str] = Field(default=None, description="Persoana sau entitatea care a făcut afirmația (sau 'Necunoscut' / 'Surse Multiple')")
    date_published: Optional[str] = Field(default=None, description="Data publicării analizei de fact-checking (format ISO/standard)")
    date_claim: Optional[str] = Field(default=None, description="Data la care s-a făcut afirmația inițială (dacă este disponibilă)")
    claim_source: Optional[str] = Field(default=None, description="Canalul prin care a fost propagată afirmația (social_media, tv, press, web, tiktok etc.)")
    original_verdict: str = Field(description="Verdictul brut extras direct din sursă")
    normalized_verdict: VerdictEnum = Field(description="Verdictul mapat la taxonomia standard")
    factchecker: Optional[str] = Field(default=None, description="Numele autorului sau organizației de fact-checking")
    explanation_body: str = Field(description="Textul analizei, argumentarea sau rezumatul detaliat")
    tags: List[str] = Field(default_factory=list, description="Etichete tematice, categorii sau narațiuni asociate")
    extraction_method: str = Field(default="direct_mapping", description="Metoda utilizată pentru extragere (claim_review, direct_mapping, regex_heuristic, ollama_fallback)")
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

# Dicționare specifice de mapare per sursă
VERDICT_MAPPINGS: Dict[str, Dict[str, VerdictEnum]] = {
    "factual.ro": {
        "adevarat": VerdictEnum.TRUE,
        "adevărat": VerdictEnum.TRUE,
        "true": VerdictEnum.TRUE,
        "fals": VerdictEnum.FALSE,
        "false": VerdictEnum.FALSE,
        "fals / dezinformare": VerdictEnum.FALSE,
        "partial adevarat": VerdictEnum.MOSTLY_TRUE,
        "parțial adevărat": VerdictEnum.MOSTLY_TRUE,
        "mostly true": VerdictEnum.MOSTLY_TRUE,
        "mostly_true": VerdictEnum.MOSTLY_TRUE,
        "partial fals": VerdictEnum.MOSTLY_FALSE,
        "parțial fals": VerdictEnum.MOSTLY_FALSE,
        "mostly false": VerdictEnum.MOSTLY_FALSE,
        "mostly_false": VerdictEnum.MOSTLY_FALSE,
        "trunchiat": VerdictEnum.MIXTURE,
        "context": VerdictEnum.MIXTURE,
        "context lipsa": VerdictEnum.MIXTURE,
        "context lipsă": VerdictEnum.MIXTURE,
        "fara context": VerdictEnum.MIXTURE,
        "fără context": VerdictEnum.MIXTURE,
        "scos din context": VerdictEnum.MIXTURE,
        "scoasa din context": VerdictEnum.MIXTURE,
        "scoasă din context": VerdictEnum.MIXTURE,
        "mixture": VerdictEnum.MIXTURE,
        "atentie la fake": VerdictEnum.FALSE,
        "atenție la fake": VerdictEnum.FALSE,
        "nedovedit": VerdictEnum.UNPROVEN,
        "neverificabil": VerdictEnum.UNPROVEN,
    },
    "brodhub.eu": {
        "true": VerdictEnum.TRUE,
        "adevărat": VerdictEnum.TRUE,
        "adevarat": VerdictEnum.TRUE,
        "false": VerdictEnum.FALSE,
        "fals": VerdictEnum.FALSE,
        "mostly true": VerdictEnum.MOSTLY_TRUE,
        "mostly_true": VerdictEnum.MOSTLY_TRUE,
        "parțial adevărat": VerdictEnum.MOSTLY_TRUE,
        "partial adevarat": VerdictEnum.MOSTLY_TRUE,
        "mostly false": VerdictEnum.MOSTLY_FALSE,
        "mostly_false": VerdictEnum.MOSTLY_FALSE,
        "parțial fals": VerdictEnum.MOSTLY_FALSE,
        "partial fals": VerdictEnum.MOSTLY_FALSE,
        "mixture": VerdictEnum.MIXTURE,
        "amestec": VerdictEnum.MIXTURE,
        "unproven": VerdictEnum.UNPROVEN,
        "nedovedit": VerdictEnum.UNPROVEN,
    },
    "context.ro": {
        "true": VerdictEnum.TRUE,
        "adevărat": VerdictEnum.TRUE,
        "adevarat": VerdictEnum.TRUE,
        "false": VerdictEnum.FALSE,
        "fals": VerdictEnum.FALSE,
        "mostly true": VerdictEnum.MOSTLY_TRUE,
        "mostly_true": VerdictEnum.MOSTLY_TRUE,
        "parțial adevărat": VerdictEnum.MOSTLY_TRUE,
        "mostly false": VerdictEnum.MOSTLY_FALSE,
        "mostly_false": VerdictEnum.MOSTLY_FALSE,
        "parțial fals": VerdictEnum.MOSTLY_FALSE,
        "mixture": VerdictEnum.MIXTURE,
        "amestec": VerdictEnum.MIXTURE,
    },
    "rador.ro": {
        "fals": VerdictEnum.FALSE,
        "false": VerdictEnum.FALSE,
        "fake news": VerdictEnum.FALSE,
        "dezinformare": VerdictEnum.FALSE,
        "manipulare": VerdictEnum.MOSTLY_FALSE,
        "adevarat": VerdictEnum.TRUE,
        "adevărat": VerdictEnum.TRUE,
        "true": VerdictEnum.TRUE,
        "partial adevarat": VerdictEnum.MOSTLY_TRUE,
        "parțial adevărat": VerdictEnum.MOSTLY_TRUE,
        "partial fals": VerdictEnum.MOSTLY_FALSE,
        "parțial fals": VerdictEnum.MOSTLY_FALSE,
        "context lipsa": VerdictEnum.MIXTURE,
        "context lipsă": VerdictEnum.MIXTURE,
    },
    "veridica.ro": {
        "dezinformare": VerdictEnum.FALSE,
        "fake news": VerdictEnum.FALSE,
        "fals": VerdictEnum.FALSE,
        "false": VerdictEnum.FALSE,
        "propaganda": VerdictEnum.FALSE,
        "propagandă": VerdictEnum.FALSE,
        "manipulare": VerdictEnum.MOSTLY_FALSE,
        "stire falsa": VerdictEnum.FALSE,
        "știre falsă": VerdictEnum.FALSE,
        "naratiune falsa": VerdictEnum.FALSE,
        "narațiune falsă": VerdictEnum.FALSE,
        "context fals": VerdictEnum.MIXTURE,
        "context manipulat": VerdictEnum.MOSTLY_FALSE,
    },
    "verificat_afp": {
        "fals": VerdictEnum.FALSE,
        "false": VerdictEnum.FALSE,
        "mostly false": VerdictEnum.MOSTLY_FALSE,
        "partially false": VerdictEnum.MOSTLY_FALSE,
        "parțial fals": VerdictEnum.MOSTLY_FALSE,
        "true": VerdictEnum.TRUE,
        "adevărat": VerdictEnum.TRUE,
        "mostly true": VerdictEnum.MOSTLY_TRUE,
        "parțial adevărat": VerdictEnum.MOSTLY_TRUE,
        "mixture": VerdictEnum.MIXTURE,
        "mixt": VerdictEnum.MIXTURE,
        "înșelător": VerdictEnum.MOSTLY_FALSE,
        "inselator": VerdictEnum.MOSTLY_FALSE,
        "misleading": VerdictEnum.MOSTLY_FALSE,
        "alterat": VerdictEnum.FALSE,
        "modificat": VerdictEnum.FALSE,
        "scos din context": VerdictEnum.MIXTURE,
    },
}

# Mapare de rezervă bazată pe cuvinte cheie
GENERAL_VERDICT_KEYWORDS: List[Tuple[re.Pattern, VerdictEnum]] = [
    (re.compile(r"\b(mostly\s*true|par[tț]ial\s*adev[aă]rat|[iî]n\s*mare\s*parte\s*adev[aă]rat)\b", re.IGNORECASE), VerdictEnum.MOSTLY_TRUE),
    (re.compile(r"\b(mostly\s*false|par[tț]ial\s*fals|[iî]n\s*mare\s*parte\s*fals|in[sș]el[aă]tor|misleading|exagerat)\b", re.IGNORECASE), VerdictEnum.MOSTLY_FALSE),
    (re.compile(r"\b(mixture|amestec|nuan[tț]at|trunchiat|context\s*lips[aă]|scos\s*din\s*context|scoas[aă]\s*din\s*context|f[aă]r[aă]\s*context|half\s*true)\b", re.IGNORECASE), VerdictEnum.MIXTURE),
    (re.compile(r"\b(unproven|nedovedit|neverificabil|neconfirmat|f[aă]r[aă]\s*dovezi|nesus[tț]inut)\b", re.IGNORECASE), VerdictEnum.UNPROVEN),
    (re.compile(r"\b(fals|false|fake(\s*news)?|dezinformare|manipulare|minciun[aă]|propagand[aă]|fabricat|trucat|modificat|alterat)\b", re.IGNORECASE), VerdictEnum.FALSE),
    (re.compile(r"\b(true|adev[aă]rat|corect|real)\b", re.IGNORECASE), VerdictEnum.TRUE),
]


def normalize_verdict(raw_verdict: Optional[str], source_dataset: str = "") -> VerdictEnum:
    """
    Normalizează verdictul brut primit din orice sursă către VerdictEnum standard.
    """
    if not raw_verdict:
        return VerdictEnum.OTHER

    cleaned = raw_verdict.strip().lower()

    # 1. Căutare în maparea directă a sursei
    source_key = source_dataset.lower()
    for s_name, mapping in VERDICT_MAPPINGS.items():
        if s_name in source_key or source_key in s_name:
            if cleaned in mapping:
                return mapping[cleaned]

    # 2. Căutare în toate mapările
    for mapping in VERDICT_MAPPINGS.values():
        if cleaned in mapping:
            return mapping[cleaned]

    # 3. Potrivire pe bază de regex / expresii regulate
    for pattern, verdict in GENERAL_VERDICT_KEYWORDS:
        if pattern.search(cleaned):
            return verdict

    return VerdictEnum.OTHER


# =====================================================================
# 3. UTILITARE CURĂȚARE ȘI CANONIZARE
# =====================================================================

def canonicalize_url(url: Optional[str]) -> str:
    """Curăță și normalizează URL-ul pentru deduplicare precisă."""
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
    """Generează un ID determinist și unic pentru înregistrare."""
    source_prefix = re.sub(r"[^a-zA-Z0-9]", "", source).lower()[:6]
    content_key = f"{canonicalize_url(url)}|{claim.strip().lower()}"
    hash_digest = hashlib.sha256(content_key.encode("utf-8")).hexdigest()[:12]
    return f"{source_prefix}_{hash_digest}"


def clean_text(text: Optional[str]) -> str:
    """Elimină spațiile albe redundante și caracterele speciale de control."""
    if not text:
        return ""
    text = re.sub(r"\r\n|\r", "\n", str(text))
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def strip_verdict_prefix_from_title(title: str) -> Tuple[str, Optional[str]]:
    """
    Extrage prefixele tipice din titluri (ex: 'FALS | ...', 'DEZINFORMARE: ...', 'CONTEXT LIPSĂ | ...').
    Returnează (titlu_curatat, prefix_gasit).
    """
    pattern = r"^\s*(fals|adev[aă]rat|par[tț]ial\s*adev[aă]rat|par[tț]ial\s*fals|context\s*lips[aă]|trunchiat|dezinformare|fake\s*news|manipulare|propagand[aă]|nu)\s*[:\|–-]\s*"
    match = re.match(pattern, title, flags=re.IGNORECASE)
    if match:
        prefix = match.group(1).strip()
        cleaned = title[match.end():].strip()
        return cleaned, prefix
    return title, None


# =====================================================================
# 4. MODUL ENRICHMENT CU OLLAMA (Fallback / Extragere Semantică)
# =====================================================================

class OllamaExtractor:
    """
    Extrage semantic câmpuri lipsă (afirmație, autor afirmație, verdict) folosind Ollama.
    Include cache persistent (.ollama_cache_ro.json) pentru a evita reapelarea modelului
    și pentru a permite rularea doar pe articolele noi/care au eșuat anterior.
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
            logger.info(f"[Ollama] Extractor inițializat cu modelul '{self.model_name}' (num_predict={self.num_predict}, max_chars={self.max_chars})")
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
        """
        Extrage claim, claimant și verdict dacă nu sunt disponibile direct.
        Verifică mai întâi cache-ul; dacă există un rezultat valid anterior, îl refolosește.
        """
        if not self.enabled or not article_text:
            return {}

        cache_key = item_id or hashlib.sha256(f"{current_title}|{article_text[:300]}".encode()).hexdigest()[:16]

        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            if cached_item and "claim" in cached_item and "verdict" in cached_item:
                return cached_item

        truncated_text = article_text[:self.max_chars]

        prompt = f"""Ești un asistent specializat în structurarea articolelor de fact-checking în limba română.
Extrage următoarele informații din textul de mai jos:
Titlu articol: {current_title}
Text articol:
\"\"\"{truncated_text}\"\"\"
"""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a structured data extractor for fact-checking. Respond strictly in JSON matching the schema."},
                    {"role": "user", "content": prompt}
                ],
                format=LLMExtractionSchema.model_json_schema(),
                options={"temperature": 0.0, "num_predict": self.num_predict}
            )
            content = response.get("message", {}).get("content", "")
            result = json.loads(content)
            
            # Salvare în cache persistent
            self.cache[cache_key] = result
            self._save_cache()
            return result
        except Exception as e:
            logger.warning(f"[Ollama] Eroare la apelul modelului pentru '{current_title[:50]}': {e}")
            return {}


# =====================================================================
# 5. ADAPTOARE MODULARE PENTRU SURSE DE DATE
# =====================================================================

class BaseAdapter(ABC):
    def __init__(self, ollama_extractor: Optional[OllamaExtractor] = None):
        self.ollama_extractor = ollama_extractor

    @abstractmethod
    def parse(self, raw_item: Dict[str, Any]) -> Optional[CanonicalFactCheck]:
        pass


class BrodhubAdapter(BaseAdapter):
    """
    Adaptor pentru dataset_brodhub_eu_updated.jsonl
    """
    def parse(self, raw_item: Dict[str, Any]) -> Optional[CanonicalFactCheck]:
        url = raw_item.get("factcheck_analysis_link") or raw_item.get("url") or ""
        claim = clean_text(raw_item.get("statement", ""))
        raw_verdict = raw_item.get("verdict", "")
        
        if not claim and not url:
            return None

        # Titlul poate fi dedus din statement sau din link slug
        title = claim
        if len(title) > 150:
            title = title[:147] + "..."

        claimant = raw_item.get("statement_originator")
        if claimant in ("Unknown", "Necunoscut", "$authors", ""):
            claimant = None

        factchecker = raw_item.get("factchecker")
        if factchecker in ("$authors", "", None):
            factchecker = "BROD Hub"

        return CanonicalFactCheck(
            id=generate_record_id("brodhub.eu", url, claim),
            source_dataset="brodhub.eu",
            url=canonicalize_url(url),
            title=title,
            claim=claim,
            claimant=claimant,
            date_published=raw_item.get("factcheck_date") or None,
            date_claim=raw_item.get("statement_date") or None,
            claim_source=raw_item.get("statement_source") or None,
            original_verdict=str(raw_verdict),
            normalized_verdict=normalize_verdict(raw_verdict, "brodhub.eu"),
            factchecker=factchecker,
            explanation_body=clean_text(raw_item.get("analysis_text", "")),
            tags=["brodhub", "romania", "factcheck"],
            extraction_method=raw_item.get("extraction_method", "claim_review"),
            extra_metadata={
                "source_country": raw_item.get("source_country", "Romania / Bulgaria"),
                "news_source": raw_item.get("news_source", "brodhub.eu"),
            }
        )


class ContextRoAdapter(BaseAdapter):
    """
    Adaptor pentru dataset_context-ro.jsonl
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
        if claimant in ("Unknown", "Necunoscut", ""):
            claimant = None

        return CanonicalFactCheck(
            id=generate_record_id("context.ro", url, claim),
            source_dataset="context.ro",
            url=canonicalize_url(url),
            title=title,
            claim=claim,
            claimant=claimant,
            date_published=raw_item.get("factcheck_date") or None,
            date_claim=raw_item.get("statement_date") or None,
            claim_source=raw_item.get("statement_source") or "web",
            original_verdict=str(raw_verdict),
            normalized_verdict=normalize_verdict(raw_verdict, "context.ro"),
            factchecker=raw_item.get("factchecker") or "Context.ro Team",
            explanation_body=clean_text(raw_item.get("analysis_text", "")),
            tags=["context.ro", "romania", "investigatii"],
            extraction_method="ollama_qwen2.5:3b",
            extra_metadata={
                "source_country": raw_item.get("source_country", "Romania"),
            }
        )


class FactualAdapter(BaseAdapter):
    """
    Adaptor pentru dataset_factual_api.json și dataset_factual_api_2.json
    """
    def parse(self, raw_item: Dict[str, Any]) -> Optional[CanonicalFactCheck]:
        url = raw_item.get("factcheck_analysis_link") or raw_item.get("url") or ""
        raw_statement = clean_text(raw_item.get("statement", ""))
        raw_verdict = raw_item.get("verdict", "")

        if not raw_statement and not url:
            return None

        # Curățare prefix verdict din statement/titlu (ex: 'FALS | ...')
        cleaned_claim, prefix = strip_verdict_prefix_from_title(raw_statement)
        title = raw_statement

        # Dacă verdictul lipsește sau e generic, dar exista prefix în titlu
        if (not raw_verdict or raw_verdict.lower() in ("context", "")) and prefix:
            verdict_for_norm = prefix
        else:
            verdict_for_norm = raw_verdict

        claimant = raw_item.get("statement_originator")
        if claimant in ("Unknown", "Necunoscut", ""):
            claimant = None

        return CanonicalFactCheck(
            id=generate_record_id("factual.ro", url, cleaned_claim),
            source_dataset="factual.ro",
            url=canonicalize_url(url),
            title=title,
            claim=cleaned_claim,
            claimant=claimant,
            date_published=raw_item.get("factcheck_date") or None,
            date_claim=raw_item.get("statement_date") or None,
            claim_source=raw_item.get("statement_source") or "social_media",
            original_verdict=str(raw_verdict or prefix or "context"),
            normalized_verdict=normalize_verdict(verdict_for_norm, "factual.ro"),
            factchecker=raw_item.get("factchecker") or "Factual.ro Team",
            explanation_body=clean_text(raw_item.get("analysis_text", "")),
            tags=["factual.ro", "romania", "politic", "social"],
            extraction_method="factual_api",
            extra_metadata={
                "source_country": raw_item.get("source_country", "Romania"),
            }
        )


class RadorAdapter(BaseAdapter):
    """
    Adaptor pentru dataset_rador_ro.json
    """
    def parse(self, raw_item: Dict[str, Any]) -> Optional[CanonicalFactCheck]:
        url = raw_item.get("factcheck_analysis_link") or raw_item.get("url") or ""
        raw_statement = clean_text(raw_item.get("statement", ""))
        raw_title = clean_text(raw_item.get("raw_title", "")) or raw_statement
        raw_verdict = raw_item.get("verdict", "fals")

        if not raw_statement and not url:
            return None

        claimant = raw_item.get("statement_originator")
        if claimant in ("Necunoscut", "Unknown", ""):
            claimant = None

        return CanonicalFactCheck(
            id=generate_record_id("rador.ro", url, raw_statement),
            source_dataset="rador.ro",
            url=canonicalize_url(url),
            title=raw_title,
            claim=raw_statement,
            claimant=claimant,
            date_published=raw_item.get("factcheck_date") or None,
            date_claim=raw_item.get("statement_date") or None,
            claim_source=raw_item.get("statement_source") or "social_media",
            original_verdict=str(raw_verdict),
            normalized_verdict=normalize_verdict(raw_verdict, "rador.ro"),
            factchecker=raw_item.get("factchecker") or "Rador.ro",
            explanation_body=clean_text(raw_item.get("analysis_text", "")),
            tags=["rador.ro", "romania", "fake-news"],
            extraction_method="rador_crawler",
            extra_metadata={
                "source_country": raw_item.get("source_country", "Romania"),
            }
        )


class VeridicaAdapter(BaseAdapter):
    """
    Adaptor pentru dataset_veridica_ro.json (structură schema.org NewsArticle)
    """
    def parse(self, raw_item: Dict[str, Any]) -> Optional[CanonicalFactCheck]:
        url = raw_item.get("url") or ""
        headline = clean_text(raw_item.get("headline", ""))
        article_body = clean_text(raw_item.get("articleBody", ""))

        if not headline and not article_body:
            return None

        # 1. Extragere verdict și claim din headline (ex: "DEZINFORMARE: ...")
        cleaned_title, prefix = strip_verdict_prefix_from_title(headline)
        raw_verdict = prefix or "DEZINFORMARE"

        # 2. Extragere euristică din corpul articolului (secțiuni Veridica: ȘTIRE:, NARAȚIUNI:, OBIECTIVE:)
        claim = cleaned_title
        tags = ["veridica.ro", "romania", "dezinformare"]
        
        stire_match = re.search(r"ȘTIRE:\s*([^\n]+)", article_body, re.IGNORECASE)
        naratiuni_match = re.search(r"NARAȚIUNI:\s*([^\n]+)", article_body, re.IGNORECASE)
        obiective_match = re.search(r"OBIECTIVE:\s*([^\n]+)", article_body, re.IGNORECASE)

        if naratiuni_match:
            tags.append(naratiuni_match.group(1).strip())
        if obiective_match:
            tags.append(obiective_match.group(1).strip())

        # Dacă există autor în structura schema.org
        factchecker = "Veridica Team"
        author_data = raw_item.get("author")
        if isinstance(author_data, list) and len(author_data) > 0:
            factchecker = author_data[0].get("name", "Veridica Team")
        elif isinstance(author_data, dict):
            factchecker = author_data.get("name", "Veridica Team")

        # 3. Fallback / Îmbunătățire cu Ollama dacă este activat
        claimant = None
        extraction_method = "regex_heuristic"
        if self.ollama_extractor and self.ollama_extractor.enabled:
            llm_res = self.ollama_extractor.extract_fields(article_body, headline, item_id=url)
            if llm_res:
                if llm_res.get("claim"):
                    claim = llm_res["claim"]
                if llm_res.get("claimant") and llm_res["claimant"].lower() not in ("necunoscut", "unknown"):
                    claimant = llm_res["claimant"]
                if llm_res.get("verdict"):
                    raw_verdict = llm_res["verdict"]
                extraction_method = f"ollama_{self.ollama_extractor.model_name}"

        return CanonicalFactCheck(
            id=generate_record_id("veridica.ro", url, claim),
            source_dataset="veridica.ro",
            url=canonicalize_url(url),
            title=headline,
            claim=claim,
            claimant=claimant,
            date_published=raw_item.get("datePublished") or None,
            date_claim=None,
            claim_source="press / propaganda",
            original_verdict=str(raw_verdict),
            normalized_verdict=normalize_verdict(raw_verdict, "veridica.ro"),
            factchecker=factchecker,
            explanation_body=article_body,
            tags=tags,
            extraction_method=extraction_method,
            extra_metadata={
                "articleSection": raw_item.get("articleSection", ""),
                "publisher": raw_item.get("publisher", {}).get("name", "Veridica.ro") if isinstance(raw_item.get("publisher"), dict) else "Veridica.ro",
            }
        )


class VerificatAFPAdapter(BaseAdapter):
    """
    Adaptor pentru dataset_verificat_afp.json (structură schema.org ClaimReview)
    """
    def parse(self, raw_item: Dict[str, Any]) -> Optional[CanonicalFactCheck]:
        # Câmpurile ClaimReview conform schema.org
        claim = clean_text(raw_item.get("claimReviewed", ""))
        title = clean_text(raw_item.get("name", "")) or claim
        url = raw_item.get("url") or raw_item.get("source_url") or ""

        if not claim and not url:
            return None

        # Rating / Verdict
        review_rating = raw_item.get("reviewRating", {})
        raw_verdict = review_rating.get("alternateName") or review_rating.get("ratingValue") or "Fals"

        # Item reviewed & claimant
        item_reviewed = raw_item.get("itemReviewed", {})
        claimant = None
        date_claim = None
        claim_url = None
        if isinstance(item_reviewed, dict):
            date_claim = item_reviewed.get("datePublished")
            claim_url = item_reviewed.get("url")
            author_obj = item_reviewed.get("author", {})
            if isinstance(author_obj, dict):
                claimant = author_obj.get("name")
                if claimant in ("Surse Multiple", "Unknown", "Necunoscut", ""):
                    claimant = None

        # Factchecker
        author_info = raw_item.get("author", {})
        factchecker = "AFP"
        if isinstance(author_info, dict):
            factchecker = author_info.get("name", "AFP")

        return CanonicalFactCheck(
            id=generate_record_id("afp", url, claim),
            source_dataset="verificat.afp.com",
            url=canonicalize_url(url),
            title=title,
            claim=claim,
            claimant=claimant,
            date_published=raw_item.get("datePublished") or None,
            date_claim=date_claim,
            claim_source=claim_url or "social_media",
            original_verdict=str(raw_verdict),
            normalized_verdict=normalize_verdict(str(raw_verdict), "verificat_afp"),
            factchecker=factchecker,
            explanation_body=f"Verificare AFP: {title}. Afirmație: {claim}",
            tags=["afp", "verificat", "romania", "claim_review"],
            extraction_method="claim_review",
            extra_metadata={
                "ratingValue": review_rating.get("ratingValue"),
                "claimUrl": claim_url,
            }
        )


# =====================================================================
# 6. PIPELINE ETL DE ÎNCĂRCARE, PROCESARE ȘI DEDUPLICARE
# =====================================================================

class RomaniaFactCheckETLPipeline:
    def __init__(
        self,
        input_dir: Path,
        use_ollama: bool = False,
        ollama_model: str = "qwen2.5:3b",
        num_predict: int = 2048,
        max_chars: int = 6000,
    ):
        self.input_dir = Path(input_dir)
        cache_file = self.input_dir / ".ollama_cache_ro.json"
        self.ollama_extractor = OllamaExtractor(
            model_name=ollama_model,
            enabled=use_ollama,
            num_predict=num_predict,
            max_chars=max_chars,
            cache_file=cache_file,
        )

        # Mapare de fișiere la adaptoare corespunzătoare
        self.file_adapter_map: Dict[str, BaseAdapter] = {
            "dataset_brodhub_eu_updated.jsonl": BrodhubAdapter(self.ollama_extractor),
            "dataset_context-ro.jsonl": ContextRoAdapter(self.ollama_extractor),
            "dataset_factual_api.json": FactualAdapter(self.ollama_extractor),
            "dataset_factual_api_2.json": FactualAdapter(self.ollama_extractor),
            "dataset_rador_ro.json": RadorAdapter(self.ollama_extractor),
            "dataset_veridica_ro.json": VeridicaAdapter(self.ollama_extractor),
            "dataset_verificat_afp.json": VerificatAFPAdapter(self.ollama_extractor),
        }

    def _read_file_items(self, file_path: Path) -> Generator[Dict[str, Any], None, None]:
        """Citește elementele fie din JSON simplu (listă), fie din JSON Lines (.jsonl)."""
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
        """
        Execută întregul proces de ETL și deduplicare.
        """
        logger.info(f"==> Începe procesarea dataseturilor din: {self.input_dir.resolve()}")
        
        seen_urls: Set[str] = set()
        seen_claims: Set[str] = set()
        canonical_records: List[CanonicalFactCheck] = []
        stats_by_source: Dict[str, int] = {}
        duplicates_dropped = 0

        for filename, adapter in self.file_adapter_map.items():
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

                    # Deduplicare inteligentă pe URL
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
        logger.info(f"Total înregistrări canonice unificate: {len(canonical_records)}")
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
    base_name: str = "unified_romania_factchecks",
    save_json: bool = True,
    save_parquet: bool = True,
):
    """
    Salvează datele unificate în formatele solicitate.
    """
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
                # Flatten tags / dicts for parquet compatibility if needed
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
        description="Pipeline de unificare și normalizare pentru dataseturile de fact-checking din România."
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
        default="unified_romania_factchecks",
        help="Numele de bază al fișierului de ieșire (fără extensie, default: unified_romania_factchecks)",
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

    pipeline = RomaniaFactCheckETLPipeline(
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
