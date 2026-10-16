"""Turn a DOI / arXiv id / link (or pasted text) into a Paper, and list a paper's references.
Owner: Engineering (Sakshaat).

Sources (all free, no key required):
  * OpenAlex             abstract (inverted index) + metadata by DOI
  * arXiv export API     abstract by arXiv id
  * Semantic Scholar     fallback metadata + the reference list WITH citing sentences (`contexts`)

NOT VERIFIED LIVE from the build sandbox (it had no route to these hosts): the response shapes below follow each
API's documentation and are covered by tests with hand-written fixtures.  Before relying on this:
    python -m src.ingest 10.1038/nature14539
    python -m src.ingest arXiv:2004.14974
and fix anything that differs.  If an API is down or rate-limited, the app's "paste an abstract" path
always works and is what the user tests should fall back to.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .schema import Paper, Reference

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.I)
ARXIV_NEW = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d])")
ARXIV_OLD = re.compile(r"([a-z\-]+(?:\.[A-Za-z]{2})?/\d{7})(?:v\d+)?", re.I)
S2_URL = re.compile(r"semanticscholar\.org/paper/(?:[^/\s]+/)?([0-9a-f]{40})", re.I)


class IngestError(Exception):
    """Carries a message that is safe and helpful to show to the user."""


@dataclass(frozen=True)
class ParsedId:
    kind: str  # doi | arxiv | s2
    value: str


def parse_identifier(text: str) -> Optional[ParsedId]:
    text = (text or "").strip()
    if not text:
        return None
    m = DOI_RE.search(text)
    if m:
        return ParsedId("doi", m.group(0).rstrip(".,;)]}>\"'").lower())
    m = S2_URL.search(text)
    if m:
        return ParsedId("s2", m.group(1).lower())
    if "arxiv" in text.lower() or re.fullmatch(r"\s*\d{4}\.\d{4,5}(v\d+)?\s*", text):
        m = ARXIV_NEW.search(text) or ARXIV_OLD.search(text)
        if m:
            return ParsedId("arxiv", m.group(1))
    return None


def abstract_from_inverted_index(inv: Optional[dict]) -> str:
    """OpenAlex ships abstracts as {word: [positions]}; rebuild the running text."""
    if not inv:
        return ""
    slots = {}
    for word, positions in inv.items():
        for p in positions:
            slots[p] = word
    return " ".join(slots[i] for i in sorted(slots))


_CITE_MARKERS = re.compile(
    r"\[\s*\d+(?:\s*[,–—-]\s*\d+)*\s*\]"                      # [12], [3, 4], [5-7]
    r"|\(\s*[A-Z][A-Za-z\-']+(?: et al\.?| and [A-Z][A-Za-z\-']+)?,?\s*\d{4}[a-z]?"
    r"(?:\s*;\s*[A-Z][^)]{0,60}?\d{4}[a-z]?)*\s*\)"                      # (Smith et al., 2019; Lee, 2020)
)


def clean_context(text: str) -> str:
    """Strip citation markers from a citing sentence so it reads as a plain claim."""
    text = _CITE_MARKERS.sub("", text or "")
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


_CITATION_HINTS = re.compile(
    r"\b(?:proceedings|conference|symposium|workshop|association for computational linguistics|journal of|vol\.|"
    r"volume \d|pp\.|pages \d|et al\.?|in:|isbn|issn|press|publisher)(?!\w)", re.I)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def looks_like_citation(text: str) -> bool:
    """True when 'abstract' text is really a citation string (authors, venue, year), not an abstract.

    Seen live on 2026-09-29: for 10.18653/v1/2020.emnlp-main.609 OpenAlex returned the citation as the abstract, and the
    verifier then judged the claim against author names and a venue (reports/session05.md, "What did not work").
    Deliberately conservative: only SHORT texts (< 80 words) with at least three venue/year cues are rejected, so a
    real abstract that mentions a year and a conference is kept.
    """
    text = (text or "").strip()
    if not text:
        return False
    n_words = len(text.split())
    cues = len(_CITATION_HINTS.findall(text)) + len(_YEAR.findall(text))
    return n_words < 80 and cues >= 3


def usable_abstract(text: str) -> bool:
    return bool((text or "").strip()) and not looks_like_citation(text)


def paper_from_text(title: str, abstract: str) -> Paper:
    """A pasted abstract: always works offline."""
    if not (abstract or "").strip():
        raise IngestError("Please paste an abstract (a few sentences is enough).")
    pid = "pasted-" + hashlib.sha1((title + abstract).encode("utf-8")).hexdigest()[:10]
    return Paper(paper_id=pid, title=(title or "Pasted abstract").strip(), abstract=abstract.strip(), source="pasted")


class Fetcher:
    def __init__(self, session=None, cache_dir="data/cache/ingest", mailto: Optional[str] = None,
                 s2_api_key: Optional[str] = None, timeout: int = 15, retries: int = 3, sleep=time.sleep):
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.mailto = mailto or os.environ.get("CITECHECK_MAILTO")
        self.s2_key = s2_api_key or os.environ.get("S2_API_KEY")
        self.timeout, self.retries, self._sleep = timeout, retries, sleep

    # ---- HTTP -----------------------------------------------------------------------------
    def _headers(self):
        ua = "CiteCheck-course-project/0.2" + (f" (mailto:{self.mailto})" if self.mailto else "")
        h = {"User-Agent": ua}
        if self.s2_key:
            h["x-api-key"] = self.s2_key
        return h

    def _get(self, url, params=None):
        """Return the response (or None on 404). Retries 429/5xx with exponential back-off."""
        last = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, params=params, headers=self._headers(), timeout=self.timeout)
            except Exception as e:  # network down, DNS, timeout...
                last = f"network error: {type(e).__name__}"
                self._sleep(2 ** attempt * 0.5)
                continue
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                self._sleep(2 ** attempt)
                continue
            if r.status_code >= 400:
                raise IngestError(f"The paper service answered HTTP {r.status_code}.")
            return r
        raise IngestError(f"The paper service is not responding ({last}). Try again, or paste the abstract instead.")

    # ---- cache ----------------------------------------------------------------------------
    def _cache_file(self, pid: ParsedId) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / (hashlib.sha1(f"{pid.kind}:{pid.value}".encode()).hexdigest()[:16] + ".json")

    # ---- sources --------------------------------------------------------------------------
    def _openalex(self, doi: str) -> Optional[Paper]:
        params = {"mailto": self.mailto} if self.mailto else None
        r = self._get(f"https://api.openalex.org/works/https://doi.org/{doi}", params)
        if r is None:
            return None
        j = r.json()
        loc = j.get("primary_location") or {}
        return Paper(
            paper_id=f"doi:{doi}", title=j.get("display_name") or j.get("title") or "",
            abstract=abstract_from_inverted_index(j.get("abstract_inverted_index")),
            year=j.get("publication_year"), venue=((loc.get("source") or {}).get("display_name")),
            doi=doi, url=f"https://doi.org/{doi}", source="openalex",
        )

    def _arxiv(self, aid: str) -> Optional[Paper]:
        r = self._get("http://export.arxiv.org/api/query", {"id_list": aid})
        if r is None:
            return None
        ns = {"a": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            return None
        entry = root.find("a:entry", ns)
        if entry is None or entry.find("a:summary", ns) is None:
            return None
        clean = lambda s: re.sub(r"\s+", " ", s or "").strip()  # noqa: E731
        published = clean(entry.findtext("a:published", default="", namespaces=ns))
        return Paper(
            paper_id=f"arxiv:{aid}", title=clean(entry.findtext("a:title", default="", namespaces=ns)),
            abstract=clean(entry.findtext("a:summary", default="", namespaces=ns)),
            year=int(published[:4]) if published[:4].isdigit() else None, venue="arXiv",
            url=f"https://arxiv.org/abs/{aid}", source="arxiv",
        )

    def _s2_paper(self, key: str) -> Optional[Paper]:
        r = self._get(f"https://api.semanticscholar.org/graph/v1/paper/{key}",
                      {"fields": "title,abstract,year,venue,externalIds,url"})
        if r is None:
            return None
        j = r.json()
        ext = j.get("externalIds") or {}
        return Paper(paper_id=f"s2:{j.get('paperId', key)}", title=j.get("title") or "", abstract=j.get("abstract") or "",
                     year=j.get("year"), venue=j.get("venue") or None, doi=(ext.get("DOI") or "").lower() or None,
                     url=j.get("url"), source="s2")

    # ---- public API -----------------------------------------------------------------------
    def fetch_paper(self, text: str, use_cache: bool = True) -> Paper:
        pid = parse_identifier(text)
        if pid is None:
            raise IngestError("I could not find a DOI, arXiv id or Semantic Scholar link in that. "
                              "Paste e.g. 10.1038/nature14539 or arXiv:2004.14974, or paste the abstract.")
        cf = self._cache_file(pid)
        if use_cache and cf is not None and cf.exists():
            return Paper(**json.loads(cf.read_text(encoding="utf-8")))

        paper = None
        if pid.kind == "doi":
            paper = self._openalex(pid.value)
            if paper is None or not usable_abstract(paper.abstract):
                alt = self._s2_paper(f"DOI:{pid.value}")
                paper = alt if alt and usable_abstract(alt.abstract) else (paper or alt)
        elif pid.kind == "arxiv":
            paper = self._arxiv(pid.value)
            if paper is None or not usable_abstract(paper.abstract):
                alt = self._s2_paper(f"ARXIV:{pid.value}")
                paper = alt if alt and usable_abstract(alt.abstract) else (paper or alt)
        else:
            paper = self._s2_paper(pid.value)

        if paper is None:
            raise IngestError("No paper found for that identifier.")
        if not paper.abstract:
            raise IngestError(f"Found “{paper.title or pid.value}” but no abstract is available from open sources. "
                              "Paste the abstract instead.")
        if looks_like_citation(paper.abstract):  # never judge a claim against author names and a venue
            raise IngestError(f"Found “{paper.title or pid.value}”, but the open sources returned citation details "
                              "(authors, venue, year) instead of an abstract. Paste the abstract instead.")
        if cf is not None:
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(json.dumps(asdict(paper)), encoding="utf-8")
        return paper

    def fetch_references(self, text: str, limit: int = 30) -> list:
        """Outgoing citations of a paper WITH the sentence that cites each one (Semantic Scholar `contexts`).

        Many papers have no contexts (publisher licensing); those references come back with context="".
        """
        pid = parse_identifier(text)
        if pid is None:
            raise IngestError("I could not find a DOI, arXiv id or Semantic Scholar link in that.")
        key = {"doi": f"DOI:{pid.value}", "arxiv": f"ARXIV:{pid.value}", "s2": pid.value}[pid.kind]
        r = self._get(f"https://api.semanticscholar.org/graph/v1/paper/{key}/references",
                      {"fields": "contexts,intents,citedPaper.paperId,citedPaper.title,citedPaper.abstract,"
                                 "citedPaper.year,citedPaper.venue,citedPaper.externalIds", "limit": limit})
        if r is None:
            raise IngestError("Semantic Scholar has no reference list for that paper.")
        refs = []
        for row in r.json().get("data", []):
            cp = row.get("citedPaper") or {}
            if not cp.get("title"):
                continue
            ext = cp.get("externalIds") or {}
            ctxs = tuple(dict.fromkeys(c.strip() for c in (row.get("contexts") or []) if isinstance(c, str) and c.strip()))
            refs.append(Reference(
                cited=Paper(paper_id=f"s2:{cp.get('paperId')}", title=cp.get("title") or "", abstract=cp.get("abstract") or "",
                            year=cp.get("year"), venue=cp.get("venue") or None, doi=(ext.get("DOI") or "").lower() or None,
                            url=f"https://www.semanticscholar.org/paper/{cp.get('paperId')}" if cp.get("paperId") else None,
                            source="s2"),
                context=ctxs[0] if ctxs else "", intents=tuple(row.get("intents") or ()), contexts=ctxs,
            ))
        return refs


if __name__ == "__main__":  # manual smoke test:  python -m src.ingest 10.1038/nature14539
    import sys

    f = Fetcher()
    arg = sys.argv[1] if len(sys.argv) > 1 else "10.1038/nature14539"
    p = f.fetch_paper(arg)
    print(p.source, "|", p.title, "|", p.year, "\n", p.abstract[:300])
