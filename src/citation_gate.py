"""Citation-context gate: is this "citing sentence" a claim worth verifying at all?   Owner: Modeling.

Labelling 80 real Semantic Scholar contexts (data/realcite/README.md) found that about a quarter are not claims:
reference-list entries, author lists, addresses and page headers, table fragments, sentences whose cited clause is
cut off, and citations of a method, tool or material.  Verifying those produces confident verdicts on noise, so the
audit runs this gate first and reports the reason instead of a verdict.

Rules only (no model, runs anywhere).  Deliberately conservative: each rule matches a surface pattern that real
claims almost never have, and the held-out half of data/realcite measures how many real claims it wrongly drops.
It cannot catch contexts Semantic Scholar attached to the wrong reference; those read like ordinary claims.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_FRONT_MATTER = re.compile(
    r"https?://|doi\.org/|\bReceived \d|\bAccepted \d|\bCorrespond(?:ence|ing author)|\bAbbreviations:|\bP\.?O\.? Box\b"
    r"|\b[\w.+-]+@[\w-]+\.[\w.]+", re.I)
_REFERENCE_ENTRY = re.compile(r"^\s*\[?\d{1,4}[\].]?\s+[A-Z][A-Za-z'\-]+ [A-Z]{1,3}[,.]")  # "[129] Liu Y, Petreaca M, ..."
_AUTHOR_TOKEN = re.compile(r"\b[A-Z][a-z]+(?:[- ][A-Z][a-z]+)?,?\d{1,2}[*†‡]?(?=[,\s;]|$)")  # "Kraus,4" "Hahn,4*"
_METHOD = re.compile(
    r"\b(?:as (?:previously |was )?described(?: previously| before)?|described previously|previously described"
    r"|according to the (?:protocol|method|manufacturer)|kindly provided|were (?:a )?(?:kind )?gifts? from"
    r"|stereotaxic coordinates|brain atlas|primers? (?:were )?(?:described|designed) in|plating density)\b", re.I)
_LEADING_MARKER = re.compile(r"^\s*[\[(]\s*\d")  # "[71], whereas ..." : the clause [71] supports is cut off
_MARKER = re.compile(
    r"\[\s*\d|\(\s*\d{1,3}\s*[,)\-–]|\bet al\b|\(\s*[A-Z][^()]{0,60}?\b(?:19|20)\d{2}[a-z]?\s*[;)]"
    r"|\b(?:19|20)\d{2}[a-z]?\)|\d+\s*[,–-]\s*\d+\s*\.?\s*$|[A-Za-z)]\.?\s?\d{1,3}(?:[,–-]\d{1,3})*\s*\.?\s*$|\.\d{1,3}\b"
    r"|^\s*\d{1,3}(?:[\s,–-]+\d{1,3})*\s+[A-Z]")  # superscript markers that lost their formatting, at either end


@dataclass(frozen=True)
class GateResult:
    usable: bool
    rule: str = ""  # which rule rejected it
    reason: str = ""  # shown to the user


def _reject(rule: str, reason: str) -> GateResult:
    return GateResult(False, rule, reason)


def assess_context(text: str) -> GateResult:
    t = (text or "").strip()
    if not t:
        return _reject("empty", "no citing sentence")
    if _FRONT_MATTER.search(t):
        return _reject("front_matter", "page header, address or article metadata, not a citing sentence")
    if _REFERENCE_ENTRY.search(t):
        return _reject("reference_entry", "a reference-list entry, not a citing sentence")
    if len(_AUTHOR_TOKEN.findall(t)) >= 3:
        return _reject("author_list", "an author list, not a citing sentence")
    ends_like_sentence = re.search(r"[.!?]\s*[\[(]?[\d,\s–-]*[\])]?\s*\.?$", t)
    if not ends_like_sentence and (len(t.split()) < 12 or len(re.findall(r"\b[a-z]{3,}\b", t)) < 4):
        return _reject("fragment", "a table cell or sentence fragment, too short to state a claim")
    if _LEADING_MARKER.search(t):
        return _reject("cut_off", "the clause the citation supports is cut off")
    if _METHOD.search(t):
        return _reject("method", "the paper is cited for a method, tool or material, not a finding")
    if not _MARKER.search(t):
        return _reject("no_marker", "no citation marker: likely the citing authors' own result")
    return GateResult(True)
