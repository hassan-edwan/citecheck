"""Citation-context analysis: which claims does a citing sentence attribute to the cited paper?   Owner: Modeling.

The citation audit used to send the whole cleaned citing sentence to the verifier.  Real citing sentences are not
SciFact-style atomic claims: they bundle several claims ("X improves accuracy but costs compute [12]"), mix the
citing authors' own results with the cited work ("Unlike [12], our model is faster"), or cite a paper only for a
method.  A small LOCAL LLM (src/reasoning.LocalLLM) reads the sentence and returns the atomic claims attributed to
the citation, each with the exact words it restates.  Nothing it writes is trusted unchecked:

    grounding      every claim must come with a source_quote that literally occurs in the citing sentence
    faithfulness   the NLI verifier must find quote => claim AND claim => quote, so a rewrite cannot add or drop
                   content ("is associated with" must not become "causes", and vice versa)
    scope          claims the LLM marks as needing more context, or as the citing authors' own, are not verified

aggregate() turns per-claim verdicts into one per reference: any CONTRADICT -> CONTRADICT; all SUPPORT -> SUPPORT;
some SUPPORT and the rest NEI -> PARTIAL; otherwise NEI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from .reasoning import parse_json
from .schema import CONTRADICT, NEI, PARTIAL, SUPPORT

PROMPT_VERSION = "claims-v1"
ROLES = ("background", "method", "result", "comparison", "other")

_EXAMPLE = """Example. Cited paper: "Attention is all you need"
Citing sentence: Transformers improve translation quality [12] but require substantial compute, unlike our model which runs on a phone.
Answer: {"claims": [
 {"claim": "Transformers improve translation quality.", "source_quote": "Transformers improve translation quality", "role": "result", "needs_more_context": false},
 {"claim": "Transformers require substantial compute.", "source_quote": "require substantial compute", "role": "result", "needs_more_context": false}]}
("our model runs on a phone" is the citing authors' own claim, so it is left out.)"""


def build_prompt(context: str, cited_title: str, intents: Sequence = ()) -> str:
    hint = f" Semantic Scholar labels this citation's intent as: {', '.join(intents)}." if intents else ""
    return (
        "You analyse a sentence from a scientific paper that cites another paper. List the claims the sentence "
        f"attributes to the CITED paper.{hint}\n"
        "Rules:\n"
        "- Split compound statements into short, self-contained claims; resolve 'it'/'this' only from the sentence itself.\n"
        "- Leave out the citing authors' own results, plans and opinions, and anything not attributed to the cited paper.\n"
        "- Keep the original strength: do not turn 'associated with' into 'causes', 'may' into 'does', or the reverse.\n"
        "- source_quote: the exact words of the sentence that the claim restates (copied, not paraphrased).\n"
        "- role: background | method | result | comparison | other.\n"
        "- needs_more_context: true if you cannot tell what exactly is attributed to the cited paper.\n"
        "- If the paper is cited only as a source of a method, tool or dataset with no factual claim, return no claims.\n"
        'Answer with one JSON object and nothing else: {"claims": [{"claim": "...", "source_quote": "...", '
        '"role": "...", "needs_more_context": false}]}\n\n'
        f"{_EXAMPLE}\n\nCited paper: \"{cited_title}\"\nCiting sentence: {context}\nAnswer:"
    )


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip(" .;,")


@dataclass
class ExtractedClaim:
    claim: str
    source_quote: str
    role: str = "result"
    status: str = "checked"  # checked | needs_more_context | unfaithful | ungrounded
    faithfulness: Optional[tuple] = None  # (P(quote => claim), P(claim => quote))
    note: str = ""


def parse_claims(raw: str, context: str) -> Optional[list]:
    """LLM output -> grounded ExtractedClaims; None when the output is unusable (caller falls back)."""
    obj = parse_json(raw)
    if obj is None or not isinstance(obj.get("claims"), list):
        return None
    ctx = _norm(context)
    out = []
    for c in obj["claims"]:
        if not isinstance(c, dict) or not str(c.get("claim", "")).strip():
            continue
        claim, quote = str(c["claim"]).strip()[:400], str(c.get("source_quote", "")).strip()
        role = str(c.get("role", "other")).lower()
        role = role if role in ROLES else "other"
        if not quote or _norm(quote) not in ctx:
            out.append(ExtractedClaim(claim, quote, role, "ungrounded", note="source_quote is not in the citing sentence"))
        elif c.get("needs_more_context") is True:
            out.append(ExtractedClaim(claim, quote, role, "needs_more_context", note="the sentence alone does not say what is cited"))
        else:
            out.append(ExtractedClaim(claim, quote, role))
    return out


class ClaimExtractor:
    """llm: anything with .generate(prompts) -> texts (src/reasoning.LocalLLM).  verifier: the NLI verifier used
    for the faithfulness check (None skips it, recorded on each claim)."""

    def __init__(self, llm, verifier=None, faithfulness_threshold: float = 0.5):
        self.llm, self.verifier, self.threshold = llm, verifier, faithfulness_threshold
        self.name = f"{getattr(llm, 'model_name', 'llm')}:{PROMPT_VERSION}"

    def check_faithfulness(self, claims: list) -> None:
        todo = [c for c in claims if c.status == "checked"]
        if not todo or self.verifier is None:
            return
        pairs = [p for c in todo for p in ((c.source_quote, c.claim), (c.claim, c.source_quote))]
        probs = self.verifier.predict_pairs(pairs)
        for i, c in enumerate(todo):
            fwd, back = probs[2 * i].support, probs[2 * i + 1].support
            c.faithfulness = (round(fwd, 3), round(back, 3))
            if min(fwd, back) < self.threshold:
                c.status, c.note = "unfaithful", f"the rewrite does not mean the same as the quote ({fwd:.2f} / {back:.2f})"

    def extract(self, items: Sequence) -> list:
        """items: [(context, cited_title, intents)] -> per item a list[ExtractedClaim], or None if the LLM failed."""
        raws = self.llm.generate([build_prompt(c, t, i) for c, t, i in items])
        results = []
        for (context, _, _), raw in zip(items, raws):
            claims = parse_claims(raw, context)
            if claims is not None:
                self.check_faithfulness(claims)
            results.append(claims)
        return results


def aggregate(labels: Sequence) -> Optional[str]:
    """Per-claim verdict labels of one reference -> its overall label (None when nothing was verified)."""
    labels = [lab for lab in labels if lab]
    if not labels:
        return None
    if CONTRADICT in labels:
        return CONTRADICT
    if all(lab == SUPPORT for lab in labels):
        return SUPPORT
    if SUPPORT in labels:
        return PARTIAL
    return NEI
