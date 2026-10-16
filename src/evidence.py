"""Evidence selection: pick the sentences of a retrieved paper that matter for a claim,
then have a verifier judge them.  Owner: Data & Evaluation (Sahil).

This module is the ONE place where "retrieved papers -> per-paper class probabilities"
is computed.  The eval harness (batched over all claims) and the product pipeline
(one claim at a time) both call score_claim_docs(), so the numbers we report come from
the same code path the user runs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .schema import Hit, Probs
from .text_utils import tokenize


# --------------------------------------------------------------------------- #
# Sentence scorers
# --------------------------------------------------------------------------- #
class LexicalSentenceScorer:
    """Score of a sentence = share of the claim's IDF mass that the sentence covers (0..1)."""

    def __init__(self, idf: Optional[dict] = None):
        self.idf = idf or {}
        self._default = max(self.idf.values()) if self.idf else 1.0

    @classmethod
    def from_docs(cls, docs: Sequence) -> "LexicalSentenceScorer":
        df, n = {}, 0
        for d in docs:
            n += 1
            for t in set(tokenize(d.text)):
                df[t] = df.get(t, 0) + 1
        return cls({t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()})

    def _w(self, tok: str) -> float:
        return self.idf.get(tok, self._default)

    def score(self, claim: str, sentences: Sequence) -> list:
        q = set(tokenize(claim))
        total = sum(self._w(t) for t in q)
        if total == 0:
            return [0.0] * len(sentences)
        return [sum(self._w(t) for t in q & set(tokenize(s))) / total for s in sentences]


class DenseSentenceScorer:
    """Cosine similarity between claim and sentence embeddings (embedder: sentence-transformers style)."""

    def __init__(self, embedder):
        self.embedder = embedder

    def score(self, claim: str, sentences: Sequence) -> list:
        if not sentences:
            return []
        vecs = self.embedder.encode([claim, *sentences], normalize_embeddings=True, show_progress_bar=False)
        vecs = np.asarray(vecs)
        return (vecs[1:] @ vecs[0]).tolist()


class CrossEncoderSentenceScorer:
    """Score = a cross-encoder's relevance of each sentence to the claim (higher = more likely evidence).

    Works with any transformers sequence-classification checkpoint: one logit (MS MARCO rerankers, and the
    selector eval/train_selector.py trains), or several labels (the last one is taken as "relevant").
    """

    def __init__(self, model_name: str, device: Optional[str] = None, max_length: int = 256, batch_size: int = 64):
        self.model_name, self.max_length, self.batch_size = model_name, max_length, batch_size
        self._device_pref, self._model = device, None
        self.name = f"cross:{model_name}"

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = self._device_pref or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).float().to(self.device).eval()

    def score(self, claim: str, sentences: Sequence) -> list:
        if not sentences:
            return []
        self._load()
        out = []
        with self._torch.no_grad():
            for s in range(0, len(sentences), self.batch_size):
                batch = list(sentences[s : s + self.batch_size])
                enc = self._tok([claim] * len(batch), batch, truncation="only_second", max_length=self.max_length,
                                padding=True, return_tensors="pt").to(self.device)
                logits = self._model(**enc).logits.float()
                out += (logits[:, 0] if logits.shape[1] == 1 else logits.log_softmax(-1)[:, -1]).cpu().tolist()
        return out


def build_sentence_scorer(spec: str = "lexical", docs: Sequence = (), **kw):
    """spec: lexical | cross:<hf-id-or-local-path>"""
    spec = (spec or "lexical").strip()
    if spec.lower() == "lexical":
        return LexicalSentenceScorer.from_docs(docs) if docs else LexicalSentenceScorer()
    if spec.lower().startswith("cross:"):
        return CrossEncoderSentenceScorer(spec[6:], **kw)
    raise ValueError(f"unknown sentence scorer '{spec}' (lexical | cross:<model>)")


def selection_recall(claim: str, sentences: Sequence, rationale_sets: Sequence, m: int, scorer) -> Optional[bool]:
    """True when the m selected sentences contain at least one COMPLETE gold rationale set (None: no rationale)."""
    sets = [set(r) for r in rationale_sets if r]
    if not sets:
        return None
    chosen = set(select_sentences(claim, sentences, m, scorer))
    return any(r <= chosen for r in sets)


def select_sentences(claim: str, sentences: Sequence, m: int = 3, scorer=None) -> list:
    """Indices (in reading order) of the m sentences most related to the claim."""
    if not sentences:
        return []
    scorer = scorer or LexicalSentenceScorer()
    scores = scorer.score(claim, sentences)
    order = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))[:m]
    return sorted(order)


def build_premise(sentences: Sequence, idxs: Sequence) -> str:
    return " ".join(sentences[i] for i in idxs)


# --------------------------------------------------------------------------- #
# Retrieved papers -> class probabilities
# --------------------------------------------------------------------------- #
@dataclass
class DocScore:
    hit: Hit
    sentence_idxs: list
    premise: str
    probs: Probs


EVIDENCE_UNITS = ("window", "abstract", "sentence")
MAX_SENTENCES_PER_DOC = 40  # sentence unit: cap on verifier calls for one (unusually long) paper


def aggregate_sentence_probs(probs: Sequence) -> tuple:
    """Per-sentence probabilities of one paper -> (paper Probs, indices of the deciding sentences).

    The paper takes the probabilities of its most decisive sentence (highest SUPPORT or CONTRADICT), so one
    clear sentence is not diluted by the rest of the abstract.  The evidence shown is every sentence whose
    top class is that same verdict with p >= 0.5, or just the most decisive sentence when there is none.
    """
    if not probs:
        return Probs(0.0, 1.0, 0.0), []
    strength = [max(p.support, p.contradict) for p in probs]
    best = max(range(len(probs)), key=lambda i: (strength[i], -i))
    side = "support" if probs[best].support >= probs[best].contradict else "contradict"
    idxs = [i for i, p in enumerate(probs) if getattr(p, side) >= 0.5 and getattr(p, side) >= p.neutral]
    return probs[best], idxs or [best]


def score_claim_docs(items: Sequence, verifier, scorer=None, m: int = 3, gold_sentences: Optional[Sequence] = None,
                     unit: str = "window") -> list:
    """Judge every (claim, retrieved paper) pair with one batched verifier call.

    items: [(claim_text, [Hit, ...]), ...]
    gold_sentences: optional, parallel to `items`; for each claim a {doc_id: [idx,...]} dict that
        overrides sentence selection (used ONLY for the oracle-rationale ablation).
    unit: what the verifier reads per paper --
        window   : the m sentences most related to the claim, joined (the Session 5 design)
        abstract : the whole abstract (no sentence selection)
        sentence : every sentence on its own; the paper's verdict comes from aggregate_sentence_probs()
    Returns: list (one per claim) of list[DocScore] in the order of the input hits.
    """
    if unit not in EVIDENCE_UNITS:
        raise ValueError(f"unknown evidence unit '{unit}' ({' | '.join(EVIDENCE_UNITS)})")
    scorer = scorer or LexicalSentenceScorer()
    pairs, index = [], []  # index[i] = (claim_no, doc_no, sentence_idx or None)
    prepared = []
    for ci, (claim, hits) in enumerate(items):
        row = []
        for di, hit in enumerate(hits):
            sents = hit.doc.sentences
            override = gold_sentences[ci].get(hit.doc.doc_id) if gold_sentences is not None else None
            if unit == "sentence":
                idxs = list(override) if override else list(range(min(len(sents), MAX_SENTENCES_PER_DOC)))
                idxs = [i for i in idxs if i < len(sents)]
                row.append((idxs, None))
                if idxs:
                    for i in idxs:
                        pairs.append((sents[i], claim))
                        index.append((ci, di, i))
                else:  # no abstract: judge the title, as the window unit does
                    pairs.append((hit.doc.title, claim))
                    index.append((ci, di, None))
                continue
            if override:
                idxs = list(override)
            elif unit == "abstract":
                idxs = list(range(len(sents)))
            else:
                idxs = select_sentences(claim, sents, m=m, scorer=scorer)
            premise = build_premise(sents, idxs) if idxs else hit.doc.title
            row.append((idxs, premise))
            pairs.append((premise, claim))
            index.append((ci, di, None))
        prepared.append(row)

    probs = verifier.predict_pairs(pairs) if pairs else []
    per_doc = {}
    for (ci, di, si), p in zip(index, probs):
        per_doc.setdefault((ci, di), []).append((si, p))
    out = [[None] * len(hits) for _, hits in items]
    for (ci, di), got in per_doc.items():
        idxs, premise = prepared[ci][di]
        hit = items[ci][1][di]
        if unit == "sentence" and got[0][0] is not None:
            doc_probs, pick = aggregate_sentence_probs([p for _, p in got])
            chosen = [got[j][0] for j in pick]
            out[ci][di] = DocScore(hit, chosen, build_premise(hit.doc.sentences, chosen), doc_probs)
        else:
            out[ci][di] = DocScore(hit, idxs, premise if premise is not None else hit.doc.title, got[0][1])
    return out
