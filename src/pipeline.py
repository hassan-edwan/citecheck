"""The end-to-end product pipeline.  Owner: Engineering (Sakshaat).

    claim  ->  retrieve papers (or use the ONE paper the user is citing)
           ->  pick the sentences of each paper that matter
           ->  verifier judges (sentences, claim)
           ->  decide(): SUPPORTS / CONTRADICTS / NOT ENOUGH EVIDENCE  (abstains below tau)
           ->  Verdict with the evidence sentences attached

The heavy lifting (sentence selection, batching, thresholding) lives in src/evidence.py and src/verdict.py, the
same code the evaluation harness calls, so reported numbers describe THIS pipeline.
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

from .config import Config, load_config
from .evidence import LexicalSentenceScorer, score_claim_docs
from .ingest import clean_context
from .schema import AuditRow, EvidenceItem, Hit, Paper, Reference, Verdict
from .verdict import decide

MAX_CLAIM_CHARS = 1500


class CiteCheckPipeline:
    def __init__(self, verifier, retriever=None, scorer=None, k: int = 3, sentences_per_paper: int = 3,
                 tau: float = 0.5, margin: float = 0.0, meta: Optional[dict] = None, warnings: Optional[list] = None,
                 evidence_unit: str = "window"):
        self.verifier, self.retriever = verifier, retriever
        self.scorer = scorer or LexicalSentenceScorer()
        self.k, self.m, self.tau, self.margin = k, sentences_per_paper, tau, margin
        self.evidence_unit = evidence_unit
        self.meta = meta or {}
        self.warnings = warnings or []

    @property
    def has_corpus(self) -> bool:
        return self.retriever is not None

    # ------------------------------------------------------------------------------------------
    def verify_claim(self, claim: str, docs: Optional[Sequence] = None) -> Verdict:
        """docs=None: search the loaded corpus.  docs=[Doc,...]: judge exactly those papers (the cited one)."""
        claim = (claim or "").strip()
        if not claim:
            raise ValueError("Enter a claim to check.")
        note = ""
        if len(claim) > MAX_CLAIM_CHARS:
            claim, note = claim[:MAX_CLAIM_CHARS], f"Claim truncated to {MAX_CLAIM_CHARS} characters. "
        t0 = time.perf_counter()
        if docs is None:
            if self.retriever is None:
                raise ValueError("No paper corpus is loaded. Give a paper (DOI / abstract) or run `python -m data.fetch_scifact`.")
            hits = self.retriever.search(claim, self.k)
        else:
            hits = [Hit(d, 0.0, i + 1) for i, d in enumerate(docs)]
        scored = score_claim_docs([(claim, hits)], self.verifier, self.scorer, m=self.m, unit=self.evidence_unit)[0]
        dec = decide([s.probs for s in scored], tau=self.tau, margin=self.margin)

        strength = lambda i: max(scored[i].probs.support, scored[i].probs.contradict)  # noqa: E731
        order = sorted(range(len(scored)), key=lambda i: -strength(i))
        if dec.doc_index is not None:
            order = [dec.doc_index] + [i for i in order if i != dec.doc_index]
        evidence = []
        for i in order:
            s, doc = scored[i], scored[i].hit.doc
            evidence.append(EvidenceItem(
                doc_id=doc.doc_id, title=doc.title, url=doc.url, sentence_idxs=s.sentence_idxs,
                sentences=[doc.sentences[j] for j in s.sentence_idxs], probs=s.probs,
                retrieval_score=s.hit.score if docs is None else None, retrieval_rank=s.hit.rank if docs is None else None,
            ))
        return Verdict(
            claim=claim, label=dec.label, confidence=dec.confidence, evidence=evidence, abstained=dec.abstained,
            note=(note + dec.note).strip(), latency_ms=1000 * (time.perf_counter() - t0), backend=dict(self.meta),
        )

    def verify_against_paper(self, claim: str, paper: Paper) -> Verdict:
        """The core use case: 'I cite this paper for this claim. Does it hold up?'"""
        doc = paper.to_doc()
        if not doc.sentences:
            raise ValueError("That paper has no abstract text to check against.")
        return self.verify_claim(claim, docs=[doc])

    # ------------------------------------------------------------------------------------------
    def audit(self, references: Sequence, max_items: int = 10, progress: Optional[Callable] = None) -> list:
        """Check each (citing sentence, cited paper) pair.  EXPERIMENTAL: no gold labels exist for this setting."""
        rows = []
        for n, ref in enumerate(list(references)[:max_items], 1):
            if progress:
                progress(n, min(len(references), max_items))
            if not ref.context.strip():
                rows.append(AuditRow(ref, None, "no citing sentence available for this reference"))
            elif not ref.cited.abstract.strip():
                rows.append(AuditRow(ref, None, "cited paper has no abstract available"))
            else:
                claim = clean_context(ref.context)
                rows.append(AuditRow(ref, self.verify_against_paper(claim, ref.cited)))
        return rows


def build_pipeline(cfg: Optional[Config] = None, allow_fallback: bool = True) -> CiteCheckPipeline:
    """Assemble the pipeline from config, degrading gracefully (and loudly) when a piece is unavailable."""
    from .data_io import find_scifact_dir, load_corpus
    from .nli import LexicalVerifier, build_verifier
    from .retrieval import BM25Retriever, build_retriever

    cfg = cfg or load_config()
    warnings = []

    corpus, ddir = None, find_scifact_dir(cfg.data_dir)
    if ddir is not None:
        corpus = load_corpus(ddir / "corpus.jsonl")
    else:
        warnings.append(f"No paper corpus found under '{cfg.data_dir}': searching a corpus is disabled "
                        "(checking a specific paper still works). Run `python -m data.fetch_scifact`.")

    fell_back = False
    retriever = None
    if corpus:
        try:
            retriever = build_retriever(cfg.retriever, corpus, model_name=cfg.embed_model)
        except Exception as e:
            if not allow_fallback:
                raise
            retriever, fell_back = BM25Retriever(corpus), True
            warnings.append(f"Retriever '{cfg.retriever}' unavailable ({type(e).__name__}); using BM25.")

    verifier = build_verifier(cfg.verifier)
    if cfg.verifier.lower().startswith("nli"):
        try:
            verifier._load()
        except Exception as e:
            if not allow_fallback:
                raise
            verifier, fell_back = LexicalVerifier(), True
            warnings.append(f"NLI model could not be loaded ({type(e).__name__}: {e}). Using the word-overlap DEMO backend: "
                            "its verdicts are NOT reliable.")

    dep = cfg.deployed or {}
    meta = {
        "retriever": getattr(retriever, "name", None), "verifier": verifier.name, "k": cfg.k, "tau": cfg.tau,
        "evidence_unit": cfg.evidence_unit,
        "validated": bool(cfg.validated and not fell_back),
        "demo_only": verifier.name in ("lexical", "majority"),
        "deployed_run": dep.get("run_name"), "dev_macro_f1": dep.get("dev_macro_f1"),
        "dev_macro_f1_ci95": dep.get("dev_macro_f1_ci95"), "dev_false_support_rate": dep.get("dev_false_support_rate"),
        "overrides": list(cfg.overrides),
    }
    scorer = LexicalSentenceScorer.from_docs(corpus) if corpus else LexicalSentenceScorer()
    return CiteCheckPipeline(verifier, retriever, scorer, cfg.k, cfg.sentences_per_paper, cfg.tau, cfg.margin, meta, warnings,
                             evidence_unit=cfg.evidence_unit)
