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
from .evidence import LexicalSentenceScorer, build_sentence_scorer, score_claim_docs
from .ingest import clean_context
from .citation_gate import assess_context
from .claims import aggregate
from .schema import NEI, SUPPORT, AuditRow, ClaimCheck, EvidenceItem, Hit, Paper, Reference, Verdict
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
        self.claim_extractor = None  # src/claims.ClaimExtractor; None = check whole citing sentences
        self.citation_gate = True  # src/citation_gate: skip contexts that are not claims before verifying
        self.audit_tau = None  # abstain threshold for the citation audit; None = the claim checker's tau
        self.meta = meta or {}
        self.warnings = warnings or []

    @property
    def has_corpus(self) -> bool:
        return self.retriever is not None

    # ------------------------------------------------------------------------------------------
    def verify_claim(self, claim: str, docs: Optional[Sequence] = None, tau: Optional[float] = None) -> Verdict:
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
        dec = decide([s.probs for s in scored], tau=self.tau if tau is None else tau, margin=self.margin)

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

    def verify_against_paper(self, claim: str, paper: Paper, tau: Optional[float] = None) -> Verdict:
        """The core use case: 'I cite this paper for this claim. Does it hold up?'"""
        doc = paper.to_doc()
        if not doc.sentences:
            raise ValueError("That paper has no abstract text to check against.")
        return self.verify_claim(claim, docs=[doc], tau=tau)

    # ------------------------------------------------------------------------------------------
    def audit(self, references: Sequence, max_items: int = 10, progress: Optional[Callable] = None) -> list:
        """Check each cited paper against EVERY sentence that cites it.  EXPERIMENTAL: see eval/run_realcite.py.

        With a claim extractor (src/claims.py) each citing sentence is first split into the claims it attributes to
        the cited paper; without one, the whole cleaned sentence is the claim (the Session 5 behaviour).
        """
        refs = list(references)[:max_items]
        gated, todo = {}, []
        for i, ref in enumerate(refs):
            if not ref.cited.abstract.strip():
                continue
            for ctx in ref.all_contexts:
                g = assess_context(ctx) if self.citation_gate else None
                if g is not None and not g.usable:
                    gated.setdefault(i, []).append(ClaimCheck(ctx, "", status="not_a_claim", note=g.reason))
                else:
                    todo.append((i, ctx))
        extracted = [None] * len(todo)
        if self.claim_extractor is not None and todo:
            try:
                extracted = self.claim_extractor.extract([(c, refs[i].cited.title, refs[i].intents) for i, c in todo])
            except Exception as e:  # a broken local LLM must not take the audit down
                self.warnings.append(f"Claim extraction unavailable ({type(e).__name__}); checked whole sentences.")
        per_ref = {i: list(v) for i, v in gated.items()}
        for (i, ctx), ex in zip(todo, extracted):
            if ex is None:
                note = "claim extraction failed; checked the whole sentence" if self.claim_extractor is not None else ""
                checks = [ClaimCheck(ctx, clean_context(ctx), ctx, note=note)]
            elif not ex:
                checks = [ClaimCheck(ctx, "", status="not_attributed", note="no factual claim is attributed to this citation")]
            else:
                checks = [ClaimCheck(ctx, c.claim, c.source_quote, c.role, status=c.status, note=c.note) for c in ex]
            per_ref.setdefault(i, []).extend(checks)

        rows = []
        for n, ref in enumerate(refs, 1):
            if progress:
                progress(n, len(refs))
            if not ref.all_contexts:
                rows.append(AuditRow(ref, None, "no citing sentence available for this reference"))
                continue
            if not ref.cited.abstract.strip():
                rows.append(AuditRow(ref, None, "cited paper has no abstract available"))
                continue
            checks = per_ref.get(n - 1, [])
            for c in checks:
                if c.status == "checked":
                    c.verdict = self.verify_against_paper(c.claim, ref.cited, tau=self.audit_tau)
            checked = [c for c in checks if c.verdict is not None]
            if not checked:
                rows.append(AuditRow(ref, None, "no claim attributed to this citation could be checked", checks))
                continue
            rows.append(AuditRow(ref, combine_claim_verdicts(checked), claims=checks))
        return rows


def combine_claim_verdicts(checked: Sequence) -> Verdict:
    """Per-claim verdicts of one reference -> one Verdict (a single claim's verdict is returned unchanged)."""
    if len(checked) == 1:
        return checked[0].verdict
    verdicts = [c.verdict for c in checked]
    label = aggregate([v.label for v in verdicts])
    lead = next((v for v in verdicts if v.label == label), None) or next(
        (v for v in verdicts if v.label == SUPPORT), verdicts[0])  # PARTIAL: show the supported claim's evidence
    n_sup = sum(v.label == SUPPORT for v in verdicts)
    return Verdict(
        claim=" | ".join(v.claim for v in verdicts), label=label,
        confidence=min(v.confidence for v in verdicts if v.label == lead.label), evidence=lead.evidence,
        abstained=label == NEI and all(v.abstained for v in verdicts),
        note=f"{n_sup} of {len(verdicts)} claims supported", backend=lead.backend,
    )


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
    if cfg.sentence_scorer != "lexical":
        try:
            scorer = build_sentence_scorer(cfg.sentence_scorer)
            scorer._load()
        except Exception as e:
            if not allow_fallback:
                raise
            scorer = LexicalSentenceScorer.from_docs(corpus) if corpus else LexicalSentenceScorer()
            meta["validated"] = False
            warnings.append(f"Sentence selector '{cfg.sentence_scorer}' could not be loaded ({type(e).__name__}); "
                            "using word overlap, so the reported score does not apply.")
    meta["sentence_scorer"] = getattr(scorer, "name", "lexical")
    pipe = CiteCheckPipeline(verifier, retriever, scorer, cfg.k, cfg.sentences_per_paper, cfg.tau, cfg.margin, meta, warnings,
                             evidence_unit=cfg.evidence_unit)
    pipe.citation_gate, pipe.audit_tau = cfg.citation_gate, cfg.audit_tau
    meta["audit_tau"] = cfg.audit_tau
    if cfg.claim_extractor:  # loads lazily on the first audit; a failure falls back to whole sentences, with a warning
        from .claims import ClaimExtractor
        from .reasoning import LocalLLM

        pipe.claim_extractor = ClaimExtractor(LocalLLM(cfg.claim_extractor), verifier=verifier)
        meta["claim_extractor"] = cfg.claim_extractor
    return pipe
