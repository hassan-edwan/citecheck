"""Shared data structures and interfaces for CiteCheck.

Owner: Engineering (Sakshaat).  Everyone imports from here, so change it only
through a reviewed PR and keep it backwards compatible.

Label vocabulary
----------------
Internally we use SciFact-style labels (SUPPORT / CONTRADICT / NEI).  What a
user sees is the wording from our lean canvas (see DISPLAY).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

SUPPORT = "SUPPORT"
CONTRADICT = "CONTRADICT"
NEI = "NEI"
LABELS = (SUPPORT, CONTRADICT, NEI)
PARTIAL = "PARTIAL"  # citation audit only: some parts of a citing sentence are supported, the rest are not shown

DISPLAY = {
    SUPPORT: "SUPPORTS",
    CONTRADICT: "CONTRADICTS",
    NEI: "NOT ENOUGH EVIDENCE",
    PARTIAL: "PARTLY SUPPORTED",
}


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Doc:
    """One retrievable paper (title + abstract split into sentences)."""

    doc_id: str
    title: str
    sentences: tuple = ()
    url: Optional[str] = None
    source: str = "corpus"  # corpus | doi | arxiv | pasted | s2

    @property
    def abstract(self) -> str:
        return " ".join(self.sentences)

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract}".strip()


@dataclass(frozen=True)
class Paper:
    """A paper as fetched from the outside world (DOI / arXiv / pasted)."""

    paper_id: str
    title: str = ""
    abstract: str = ""
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    source: str = "unknown"

    def to_doc(self) -> Doc:
        from .text_utils import split_sentences  # local import: avoid cycles

        return Doc(
            doc_id=self.paper_id,
            title=self.title,
            sentences=tuple(split_sentences(self.abstract)),
            url=self.url,
            source=self.source,
        )


@dataclass(frozen=True)
class Reference:
    """One outgoing citation of a paper, with the citing sentence if known."""

    cited: Paper
    context: str = ""  # the sentence(s) in the citing paper around the citation (the first, when there are several)
    intents: tuple = ()
    contexts: tuple = ()  # EVERY citing sentence Semantic Scholar returned for this reference

    @property
    def all_contexts(self) -> tuple:
        return self.contexts or ((self.context,) if self.context.strip() else ())


# --------------------------------------------------------------------------- #
# Model outputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Probs:
    """NLI-style class probabilities for one (evidence, claim) pair."""

    support: float
    neutral: float
    contradict: float

    def as_dict(self) -> dict:
        return {"support": self.support, "neutral": self.neutral, "contradict": self.contradict}

    @property
    def label(self) -> str:
        best = max(
            ((self.support, SUPPORT), (self.contradict, CONTRADICT), (self.neutral, NEI)),
            key=lambda t: t[0],
        )
        return best[1]


@dataclass(frozen=True)
class Hit:
    doc: Doc
    score: float
    rank: int  # 1-based


@dataclass
class EvidenceItem:
    """The evidence a verdict is based on: one paper + the sentences we used."""

    doc_id: str
    title: str
    url: Optional[str]
    sentence_idxs: list
    sentences: list
    probs: Probs
    retrieval_score: Optional[float] = None
    retrieval_rank: Optional[int] = None


@dataclass
class Verdict:
    claim: str
    label: str  # SUPPORT | CONTRADICT | NEI  (internal vocabulary)
    confidence: float  # probability of the predicted label's class (0..1)
    evidence: list = field(default_factory=list)  # list[EvidenceItem], best first
    abstained: bool = False  # True when NEI came from the confidence threshold
    note: str = ""
    latency_ms: float = 0.0
    backend: dict = field(default_factory=dict)

    @property
    def display_label(self) -> str:
        return DISPLAY[self.label]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["display_label"] = self.display_label
        return d


@dataclass
class RelevanceResult:
    score: float  # 0..1
    band: str  # "likely relevant" | "maybe relevant" | "likely not relevant"
    matched_terms: list = field(default_factory=list)
    method: str = "lexical"
    note: str = ""


@dataclass
class ClaimCheck:
    """One claim checked for a citation: the citing sentence it came from and the verdict on it."""

    context: str  # the citing sentence
    claim: str  # what was checked against the cited paper
    source_quote: str = ""  # the exact words of the citing sentence the claim restates
    role: str = "result"  # background | method | result | comparison | other
    verdict: Optional[Verdict] = None
    status: str = "checked"  # checked | needs_more_context | unfaithful | not_attributed
    note: str = ""


@dataclass
class AuditRow:
    reference: Reference
    verdict: Optional[Verdict]  # the reference's overall verdict (may be PARTIAL)
    skipped_reason: str = ""
    claims: list = field(default_factory=list)  # list[ClaimCheck]


# --------------------------------------------------------------------------- #
# Interfaces (structural typing: any class with these methods works)
# --------------------------------------------------------------------------- #
@runtime_checkable
class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10) -> list:  # list[Hit]
        ...


@runtime_checkable
class Verifier(Protocol):
    """Judges whether a premise supports / contradicts / is neutral to a hypothesis."""

    name: str

    def predict_pairs(self, pairs: Sequence) -> list:  # pairs: [(premise, hypothesis)] -> list[Probs]
        ...


@runtime_checkable
class SentenceScorer(Protocol):
    def score(self, claim: str, sentences: Sequence) -> list:  # list[float]
        ...
