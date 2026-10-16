"""Runtime configuration.  Owner: Engineering (Sakshaat).

The product must run the SAME configuration we report numbers for (course rule).  Priority:

    environment variable  >  eval/results/deployed_config.json (written by `eval.run_eval --promote`)  >  defaults

`Config.validated` is True only when nothing model-related was overridden, and the app shows the
dev score of the deployed run next to every verdict.  Anything else is labelled UNVALIDATED.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEPLOYED_PATH = Path("eval/results/deployed_config.json")

_MODEL_FIELDS = ("retriever", "embed_model", "verifier", "k", "sentences_per_paper", "evidence_unit", "sentence_scorer",
                 "tau", "margin")


@dataclass
class Config:
    retriever: str = "bm25"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    verifier: str = "nli"
    k: int = 3
    sentences_per_paper: int = 3
    sentence_scorer: str = "lexical"  # lexical | cross:<model> (src/evidence.build_sentence_scorer)
    evidence_unit: str = "window"  # window | abstract | sentence (src/evidence.py)
    tau: float = 0.5
    margin: float = 0.0
    data_dir: str = "data/scifact"
    citation_gate: bool = True  # skip citing "sentences" that are not claims (src/citation_gate.py); audit only
    audit_tau: Optional[float] = None  # the audit's own abstain threshold, tuned by eval/run_realcite.py --tune-tau
    claim_extractor: Optional[str] = None  # local LLM for the citation audit (src/claims.py); audit-only, experimental
    validated: bool = False
    deployed: Optional[dict] = None  # the raw deployed_config.json, when present
    overrides: list = field(default_factory=list)  # names of env vars that changed model behaviour

    def describe(self) -> dict:
        return {f: getattr(self, f) for f in _MODEL_FIELDS}


def load_deployed(path: Path = DEPLOYED_PATH) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _env(name: str, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else None


def load_config(deployed_path: Path = DEPLOYED_PATH) -> Config:
    cfg = Config()
    dep = load_deployed(deployed_path)
    if dep:
        for f in _MODEL_FIELDS:
            if f in dep:
                setattr(cfg, f, dep[f])
        cfg.deployed, cfg.validated = dep, True
    env_map = {
        "CITECHECK_RETRIEVER": ("retriever", str),
        "CITECHECK_VERIFIER": ("verifier", str),
        "CITECHECK_EMBED_MODEL": ("embed_model", str),
        "CITECHECK_K": ("k", int),
        "CITECHECK_EVIDENCE_UNIT": ("evidence_unit", str),
        "CITECHECK_SENTENCE_SCORER": ("sentence_scorer", str),
        "CITECHECK_TAU": ("tau", float),
        "CITECHECK_MARGIN": ("margin", float),
    }
    for var, (attr, cast) in env_map.items():
        v = _env(var, cast)
        if v is not None and v != getattr(cfg, attr):
            setattr(cfg, attr, v)
            cfg.overrides.append(var)
            cfg.validated = False
    cfg.data_dir = _env("CITECHECK_DATA_DIR") or cfg.data_dir
    cfg.claim_extractor = _env("CITECHECK_CLAIM_EXTRACTOR") or (dep or {}).get("claim_extractor") or None
    gate = _env("CITECHECK_CITATION_GATE")
    cfg.citation_gate = (gate.lower() not in ("0", "false", "off")) if gate else bool((dep or {}).get("citation_gate", True))
    cfg.audit_tau = _env("CITECHECK_AUDIT_TAU", float) or (dep or {}).get("audit_tau")
    return cfg
