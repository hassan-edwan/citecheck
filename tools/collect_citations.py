"""Collect REAL citing sentences for the real-citation evaluation set.   Owner: Modeling.

    python -m tools.collect_citations --n-papers 40 --per-paper 2      # needs internet (Colab is fine)

Where the data comes from: the SciFact corpus ids are Semantic Scholar corpus ids, so for papers already in our local
corpus we can ask Semantic Scholar for the sentences in OTHER papers that cite them.  Each candidate is a
(real citing sentence, cited paper) pair whose cited abstract we already have locally, so nothing but the citing
sentence is downloaded.  Cited papers that any TRAIN claim uses are excluded, so the fine-tuned verifier and
selector have never been trained on them.  Each cited paper's title is checked against the corpus title before its
citations are used.

Output: data/realcite/raw/candidates.jsonl (git-ignored: the citing sentences are Semantic Scholar text, which we do
not redistribute; see data/realcite/README.md).  Labels live in data/realcite/labels.jsonl, keyed by pair_id.
An API key (env S2_API_KEY) is optional; without one the script waits between calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

from src.data_io import find_scifact_dir, load_claims, load_corpus

API = "https://api.semanticscholar.org/graph/v1/paper"


def pair_id(citing_id: str, cited_doc_id: str, context: str) -> str:
    return hashlib.sha1(f"{citing_id}|{cited_doc_id}|{context}".encode()).hexdigest()[:12]


def context_sha1(context: str) -> str:
    return hashlib.sha1(context.encode()).hexdigest()


def title_match(a: str, b: str) -> float:
    ta, tb = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


def eligible_docs(corpus, train_claims) -> list:
    used = {d for c in train_claims for d in list(c.gold_docs) + list(c.cited_doc_ids)}
    return sorted(d.doc_id for d in corpus if d.doc_id not in used and d.sentences)


class S2:
    def __init__(self, session=None, api_key=None, sleep=1.1, retries=4):
        import requests

        self.http = session or requests.Session()
        self.headers = {"x-api-key": api_key} if api_key else {}
        self.sleep, self.retries = (0.15 if api_key else sleep), retries

    def get(self, url, params=None):
        for attempt in range(self.retries + 1):
            time.sleep(self.sleep)
            r = self.http.get(url, params=params, headers=self.headers, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt * 2)
                continue
            return r.json() if r.status_code == 200 else None
        return None


def collect(corpus_by_id, doc_ids, s2, n_papers, per_paper, rng, min_chars=40, log=print):
    out, checked = [], 0
    for doc_id in doc_ids:
        if len({c["cited_doc_id"] for c in out}) >= n_papers:
            break
        checked += 1
        meta = s2.get(f"{API}/CorpusId:{doc_id}", {"fields": "title"})
        local = corpus_by_id[doc_id]
        if not meta or title_match(meta.get("title", ""), local.title) < 0.5:
            log(f"skip {doc_id}: title mismatch or not found")
            continue
        data = s2.get(f"{API}/CorpusId:{doc_id}/citations",
                      {"fields": "contexts,intents,citingPaper.paperId,citingPaper.title,citingPaper.year", "limit": 100})
        rows = []
        for row in (data or {}).get("data", []):
            cp = row.get("citingPaper") or {}
            for ctx in dict.fromkeys(c.strip() for c in row.get("contexts") or [] if isinstance(c, str)):
                if len(ctx) >= min_chars and cp.get("paperId"):
                    rows.append((cp, ctx, tuple(row.get("intents") or ())))
        if not rows:
            log(f"skip {doc_id}: no citing sentences available")
            continue
        for cp, ctx, intents in rng.sample(rows, min(per_paper, len(rows))):
            out.append({
                "pair_id": pair_id(cp["paperId"], doc_id, ctx), "cited_doc_id": doc_id, "cited_title": local.title,
                "citing_paper_id": cp["paperId"], "citing_title": cp.get("title") or "", "citing_year": cp.get("year"),
                "context": ctx, "context_sha1": context_sha1(ctx), "intents": list(intents),
            })
        log(f"{doc_id}: {len(rows)} citing sentences, kept {min(per_paper, len(rows))}")
    return out, checked


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/scifact")
    ap.add_argument("--n-papers", type=int, default=40)
    ap.add_argument("--per-paper", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/realcite/raw/candidates.jsonl")
    a = ap.parse_args(argv)

    data_dir = find_scifact_dir(a.data_dir)
    if data_dir is None:
        raise SystemExit("SciFact not found. Run: python -m data.fetch_scifact")
    corpus = load_corpus(data_dir / "corpus.jsonl")
    by_id = {d.doc_id: d for d in corpus}
    rng = random.Random(a.seed)
    docs = eligible_docs(corpus, load_claims(data_dir / "claims_train.jsonl"))
    rng.shuffle(docs)
    rows, checked = collect(by_id, docs, S2(api_key=os.environ.get("S2_API_KEY")), a.n_papers, a.per_paper, rng)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"[saved] {out}: {len(rows)} pairs from {len({r['cited_doc_id'] for r in rows})} cited papers "
          f"({checked} papers tried, {len(docs)} eligible)")


if __name__ == "__main__":
    main()
