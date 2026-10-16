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
An API key (env S2_API_KEY, free from semanticscholar.org/product/api) is optional; without one, requests share a
public rate limit, so the script paces itself, retries rate-limited papers once, and saves after every paper: re-run
the same command to continue where it stopped.
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

from src.data_io import find_scifact_dir, load_claims, load_corpus, load_jsonl

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
    """Semantic Scholar Graph API with polite pacing.  Every call returns (status, data), status one of
    ok | not_found | rate_limited | error, so a 429 is never mistaken for a missing paper."""

    def __init__(self, session=None, api_key=None, sleep=3.0, retries=3):
        import requests

        self.http = session or requests.Session()
        self.headers = {"x-api-key": api_key} if api_key else {}
        self.sleep, self.retries = (0.2 if api_key else sleep), retries

    def _call(self, method, url, **kw):
        status = "error"
        for attempt in range(self.retries + 1):
            time.sleep(self.sleep)
            try:
                r = self.http.request(method, url, headers=self.headers, timeout=30, **kw)
            except Exception:
                status = "error"
                continue
            if r.status_code == 200:
                return "ok", r.json()
            if r.status_code == 404:
                return "not_found", None
            status = "rate_limited" if r.status_code == 429 else "error"
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            return status, None
        return status, None

    def get(self, url, params=None):
        return self._call("GET", url, params=params)

    def post(self, url, params=None, json_body=None):
        return self._call("POST", url, params=params, json=json_body)


def fetch_titles(s2, doc_ids, chunk=500):
    """One batch call per 500 papers: {doc_id: S2 title or None}; raises if the batch endpoint is unavailable."""
    titles = {}
    for i in range(0, len(doc_ids), chunk):
        ids = doc_ids[i : i + chunk]
        status, data = s2.post(f"{API}/batch", {"fields": "title"}, {"ids": [f"CorpusId:{d}" for d in ids]})
        if status != "ok" or not isinstance(data, list):
            raise RuntimeError(f"Semantic Scholar batch lookup failed ({status}); try again in a minute or set S2_API_KEY")
        titles.update({d: (row or {}).get("title") for d, row in zip(ids, data)})
    return titles


def collect(corpus_by_id, doc_ids, s2, n_papers, per_paper, rng, min_chars=40, log=print, done=(), on_rows=None):
    """Sample up to per_paper real citing sentences for each of n_papers cited papers.

    done: cited doc ids already collected (skipped, and counted towards n_papers).  on_rows(rows) is called after
    each paper, so a long run can save as it goes.  Rate-limited papers are retried once at the end.
    """
    out, done = [], set(done)
    todo = [d for d in doc_ids if d not in done][: max(0, 3 * (n_papers - len(done)))]
    titles = fetch_titles(s2, todo) if todo else {}
    reasons, retry, checked = {}, [], 0

    def one(doc_id):
        local = corpus_by_id[doc_id]
        if not titles.get(doc_id):
            return "not in Semantic Scholar"
        if title_match(titles[doc_id], local.title) < 0.5:
            return "title mismatch"
        status, data = s2.get(f"{API}/CorpusId:{doc_id}/citations",
                              {"fields": "contexts,intents,citingPaper.paperId,citingPaper.title,citingPaper.year", "limit": 100})
        if status != "ok":
            return status
        rows = []
        for row in (data or {}).get("data", []):
            cp = row.get("citingPaper") or {}
            for ctx in dict.fromkeys(c.strip() for c in row.get("contexts") or [] if isinstance(c, str)):
                if len(ctx) >= min_chars and cp.get("paperId"):
                    rows.append((cp, ctx, tuple(row.get("intents") or ())))
        if not rows:
            return "no citing sentences available"
        kept = [{
            "pair_id": pair_id(cp["paperId"], doc_id, ctx), "cited_doc_id": doc_id, "cited_title": local.title,
            "citing_paper_id": cp["paperId"], "citing_title": cp.get("title") or "", "citing_year": cp.get("year"),
            "context": ctx, "context_sha1": context_sha1(ctx), "intents": list(intents),
        } for cp, ctx, intents in rng.sample(rows, min(per_paper, len(rows)))]
        out.extend(kept)
        if on_rows:
            on_rows(kept)
        log(f"{doc_id}: {len(rows)} citing sentences, kept {len(kept)}")
        return None

    for pass_ids in (todo, None):
        for doc_id in (pass_ids if pass_ids is not None else list(retry)):
            if len(done | {r["cited_doc_id"] for r in out}) >= n_papers:
                break
            checked += pass_ids is not None
            why = one(doc_id)
            if why == "rate_limited" and pass_ids is not None:
                retry.append(doc_id)
            if why:
                reasons[why] = reasons.get(why, 0) + 1
                log(f"skip {doc_id}: {why.replace('_', ' ')}")
    log("skipped: " + (", ".join(f"{k} {v}" for k, v in reasons.items()) or "none"))
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
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    have = load_jsonl(out) if out.exists() else []  # resume: keep what an earlier run collected
    if have:
        print(f"resuming: {len(have)} pairs from {len({r['cited_doc_id'] for r in have})} papers already in {out}")

    def save(rows):
        with open(out, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)

    rows, checked = collect(by_id, docs, S2(api_key=os.environ.get("S2_API_KEY")), a.n_papers, a.per_paper, rng,
                            done={r["cited_doc_id"] for r in have}, on_rows=save)
    total = have + rows
    print(f"[saved] {out}: {len(total)} pairs from {len({r['cited_doc_id'] for r in total})} cited papers "
          f"({checked} papers tried this run, {len(docs)} eligible)")


if __name__ == "__main__":
    main()
