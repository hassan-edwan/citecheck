"""Score the citation audit on REAL citing sentences (data/realcite).   Owner: Modeling.

    python -m eval.run_realcite --name realcite_audit --model <verifier> \\
        --claim-extractor Qwen/Qwen3-4B                      # also runs the audit WITH LLM claim extraction

Compares, on the same labelled pairs and the same verifier and tau:
    A  whole_sentence   the cleaned citing sentence is the claim (the Session 5 audit)
    B  extracted        a local LLM splits the sentence into the claims it attributes to the cited paper (src/claims.py)

Gold labels (data/realcite/labels.jsonl, one per pair, see data/realcite/README.md):
    SUPPORT | PARTIAL | CONTRADICT | NEI | NOT_A_CLAIM (the paper is cited for a method/tool, nothing to verify)
Scored classes: SUPPORT, PARTIAL, CONTRADICT, NEI, with NOT_A_CLAIM counted as NEI for the verdict metrics (a
system that refuses to judge a method citation is right).  false-SUPPORT = predicted SUPPORT, gold anything else.

Results are flagged `labels_unreviewed` while any label has no human reviewer: draft labels can show whether the
plumbing works, never that one system is better.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from eval import metrics as M
from src.data_io import find_scifact_dir, load_corpus, load_jsonl
from src.schema import CONTRADICT, NEI, PARTIAL, SUPPORT, Paper, Reference

CLASSES = (SUPPORT, PARTIAL, CONTRADICT, NEI)
GOLD = CLASSES + ("NOT_A_CLAIM",)


def load_pairs(labels_path, raw_path):
    """Join labels with the locally collected citing sentences; refuse silently changed text."""
    raw = {r["pair_id"]: r for r in load_jsonl(raw_path)}
    pairs, missing = [], []
    for lab in load_jsonl(labels_path):
        if lab["label"] not in GOLD:
            raise ValueError(f"{lab['pair_id']}: unknown label {lab['label']!r} ({' | '.join(GOLD)})")
        r = raw.get(lab["pair_id"])
        if r is None:
            missing.append(lab["pair_id"])
            continue
        if r["context_sha1"] != lab["context_sha1"]:
            raise ValueError(f"{lab['pair_id']}: the citing sentence differs from the one that was labelled")
        pairs.append({**r, **lab})
    return pairs, missing


def gold_class(label: str) -> str:
    return NEI if label == "NOT_A_CLAIM" else label


def predict(pipe, pairs, corpus_by_id):
    refs = []
    for p in pairs:
        doc = corpus_by_id[p["cited_doc_id"]]
        refs.append(Reference(Paper(f"scifact:{doc.doc_id}", doc.title, " ".join(doc.sentences)),
                              context=p["context"], contexts=(p["context"],), intents=tuple(p.get("intents") or ())))
    rows = pipe.audit(refs, max_items=len(refs))
    preds = []
    for p, row in zip(pairs, rows):
        label = row.verdict.label if row.verdict else NEI
        preds.append({
            "pair_id": p["pair_id"], "gold": p["label"], "pred": label, "skipped": row.verdict is None,
            "skipped_reason": row.skipped_reason,
            # claim text is a (near-)copy of the citing sentence, so only its status and verdict are kept
            "claims": [{"status": c.status, "role": c.role, "verdict": c.verdict.label if c.verdict else None}
                       for c in row.claims],
        })
    return preds


def score(preds, n_boot=1000, seed=0):
    yt, yp = [gold_class(p["gold"]) for p in preds], [p["pred"] for p in preds]
    f1 = lambda t, p: M.macro_f1(t, p, labels=CLASSES)  # noqa: E731
    not_a_claim = [p for p in preds if p["gold"] == "NOT_A_CLAIM"]
    return {
        "n": len(preds), "macro_f1": f1(yt, yp), "macro_f1_ci95": M.bootstrap_ci(yt, yp, f1, n_boot=n_boot, seed=seed),
        "accuracy": M.accuracy(yt, yp), "per_class": M.prf_per_class(yt, yp, CLASSES),
        "confusion_rows_gold_cols_pred": {"labels": list(CLASSES), "matrix": M.confusion_matrix(yt, yp, CLASSES)},
        "false_support_rate": M.false_support_rate(yt, yp),
        "method_citations_skipped": (sum(p["skipped"] for p in not_a_claim) / len(not_a_claim)) if not_a_claim else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--labels", default="data/realcite/labels.jsonl")
    ap.add_argument("--raw", default="data/realcite/raw/candidates.jsonl")
    ap.add_argument("--data-dir", default="data/scifact")
    ap.add_argument("--model", default=None, help="NLI verifier (default: the deployed one)")
    ap.add_argument("--tau", type=float, default=None, help="default: the deployed tau")
    ap.add_argument("--claim-extractor", default=None, help="local LLM for system B; omit to score only system A")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/results")
    a = ap.parse_args(argv)

    from src.claims import ClaimExtractor
    from src.config import load_config
    from src.evidence import LexicalSentenceScorer
    from src.nli import build_verifier
    from src.pipeline import CiteCheckPipeline
    from src.reasoning import LocalLLM

    data_dir = find_scifact_dir(a.data_dir)
    if data_dir is None:
        sys.exit("SciFact not found. Run: python -m data.fetch_scifact")
    if not Path(a.raw).exists():
        sys.exit(f"{a.raw} not found. Run: python -m tools.collect_citations (needs internet)")
    corpus = load_corpus(data_dir / "corpus.jsonl")
    by_id = {d.doc_id: d for d in corpus}
    pairs, missing = load_pairs(a.labels, a.raw)
    if missing:
        print(f"[warn] {len(missing)} labelled pairs are not in {a.raw}; re-run tools.collect_citations with the same seed")
    if not pairs:
        sys.exit("No labelled pairs to score.")

    cfg = load_config()
    spec = f"nli:{a.model}" if a.model else cfg.verifier
    verifier = build_verifier(spec, **({"device": a.device} if spec.lower().startswith("nli") else {}))
    tau = a.tau if a.tau is not None else cfg.tau
    pipe = CiteCheckPipeline(verifier, scorer=LexicalSentenceScorer.from_docs(corpus), k=1, tau=tau)

    systems = {"whole_sentence": predict(pipe, pairs, by_id)}
    if a.claim_extractor:
        pipe.claim_extractor = ClaimExtractor(LocalLLM(a.claim_extractor, device=a.device), verifier=verifier)
        systems["extracted"] = predict(pipe, pairs, by_id)

    unreviewed = sum(1 for p in pairs if not p.get("reviewed_by"))
    results = {
        "name": a.name, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verifier": verifier.name, "tau": tau, "claim_extractor": a.claim_extractor,
        "n_pairs": len(pairs), "gold_distribution": dict(Counter(p["label"] for p in pairs)),
        "labels_unreviewed": unreviewed, "warnings": pipe.warnings,
        "systems": {k: score(v, a.n_boot, a.seed) for k, v in systems.items()},
    }
    if len(systems) == 2:
        ids = [p["pair_id"] for p in systems["whole_sentence"]]
        b, base = ({p["pair_id"]: p for p in systems[k]} for k in ("extracted", "whole_sentence"))
        f1 = lambda t, p: M.macro_f1(t, p, labels=CLASSES)  # noqa: E731
        cmp_ = M.paired_bootstrap_diff([gold_class(b[i]["gold"]) for i in ids], [b[i]["pred"] for i in ids],
                                       [base[i]["pred"] for i in ids], f1, a.n_boot, a.seed)
        results["extracted_vs_whole_sentence"] = cmp_
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:  # no citing text: it is not ours to publish
        for k, v in systems.items():
            for p in v:
                f.write(json.dumps({"system": k, **p}) + "\n")
    flag = f"  ** {unreviewed} of {len(pairs)} labels not yet reviewed by a person: do not quote these numbers **" if unreviewed else ""
    for k, s in results["systems"].items():
        print(f"{k:15s} macro-F1 {s['macro_f1']:.3f} {[round(x, 3) for x in s['macro_f1_ci95']]}  "
              f"false-SUPPORT {s['false_support_rate']:.3f}  (n={s['n']})")
    if "extracted_vs_whole_sentence" in results:
        c = results["extracted_vs_whole_sentence"]
        print(f"extracted - whole_sentence: {c['diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
    print(f"[saved] {out}{flag}")
    return results


if __name__ == "__main__":
    main()
