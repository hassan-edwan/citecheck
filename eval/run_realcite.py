"""Score the citation audit on REAL citing sentences (data/realcite), and tune its gate-and-threshold stage.   Owner: Modeling.

    python -m eval.run_realcite --name realcite_audit --model <verifier> --tune-tau \
        --claim-extractor Qwen/Qwen3-4B                      # optional: also the audit WITH LLM claim extraction

The 80 pairs are split by CITED PAPER (realcite_split, fixed hash) into a tune half and a test half.  Anything
learned from the labels is learned on the tune half only and reported on the test half:
    * the citation-context gate (src/citation_gate.py): its rules were written from tune-half examples
    * the audit's own abstain threshold (--tune-tau): best macro-F1 on the tune half with false-SUPPORT <= --fs-cap

Systems, same verifier:
    baseline         whole cleaned citing sentence is the claim, no gate (the Session 5 audit)
    gated            + citation-context gate
    gated_extracted  + gate + LLM claim extraction (src/claims.py), with --claim-extractor

Gold labels: SUPPORT | PARTIAL | CONTRADICT | NEI | NOT_A_CLAIM, scored as 4 classes with NOT_A_CLAIM counted as NEI (a
system that declines to judge a non-claim is right).  false-SUPPORT = predicted SUPPORT, gold anything else.

Results are flagged `labels_unreviewed` while any label has no human reviewer, and --promote-audit refuses them unless
--allow-unreviewed, which is recorded in deployed_config.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from eval import metrics as M
from src.data_io import find_scifact_dir, load_corpus, load_jsonl
from src.schema import CONTRADICT, NEI, PARTIAL, SUPPORT, Paper, Probs, Reference

TAU_GRID = [round(0.05 * i, 2) for i in range(1, 20)]

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


def realcite_split(cited_doc_ids, salt="realcite-v1") -> dict:
    """{cited_doc_id: "tune" | "test"}: half of the cited papers each, by a fixed hash (both pairs of a paper together)."""
    docs = sorted(set(cited_doc_ids), key=lambda d: hashlib.sha1(f"{salt}:{d}".encode()).hexdigest())
    return {d: "tune" if i < len(docs) // 2 else "test" for i, d in enumerate(docs)}


def predict(pipe, pairs, corpus_by_id):
    """Run the audit once (abstaining nowhere: audit_tau=0) and keep each claim's probabilities, so any threshold
    can be applied afterwards by relabel() without calling the verifier again."""
    refs = []
    for p in pairs:
        doc = corpus_by_id[p["cited_doc_id"]]
        refs.append(Reference(Paper(f"scifact:{doc.doc_id}", doc.title, " ".join(doc.sentences)),
                              context=p["context"], contexts=(p["context"],), intents=tuple(p.get("intents") or ())))
    keep, pipe.audit_tau = pipe.audit_tau, 0.0
    try:
        rows = pipe.audit(refs, max_items=len(refs))
    finally:
        pipe.audit_tau = keep
    preds = []
    for p, row in zip(pairs, rows):
        preds.append({
            "pair_id": p["pair_id"], "gold": p["label"], "skipped_reason": row.skipped_reason,
            # claim text is a (near-)copy of the citing sentence, so only status, role and probabilities are kept
            "claims": [{"status": c.status, "role": c.role,
                        "probs": c.verdict.evidence[0].probs.as_dict() if c.verdict and c.verdict.evidence else None}
                       for c in row.claims],
        })
    return preds


def relabel(preds, tau, margin=0.0):
    """Predictions at abstain threshold tau: per claim decide(), then the audit's aggregation."""
    from src.claims import aggregate
    from src.verdict import decide

    out = []
    for p in preds:
        labels = [decide([Probs(**c["probs"])], tau=tau, margin=margin).label for c in p["claims"] if c["probs"]]
        label = aggregate(labels)
        out.append({**p, "pred": label or NEI, "skipped": label is None, "tau": tau})
    return out


def tune_tau(preds, fs_cap):
    """Best macro-F1 with false-SUPPORT <= fs_cap (ties: larger tau); if no tau meets the cap, the lowest
    false-SUPPORT (ties: higher F1, then larger tau)."""
    rows = []
    for tau in TAU_GRID:
        s = score(relabel(preds, tau), n_boot=0)
        rows.append({"tau": tau, "macro_f1": s["macro_f1"], "false_support_rate": s["false_support_rate"]})
    ok = [r for r in rows if r["false_support_rate"] <= fs_cap]
    best = (max(ok, key=lambda r: (r["macro_f1"], r["tau"])) if ok
            else min(rows, key=lambda r: (r["false_support_rate"], -r["macro_f1"], -r["tau"])))
    return {**best, "met_cap": bool(ok), "fs_cap": fs_cap, "grid": rows}


def gate_report(pairs, split):
    """How well the citation-context gate separates NOT_A_CLAIM from real claims, per half (no model involved)."""
    from src.citation_gate import assess_context

    out = {}
    for half in ("tune", "test"):
        sub = [p for p in pairs if split[p["cited_doc_id"]] == half]
        res = [(assess_context(p["context"]), p["label"] == "NOT_A_CLAIM") for p in sub]
        caught = sum(1 for g, gold in res if not g.usable and gold)
        n_non = sum(1 for _, gold in res if gold)
        wrong = sum(1 for g, gold in res if not g.usable and not gold)
        out[half] = {"n": len(sub), "non_claims": n_non, "caught": caught, "real_claims": len(sub) - n_non,
                     "real_claims_wrongly_skipped": wrong,
                     "rules": dict(Counter(g.rule for g, _ in res if not g.usable))}
    return out


def score(preds, n_boot=1000, seed=0):
    yt, yp = [gold_class(p["gold"]) for p in preds], [p["pred"] for p in preds]
    f1 = lambda t, p: M.macro_f1(t, p, labels=CLASSES)  # noqa: E731
    not_a_claim = [p for p in preds if p["gold"] == "NOT_A_CLAIM"]
    return {
        "n": len(preds), "macro_f1": f1(yt, yp),
        "macro_f1_ci95": M.bootstrap_ci(yt, yp, f1, n_boot=n_boot, seed=seed) if n_boot else None,
        "accuracy": M.accuracy(yt, yp), "per_class": M.prf_per_class(yt, yp, CLASSES),
        "confusion_rows_gold_cols_pred": {"labels": list(CLASSES), "matrix": M.confusion_matrix(yt, yp, CLASSES)},
        "false_support_rate": M.false_support_rate(yt, yp),
        "non_claims_left_unjudged": (sum(p["skipped"] for p in not_a_claim) / len(not_a_claim)) if not_a_claim else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--labels", default="data/realcite/labels.jsonl")
    ap.add_argument("--raw", default="data/realcite/raw/candidates.jsonl")
    ap.add_argument("--data-dir", default="data/scifact")
    ap.add_argument("--model", default=None, help="NLI verifier (default: the deployed one)")
    ap.add_argument("--tau", type=float, default=None, help="audit threshold when not tuning (default: the deployed one)")
    ap.add_argument("--tune-tau", action="store_true", help="choose the audit threshold on the tune half")
    ap.add_argument("--fs-cap", type=float, default=0.10, help="false-SUPPORT ceiling for --tune-tau (tune half)")
    ap.add_argument("--claim-extractor", default=None, help="local LLM for the gated_extracted system")
    ap.add_argument("--promote-audit", action="store_true", help="write the gate and tuned audit tau to deployed_config.json")
    ap.add_argument("--allow-unreviewed", action="store_true", help="let --promote-audit use unreviewed labels (recorded)")
    ap.add_argument("--deployed", default="eval/results/deployed_config.json")
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
    unreviewed = sum(1 for p in pairs if not p.get("reviewed_by"))
    if a.promote_audit and (not a.tune_tau or (unreviewed and not a.allow_unreviewed)):
        sys.exit("--promote-audit needs --tune-tau, and reviewed labels (or --allow-unreviewed, which is recorded).")
    split = realcite_split(p["cited_doc_id"] for p in pairs)

    cfg = load_config(Path(a.deployed))
    spec = f"nli:{a.model}" if a.model else cfg.verifier
    verifier = build_verifier(spec, **({"device": a.device} if spec.lower().startswith("nli") else {}))
    base_tau = a.tau if a.tau is not None else (cfg.audit_tau if cfg.audit_tau is not None else cfg.tau)
    pipe = CiteCheckPipeline(verifier, scorer=LexicalSentenceScorer.from_docs(corpus), k=1, tau=cfg.tau)

    raw = {}
    pipe.citation_gate = False
    raw["baseline"] = predict(pipe, pairs, by_id)
    pipe.citation_gate = True
    raw["gated"] = predict(pipe, pairs, by_id)
    if a.claim_extractor:
        pipe.claim_extractor = ClaimExtractor(LocalLLM(a.claim_extractor, device=a.device), verifier=verifier)
        raw["gated_extracted"] = predict(pipe, pairs, by_id)
    primary = "gated_extracted" if a.claim_extractor else "gated"

    half = lambda preds, h: [p for p in preds if split[p["pair_id_doc"]] == h]  # noqa: E731
    for v in raw.values():
        for p, src in zip(v, pairs):
            p["pair_id_doc"], p["half"] = src["cited_doc_id"], split[src["cited_doc_id"]]

    tuning = tune_tau(half(raw[primary], "tune"), a.fs_cap) if a.tune_tau else None
    audit_tau = tuning["tau"] if tuning else base_tau
    systems, labelled = {}, {}
    for k, v in raw.items():
        tau_k = audit_tau if k == primary else base_tau  # baselines keep the deployed threshold
        labelled[k] = relabel(v, tau_k)
        systems[k] = {"tau": tau_k, "test": score(half(labelled[k], "test"), a.n_boot, a.seed),
                      "tune": score(half(labelled[k], "tune"), a.n_boot, a.seed),
                      "all": score(labelled[k], a.n_boot, a.seed)}

    f1 = lambda t, p: M.macro_f1(t, p, labels=CLASSES)  # noqa: E731
    comparisons = {}
    test_ids = [p["pair_id"] for p in half(labelled["baseline"], "test")]
    for k in [k for k in labelled if k != "baseline"]:
        mine, base = ({p["pair_id"]: p for p in labelled[x]} for x in (k, "baseline"))
        comparisons[f"{k}_vs_baseline_test"] = M.paired_bootstrap_diff(
            [gold_class(mine[i]["gold"]) for i in test_ids], [mine[i]["pred"] for i in test_ids],
            [base[i]["pred"] for i in test_ids], f1, a.n_boot, a.seed)

    results = {
        "name": a.name, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verifier": verifier.name, "deployed_tau": base_tau, "claim_extractor": a.claim_extractor,
        "n_pairs": len(pairs), "split": {"salt": "realcite-v1", "tune_papers": sum(v == "tune" for v in split.values()),
                                         "test_papers": sum(v == "test" for v in split.values())},
        "gold_distribution": dict(Counter(p["label"] for p in pairs)), "labels_unreviewed": unreviewed,
        "gate": gate_report(pairs, split), "tau_tuning": tuning, "primary_system": primary,
        "systems": systems, "comparisons": comparisons, "warnings": pipe.warnings,
    }
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:  # no citing text: it is not ours to publish
        for k, v in labelled.items():
            for p in v:
                f.write(json.dumps({"system": k, **{x: y for x, y in p.items() if x != "pair_id_doc"}}) + "\n")

    flag = f"  ** {unreviewed} of {len(pairs)} labels not yet reviewed by a person: do not quote these numbers **" if unreviewed else ""
    g = results["gate"]["test"]
    print(f"gate (test half): caught {g['caught']}/{g['non_claims']} non-claims, "
          f"wrongly skipped {g['real_claims_wrongly_skipped']}/{g['real_claims']} real claims")
    if tuning:
        print(f"audit tau tuned on the tune half: {tuning['tau']} (false-SUPPORT cap {a.fs_cap}, met: {tuning['met_cap']})")
    for k, sy in systems.items():
        t = sy["test"]
        print(f"{k:16s} tau {sy['tau']:.2f}  TEST macro-F1 {t['macro_f1']:.3f} {[round(x, 3) for x in t['macro_f1_ci95']]}  "
              f"false-SUPPORT {t['false_support_rate']:.3f}  (n={t['n']})")
    for k, c in comparisons.items():
        print(f"{k}: {c['diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
    print(f"[saved] {out}{flag}")

    if a.promote_audit:
        dep_path = Path(a.deployed)
        dep = json.loads(dep_path.read_text(encoding="utf-8")) if dep_path.exists() else {}
        t = systems[primary]["test"]
        dep.update({"citation_gate": True, "audit_tau": audit_tau, "audit_run": a.name,
                    "audit_test_macro_f1": t["macro_f1"], "audit_test_false_support_rate": t["false_support_rate"],
                    "audit_labels_reviewed": unreviewed == 0})
        if a.claim_extractor:
            dep["claim_extractor"] = a.claim_extractor
        dep_path.write_text(json.dumps(dep, indent=2), encoding="utf-8")
        print(f"[promoted] audit settings -> {dep_path}" + ("  (UNREVIEWED labels, recorded)" if unreviewed else ""))
    return results


if __name__ == "__main__":
    main()
