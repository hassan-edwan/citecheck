"""One command that produces every number we report.   Owner: Data & Evaluation (Sahil).

Examples (run from the repo root):

    # floor: always "not enough evidence"
    python -m eval.run_eval --name majority --retriever bm25 --verifier majority

    # simple non-neural baseline
    python -m eval.run_eval --name lexical --retriever bm25 --verifier lexical --compare-with eval/results/majority

    # THE zero-shot baseline (pretrained NLI, no SciFact training)
    python -m eval.run_eval --name zero_shot_nli --retriever bm25 --verifier nli --oracle \\
        --compare-with eval/results/lexical

    # ship the run that the app should use (writes eval/results/deployed_config.json)
    python -m eval.run_eval --name zero_shot_nli --retriever bm25 --verifier nli --promote

Honesty guard-rails built in:
  * tau (the abstain threshold) is tuned on the TRAIN split and reported on DEV; tuning on the
    evaluation split is refused unless you pass --allow-leak (and it is recorded).
  * runs on tests/fixtures/** are flagged synthetic_fixture and can never be promoted.
  * every result file records data hashes, package versions, git commit and all arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eval import metrics as M
from src.data_io import (
    ClaimRecord,
    find_scifact_dir,
    is_synthetic_fixture,
    label_distribution,
    load_claims,
    load_corpus,
    sha256_file,
    split_has_labels,
)
from src.evidence import EVIDENCE_UNITS, LexicalSentenceScorer, score_claim_docs
from src.nli import DEFAULT_NLI_MODEL, build_verifier
from src.retrieval import build_retriever
from src.schema import Hit, LABELS, NEI, Probs
from src.verdict import decide, doc_level_labels

TAU_GRID = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]
RETRIEVAL_DEPTH = 20


def tau_tuning_claims(claims, n, seed=0):
    """The train claims tau is tuned on: a seeded sample of n (all of them when there are <= n).

    eval.finetune_nli calls this too and leaves these claims OUT of training, so a fine-tuned
    model's tau is tuned on claims it has never seen.  Keep the two in sync by changing only this.
    """
    claims = sorted(claims, key=lambda c: c.id)
    if len(claims) <= n:
        return claims
    pick = sorted(np.random.default_rng(seed).choice(len(claims), n, replace=False))
    return [claims[i] for i in pick]


def check_tuning_claims_held_out(model, tune_claims, allow_leak=False):
    """Refuse to tune tau on train claims that a local fine-tuned checkpoint was trained on.

    A Hugging Face id (no local train_log.json) never saw SciFact, so it passes.  Returns how many tuning
    claims the model was trained on (non-zero only with allow_leak).
    """
    log_path = Path(model) / "train_log.json"
    if not log_path.exists():
        return 0
    log = json.loads(log_path.read_text(encoding="utf-8"))
    held_out = set(log.get("heldout_claim_ids") or [])
    seen = [c.id for c in tune_claims if c.id not in held_out]
    if seen and not allow_leak:
        sys.exit(
            f"{len(seen)} of the {len(tune_claims)} tau-tuning claims were used to fine-tune {model}, so tau would be "
            "tuned on training data. Retrain with eval.finetune_nli (it holds these claims out by default, with the "
            "same --tune-n/--seed), or pass --allow-leak to record the leak."
        )
    return len(seen)


# --------------------------------------------------------------------------- #
# Collecting model outputs (the expensive part; cached)
# --------------------------------------------------------------------------- #
def collect(claims, corpus_by_id, retriever, verifier, scorer, k, m, mode="e2e", depth=RETRIEVAL_DEPTH, unit="window"):
    """Run retrieval + verification once and keep everything needed to score at any tau."""
    items, rankings = [], []
    t_ret = 0.0
    for c in claims:
        if mode == "e2e":
            t0 = time.perf_counter()
            hits_all = retriever.search(c.claim, max(depth, k))
            t_ret += time.perf_counter() - t0
            rankings.append([h.doc.doc_id for h in hits_all])
            hits = hits_all[:k]
        else:  # oracle_doc / oracle_rationale: judge the gold papers (or cited papers for NEI claims)
            ids = c.gold_docs or c.cited_doc_ids
            hits = [Hit(corpus_by_id[d], 0.0, i + 1) for i, d in enumerate(ids) if d in corpus_by_id]
            rankings.append(list(ids))
        items.append((c.claim, hits))

    gold_sent = None
    if mode == "oracle_rationale":
        gold_sent = [{d: c.rationale_sentence_idxs(d) for d in c.gold_docs} for c in claims]

    t0 = time.perf_counter()
    scored = score_claim_docs(items, verifier, scorer, m=m, gold_sentences=gold_sent, unit=unit)
    t_ver = time.perf_counter() - t0

    outputs = []
    for c, ranking, row in zip(claims, rankings, scored):
        outputs.append(
            {
                "id": c.id,
                "claim": c.claim,
                "gold_label": c.gold_label,
                "gold_docs": {d: {"label": lab, "rationales": c.rationale_sets(d)} for d, lab in c.gold_doc_labels.items()},
                "cited_doc_ids": c.cited_doc_ids,
                "ranking": ranking,
                "docs": [
                    {
                        "doc_id": s.hit.doc.doc_id,
                        "rank": s.hit.rank,
                        "retrieval_score": s.hit.score,
                        "sentence_idxs": s.sentence_idxs,
                        "probs": s.probs.as_dict(),
                    }
                    for s in row
                ],
            }
        )
    n = max(len(claims), 1)
    return {"outputs": outputs, "timing": {"retrieval_ms_per_claim": 1000 * t_ret / n, "verify_ms_per_claim": 1000 * t_ver / n}}


def _cache_path(spec: dict) -> Path:
    key = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    return Path("eval/cache") / f"collect_{key}.json"


def _local_train_log(spec):
    """train_log.json of a local fine-tuned checkpoint named by an nli:<path> spec, if there is one."""
    path = Path(spec[4:]) / "train_log.json" if spec.lower().startswith("nli:") else None
    return path if path and path.is_file() else None


def _local_model_fingerprint(model):
    """sha256 prefix of a local checkpoint's weights (None for a Hugging Face id)."""
    if not model or not Path(model).is_dir():
        return None
    weights = sorted(Path(model).glob("*.safetensors")) or sorted(Path(model).glob("*.bin"))
    h = hashlib.sha256()
    for w in weights:
        h.update(sha256_file(w).encode())
    return h.hexdigest()[:16] if weights else None


def collect_cached(spec, use_cache, **kw):
    path = _cache_path(spec)
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    res = collect(**kw)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res), encoding="utf-8")
    return res


# --------------------------------------------------------------------------- #
# Scoring collected outputs at a given tau
# --------------------------------------------------------------------------- #
def _probs(doc):
    return Probs(**doc["probs"])


def predict(outputs, tau, margin=0.0):
    preds = []
    for o in outputs:
        dp = [_probs(d) for d in o["docs"]]
        dec = decide(dp, tau=tau, margin=margin)
        preds.append(
            {
                "id": o["id"],
                "gold": o["gold_label"],
                "pred": dec.label,
                "confidence": dec.confidence,
                "abstained": dec.abstained,
                "note": dec.note,
                "driver_doc": o["docs"][dec.doc_index]["doc_id"] if dec.doc_index is not None else None,
            }
        )
    return preds


ESCALATE = ("not-confident", "all")
CONFIDENT = 0.98  # stage-1 verdicts at least this strong are kept (set before any stage-2 result was seen)


def needs_reasoning(pred, output, policy):
    """Stage-1 claims passed to the stage-2 reasoner.  not-confident: stage 1 abstained, or its strongest paper
    is below CONFIDENT (the 0.9-1.0 band is the only one where finetuned_nli is right more than ~2/3 of the time)."""
    if not output["docs"]:
        return False
    if policy == "all":
        return True
    strength = max(max(d["probs"]["support"], d["probs"]["contradict"]) for d in output["docs"])
    return pred["abstained"] or strength < CONFIDENT


def apply_reasoner(preds, outputs, corpus_by_id, reasoner, policy, cache_tag, use_cache=True):
    """Re-judge the escalated claims with the stage-2 reasoner (one LLM call per judged paper; raw outputs cached,
    so the hybrid / checks / llm modes of one model share a single generation pass)."""
    from collections import Counter

    from src.reasoning import PROMPT_VERSION, combine_docs

    path = Path("eval/cache") / ("reason_" + hashlib.sha1(
        f"{reasoner.llm.model_name}|{PROMPT_VERSION}".encode()).hexdigest()[:16] + ".json")
    cache = json.loads(path.read_text(encoding="utf-8")) if use_cache and path.exists() else {}
    by_id = {o["id"]: o for o in outputs}
    todo = [p for p in preds if needs_reasoning(p, by_id[p["id"]], policy)]
    jobs = [(p["id"], d["doc_id"]) for p in todo for d in by_id[p["id"]]["docs"] if d["doc_id"] in corpus_by_id]
    key = lambda cid, did: f"{cache_tag}:{cid}:{did}"  # noqa: E731
    missing = [(c, d) for c, d in jobs if key(c, d) not in cache]
    t0 = time.perf_counter()
    if missing:
        raw = reasoner.extract([(by_id[c]["claim"], corpus_by_id[d].sentences) for c, d in missing])
        cache.update({key(c, d): r for (c, d), r in zip(missing, raw)})
        if use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache), encoding="utf-8")
    seconds = time.perf_counter() - t0
    sources, changed = Counter(), 0
    for p in todo:
        o = by_id[p["id"]]
        docs = [d for d in o["docs"] if d["doc_id"] in corpus_by_id]
        js = [reasoner.judge_raw(o["claim"], corpus_by_id[d["doc_id"]].sentences, cache[key(p["id"], d["doc_id"])]) for d in docs]
        res = combine_docs(js)
        p["stage1_pred"] = p["pred"]
        p["stage2"] = [{"doc_id": d["doc_id"], **{k: v for k, v in j.items() if k != "raw"}} for d, j in zip(docs, js)]
        for j in js:
            sources[j["source"]] += 1
        if res["label"] is None:
            continue
        changed += res["label"] != p["pred"]
        p["pred"], p["abstained"] = res["label"], res["label"] == NEI
        p["driver_doc"] = docs[res["doc_index"]]["doc_id"] if res["doc_index"] is not None else None
        p["note"] = "stage2:" + (js[res["doc_index"]]["source"] if res["doc_index"] is not None else res.get("note", "no verdict"))
    return {"policy": policy, "n_escalated": len(todo), "n_changed": changed, "n_llm_calls": len(missing),
            "n_paper_judgements": len(jobs), "generation_seconds": seconds,
            "ms_per_llm_call": 1000 * seconds / len(missing) if missing else None, "sources": dict(sources)}


def claim_level(preds, n_boot=1000, seed=0):
    rows = [p for p in preds if p["gold"] in LABELS]  # drops MIXED
    yt, yp = [p["gold"] for p in rows], [p["pred"] for p in rows]
    return {
        "n": len(rows),
        "n_mixed_excluded": len(preds) - len(rows),
        "macro_f1": M.macro_f1(yt, yp),
        "macro_f1_ci95": M.bootstrap_ci(yt, yp, M.macro_f1, n_boot=n_boot, seed=seed),
        "accuracy": M.accuracy(yt, yp),
        "per_class": M.prf_per_class(yt, yp),
        "confusion_rows_gold_cols_pred": {"labels": list(LABELS), "matrix": M.confusion_matrix(yt, yp)},
        "false_support_rate": M.false_support_rate(yt, yp),
        "support_precision": M.support_precision(yt, yp),
        "coverage": M.coverage(yp),
    }


def abstract_level(outputs, tau):
    pred, gold = {}, {}
    for o in outputs:
        gold[o["id"]] = {d: {"label": g["label"], "rationales": g["rationales"]} for d, g in o["gold_docs"].items()}
        labels = doc_level_labels([_probs(d) for d in o["docs"]], tau)
        pred[o["id"]] = {
            d["doc_id"]: {"label": lab, "sentences": d["sentence_idxs"]} for d, lab in zip(o["docs"], labels) if lab
        }
    return {
        "label_only": M.abstract_level_prf(pred, gold, require_rationale=False),
        "label_and_rationale": M.abstract_level_prf(pred, gold, require_rationale=True),
    }


def tau_sweep(outputs, margin=0.0):
    rows = []
    for tau in TAU_GRID:
        preds = predict(outputs, tau, margin)
        rows_ = [p for p in preds if p["gold"] in LABELS]
        yt, yp = [p["gold"] for p in rows_], [p["pred"] for p in rows_]
        rows.append(
            {
                "tau": tau,
                "macro_f1": M.macro_f1(yt, yp),
                "false_support_rate": M.false_support_rate(yt, yp),
                "support_precision": M.support_precision(yt, yp),
                "coverage": M.coverage(yp),
            }
        )
    return rows


def tune_tau(outputs, margin=0.0):
    """Pick tau maximising macro-F1 on the tuning outputs; ties go to the larger (more cautious) tau."""
    best = None
    for row in tau_sweep(outputs, margin):
        if best is None or row["macro_f1"] > best["macro_f1"] + 1e-12 or (
            abs(row["macro_f1"] - best["macro_f1"]) <= 1e-12 and row["tau"] > best["tau"]
        ):
            best = row
    return best


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


# module name -> distribution name.  Read versions from package metadata, not `module.__version__`:
# rank_bm25 has no __version__, which recorded `null` in every Session 5 results.json (RESULTS_session05.md, check g).
_DISTS = {"torch": "torch", "transformers": "transformers", "sentence_transformers": "sentence-transformers",
          "rank_bm25": "rank-bm25", "numpy": "numpy"}


def _versions():
    from importlib import metadata

    v = {"python": platform.python_version(), "platform": platform.platform()}
    for mod, dist in _DISTS.items():
        try:
            v[mod] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            try:
                v[mod] = getattr(__import__(mod), "__version__", None)
            except Exception:
                v[mod] = None
    return v


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description="CiteCheck evaluation harness")
    p.add_argument("--name", required=True, help="run name; results go to eval/results/<name>/")
    p.add_argument("--data-dir", default="data/scifact")
    p.add_argument("--split", default="dev", choices=["dev", "train"], help="split to REPORT on (test has no public labels)")
    p.add_argument("--retriever", default="bm25", help="bm25 | dense | hybrid | hybrid+rerank | bm25+rerank")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--verifier", default="nli", help="majority | lexical | nli | nli:<model>")
    p.add_argument("--model", default=None, help=f"NLI checkpoint (default {DEFAULT_NLI_MODEL}); HF id or local folder")
    p.add_argument("--k", type=int, default=3, help="papers judged per claim")
    p.add_argument("--sentences", type=int, default=3, help="sentences per paper fed to the verifier (window unit)")
    p.add_argument("--evidence-unit", default="window", choices=EVIDENCE_UNITS,
                   help="what the verifier reads per paper: top sentences joined | whole abstract | each sentence")
    p.add_argument("--tau", type=float, default=None, help="fixed abstain threshold (skips tuning)")
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--tune-split", default="train", choices=["train", "dev", "none"])
    p.add_argument("--tune-n", type=int, default=200, help="claims used to tune tau")
    p.add_argument("--allow-leak", action="store_true", help="allow tuning on the evaluation split (recorded in results)")
    p.add_argument("--oracle", action="store_true", help="also run oracle-paper and oracle-rationale ablations")
    p.add_argument("--compare-with", action="append", default=[], help="results dir of another run (paired bootstrap)")
    p.add_argument("--limit", type=int, default=None, help="only the first N claims (smoke test; result flagged partial)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default="eval/results")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--promote", action="store_true", help="write eval/results/deployed_config.json for the app")
    p.add_argument("--reasoner", default=None, help="stage 2: local instruction LLM (HF id or folder) for the hybrid reasoner")
    p.add_argument("--reasoner-mode", default="hybrid", choices=["hybrid", "checks", "llm"])
    p.add_argument("--escalate", default="not-confident", choices=ESCALATE, help="which claims stage 2 re-judges")
    p.add_argument("--reasoner-batch-size", type=int, default=8)
    p.add_argument("--reasoner-max-new-tokens", type=int, default=384)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    return p


def run(args) -> dict:
    data_dir = find_scifact_dir(args.data_dir)
    if data_dir is None:
        sys.exit(f"SciFact not found under '{args.data_dir}'. Run:  python -m data.fetch_scifact")
    split_path = data_dir / f"claims_{args.split}.jsonl"
    if not split_has_labels(split_path):
        sys.exit(f"{split_path} has no gold labels; cannot score it locally.")
    synthetic = is_synthetic_fixture(data_dir)

    corpus = load_corpus(data_dir / "corpus.jsonl")
    corpus_by_id = {d.doc_id: d for d in corpus}
    claims = sorted(load_claims(split_path), key=lambda c: c.id)
    if args.limit:
        claims = claims[: args.limit]

    verifier_kw = {"device": args.device, "batch_size": args.batch_size}
    spec = args.verifier if not args.model else f"nli:{args.model}"
    if spec.lower() in ("majority", "lexical"):
        verifier_kw = {}
    verifier = build_verifier(spec, **verifier_kw)
    retr_kw = {"model_name": args.embed_model} if args.retriever not in ("bm25", "bm25+rerank") else {}
    retriever = build_retriever(args.retriever, corpus, **retr_kw)
    scorer = LexicalSentenceScorer.from_docs(corpus)

    data_hash = {"corpus": sha256_file(data_dir / "corpus.jsonl")[:16], args.split: sha256_file(split_path)[:16]}
    base_spec = {
        "retriever": retriever.name, "embed": args.embed_model, "verifier": verifier.name, "k": args.k,
        "m": args.sentences, "data": data_hash, "limit": args.limit,
    }
    if args.evidence_unit != "window":  # keeps Session 5 cache keys valid
        base_spec["unit"] = args.evidence_unit
    fingerprint = _local_model_fingerprint(spec[4:] if spec.lower().startswith("nli:") else None)
    if fingerprint:  # retraining into the same folder must not reuse the old checkpoint's cached outputs
        base_spec["model_sha256_prefix"] = fingerprint

    def get(split_name, claim_list, mode):
        s = dict(base_spec, split=split_name, mode=mode, n=len(claim_list), first=claim_list[0].id if claim_list else None)
        return collect_cached(
            s, not args.no_cache, claims=claim_list, corpus_by_id=corpus_by_id, retriever=retriever,
            verifier=verifier, scorer=scorer, k=args.k, m=args.sentences, mode=mode, unit=args.evidence_unit,
        )

    # ---- tau: tuned on TRAIN, never on the split we report ---------------------------------
    tuning = {"method": "fixed", "split": None}
    if args.tau is not None:
        tau = args.tau
    elif spec.lower() == "majority" or args.tune_split == "none":
        tau = 0.5
        tuning = {"method": "default_0.5", "split": None}
    else:
        if args.tune_split == args.split and not args.allow_leak:
            sys.exit("Refusing to tune tau on the split you report on. Use --tune-split train (default) or --allow-leak.")
        tune_path = data_dir / f"claims_{args.tune_split}.jsonl"
        tune_claims = tau_tuning_claims(load_claims(tune_path), args.tune_n, args.seed)
        if args.limit:
            tune_claims = tune_claims[: args.limit]
        n_trained_on = 0
        model_path = spec[4:] if spec.lower().startswith("nli:") else None
        if args.tune_split == "train" and model_path:
            n_trained_on = check_tuning_claims_held_out(model_path, tune_claims, args.allow_leak)
        tune_out = get(f"tune-{args.tune_split}", tune_claims, "e2e")["outputs"]
        best = tune_tau(tune_out, args.margin)
        tau = best["tau"]
        tuning = {
            "method": "grid_max_macro_f1", "split": args.tune_split, "n_claims": len(tune_claims),
            "tuned_macro_f1": best["macro_f1"], "leak": args.tune_split == args.split or n_trained_on > 0,
            "n_tuning_claims_seen_in_finetuning": n_trained_on,
        }

    # ---- end-to-end ------------------------------------------------------------------------
    e2e = get(args.split, claims, "e2e")
    outputs = e2e["outputs"]
    preds = predict(outputs, tau, args.margin)
    reasoner, stage2 = None, {}
    if args.reasoner:
        if args.promote:
            sys.exit("--promote with --reasoner: the app does not run the stage-2 reasoner yet.")
        from src.reasoning import HybridReasoner, LocalLLM

        reasoner = HybridReasoner(LocalLLM(args.reasoner, device=args.device, batch_size=args.reasoner_batch_size,
                                           max_new_tokens=args.reasoner_max_new_tokens), mode=args.reasoner_mode)
        stage2["end_to_end"] = apply_reasoner(preds, outputs, corpus_by_id, reasoner, args.escalate,
                                              args.split, not args.no_cache)

    results = {
        "name": args.name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split,
        "synthetic_fixture": synthetic,
        "partial": bool(args.limit),
        "config": {
            "retriever": retriever.name, "embed_model": args.embed_model, "verifier": verifier.name, "k": args.k,
            "sentences_per_paper": args.sentences, "evidence_unit": args.evidence_unit, "tau": tau, "margin": args.margin,
        },
        "tau_tuning": tuning,
        "data": {"dir": str(data_dir), "sha256_prefix": data_hash, "n_corpus": len(corpus),
                 "n_claims": len(claims), "label_distribution": label_distribution(claims)},
        "claim_level": claim_level(preds, args.n_boot, args.seed),
        "retrieval": M.retrieval_report(
            {o["id"]: o["ranking"] for o in outputs}, {o["id"]: set(o["gold_docs"]) for o in outputs}
        ),
        "abstract_level": abstract_level(outputs, tau),
        "tau_sweep_analysis_only": tau_sweep(outputs, args.margin),
        "timing": e2e["timing"],
        **({"stage2": {"reasoner": reasoner.name, "mode": args.reasoner_mode, "escalate": args.escalate,
                       "confident_threshold": CONFIDENT, "runs": stage2,
                       "note": "claim_level, paired comparisons and cited_paper include stage 2; abstract_level, "
                               "tau_sweep and oracle_rationale are stage 1 only"}} if reasoner else {}),
        "provenance": {"git_commit": _git_commit(), "versions": _versions(), "argv": sys.argv[1:], "seed": args.seed},
    }
    if model_log := _local_train_log(spec):
        trained_unit = json.loads(model_log.read_text(encoding="utf-8")).get("args", {}).get("unit", "window")
        results["config"]["trained_on_unit"] = trained_unit
        if trained_unit != args.evidence_unit:
            print(f"[warn] {spec[4:]} was fine-tuned on '{trained_unit}' evidence but is evaluated on '{args.evidence_unit}'")
    if hasattr(verifier, "n_params") and spec.lower().startswith("nli"):
        results["timing"]["n_params"] = verifier.n_params
        results["timing"]["device"] = getattr(verifier, "device", None)

    # ---- oracle ablations: where do the errors come from? ------------------------------------
    if args.oracle:
        att = {"end_to_end": results["claim_level"]["macro_f1"]}
        for mode in ("oracle_doc", "oracle_rationale"):
            o = get(args.split, claims, mode)["outputs"]
            mode_preds = predict(o, tau, args.margin)
            if reasoner is not None and mode == "oracle_doc":
                stage2["oracle_doc"] = apply_reasoner(mode_preds, o, corpus_by_id, reasoner, args.escalate,
                                                      args.split, not args.no_cache)
            if mode == "oracle_doc":  # the product setting: the user gives the paper they cite
                cited_preds = mode_preds
                results["cited_paper"] = claim_level(mode_preds, args.n_boot, args.seed)
            att[mode] = claim_level(mode_preds, args.n_boot, args.seed)["macro_f1"]
        att["loss_from_retrieval"] = att["oracle_doc"] - att["end_to_end"]
        att["loss_from_sentence_selection"] = att["oracle_rationale"] - att["oracle_doc"]
        att["remaining_gap_is_verifier"] = 1.0 - att["oracle_rationale"]
        att["reading"] = (
            "macro-F1 with the gold paper(s) supplied (oracle_doc) or gold paper+sentences (oracle_rationale). "
            "loss_from_retrieval = what better retrieval could buy; remaining_gap_is_verifier = what only a better "
            "classifier can fix."
        )
        results["error_attribution"] = att

    # ---- paired comparison with earlier runs -----------------------------------------------
    comps = []
    mine = {p["id"]: p for p in preds if p["gold"] in LABELS}
    for other_dir in args.compare_with:
        pf = Path(other_dir) / "predictions.jsonl"
        if not pf.exists():
            print(f"[warn] --compare-with {other_dir}: no predictions.jsonl, skipped")
            continue
        theirs = {r["id"]: r for r in map(json.loads, pf.read_text(encoding="utf-8").splitlines()) if r["gold"] in LABELS}
        ids = sorted(set(mine) & set(theirs))
        if not ids:
            continue
        cmp_ = M.paired_bootstrap_diff(
            [mine[i]["gold"] for i in ids], [mine[i]["pred"] for i in ids], [theirs[i]["pred"] for i in ids],
            n_boot=args.n_boot, seed=args.seed,
        )
        cmp_.update({"vs": Path(other_dir).name, "n": len(ids)})
        comps.append(cmp_)
    results["paired_comparisons"] = comps
    if args.oracle:  # same comparison in the cited-paper setting, against runs that saved it
        cited = {p["id"]: p for p in cited_preds if p["gold"] in LABELS}
        results["cited_paper"]["paired_comparisons"] = []
        for other_dir in args.compare_with:
            pf = Path(other_dir) / "predictions_cited_paper.jsonl"
            if not pf.exists():
                continue
            theirs = {r["id"]: r for r in map(json.loads, pf.read_text(encoding="utf-8").splitlines()) if r["gold"] in LABELS}
            ids = sorted(set(cited) & set(theirs))
            if ids:
                c = M.paired_bootstrap_diff([cited[i]["gold"] for i in ids], [cited[i]["pred"] for i in ids],
                                            [theirs[i]["pred"] for i in ids], n_boot=args.n_boot, seed=args.seed)
                c.update({"vs": Path(other_dir).name, "n": len(ids)})
                results["cited_paper"]["paired_comparisons"].append(c)

    write_outputs(Path(args.out) / args.name, results, preds, outputs)
    if args.oracle:
        with open(Path(args.out) / args.name / "predictions_cited_paper.jsonl", "w", encoding="utf-8") as f:
            for p in cited_preds:
                f.write(json.dumps(p) + "\n")
    if model_log := _local_train_log(spec):  # keep the fine-tuning record (held-out ids, validation curve) with the run
        shutil.copyfile(model_log, Path(args.out) / args.name / "train_log.json")
    write_latest(Path(args.out))
    if args.promote:
        promote(Path(args.out), results, args)
    return results


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def _fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def summary_md(r: dict) -> str:
    cl, rt = r["claim_level"], r["retrieval"]
    lines = [
        f"# Run `{r['name']}`  ({r['split']} split, n={cl['n']})",
        "",
        "> " + ("**SYNTHETIC FIXTURE - plumbing check only, not a result.**" if r["synthetic_fixture"] else "Real data.")
        + ("  **PARTIAL run (--limit).**" if r["partial"] else ""),
        "",
        f"- retriever `{r['config']['retriever']}`, verifier `{r['config']['verifier']}`, k={r['config']['k']}, "
        f"sentences/paper={r['config']['sentences_per_paper']}, evidence unit={r['config'].get('evidence_unit', 'window')}, "
        f"tau={r['config']['tau']}",
        f"- tau tuning: {r['tau_tuning']}",
        "",
        "## Claim-level (3-class: SUPPORT / CONTRADICT / NEI)",
        f"- **macro-F1 {cl['macro_f1']:.3f}**  95% CI {_fmt_ci(cl['macro_f1_ci95'])}  accuracy {cl['accuracy']:.3f}",
        f"- **false-SUPPORT rate {cl['false_support_rate']:.3f}**  SUPPORT precision {cl['support_precision']:.3f}  coverage {cl['coverage']:.3f}",
        f"- claims excluded as MIXED gold: {cl['n_mixed_excluded']}",
        "",
        "| class | precision | recall | F1 | gold n |",
        "|---|---|---|---|---|",
    ]
    for lab, v in cl["per_class"].items():
        lines.append(f"| {lab} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
    lines += [
        "",
        "## Retrieval (claims with gold evidence: n=%d, avg gold papers/claim %.2f)" % (rt["n_claims"], rt["avg_gold_docs_per_claim"]),
        "| k | recall@k | precision@k |",
        "|---|---|---|",
    ]
    for k in (1, 3, 5, 10, 20):
        lines.append(f"| {k} | {rt[f'recall@{k}']:.3f} | {rt[f'precision@{k}']:.3f} |")
    lines.append(f"\nMRR {rt['mrr']:.3f}, nDCG@10 {rt['ndcg@10']:.3f}. (precision@k is capped near 1/k when ~1 paper is relevant.)")
    al = r["abstract_level"]
    lines += [
        "",
        "## Abstract-level (SciFact-leaderboard style; NOT the official scorer)",
        f"- label-only    P {al['label_only']['precision']:.3f} R {al['label_only']['recall']:.3f} F1 {al['label_only']['f1']:.3f}",
        f"- label+rationale P {al['label_and_rationale']['precision']:.3f} R {al['label_and_rationale']['recall']:.3f} F1 {al['label_and_rationale']['f1']:.3f}",
    ]
    if "error_attribution" in r:
        a = r["error_attribution"]
        lines += [
            "",
            "## Where do the errors come from? (macro-F1 with progressively more oracle help)",
            f"- end-to-end {a['end_to_end']:.3f} -> oracle paper {a['oracle_doc']:.3f} -> oracle paper+sentences {a['oracle_rationale']:.3f}",
            f"- lost to retrieval {a['loss_from_retrieval']:+.3f}; lost to sentence selection {a['loss_from_sentence_selection']:+.3f}; "
            f"remaining gap (verifier) {a['remaining_gap_is_verifier']:.3f}",
        ]
    for c in r.get("paired_comparisons", []):
        lines.append(
            f"\n**vs `{c['vs']}`**: macro-F1 difference {c['diff']:+.3f}, 95% CI [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], "
            f"P(this run better) = {c['p_a_better']:.2f}  (n={c['n']})"
        )
    if "cited_paper" in r:
        cp = r["cited_paper"]
        lines += ["", "## Cited-paper setting (the product: the user gives the paper; = oracle_doc)",
                  f"- macro-F1 {cp['macro_f1']:.3f}  95% CI {_fmt_ci(cp['macro_f1_ci95'])}  false-SUPPORT {cp['false_support_rate']:.3f}"]
        for c in cp.get("paired_comparisons", []):
            lines.append(f"- vs `{c['vs']}`: {c['diff']:+.3f}, 95% CI [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] (n={c['n']})")
    if "stage2" in r:
        s2 = r["stage2"]
        lines += ["", f"## Stage 2: `{s2['reasoner']}` (escalate: {s2['escalate']})"]
        for where, st in s2["runs"].items():
            lines.append(f"- {where}: {st['n_escalated']} claims escalated, {st['n_changed']} verdicts changed, "
                         f"{st['n_llm_calls']} LLM calls; sources {st['sources']}")
        lines.append(f"- {s2['note']}")
    t = r["timing"]
    lines += [
        "",
        f"## Cost / latency\nretrieval {t['retrieval_ms_per_claim']:.1f} ms/claim, verification {t['verify_ms_per_claim']:.1f} ms/claim"
        + (f", model params {t['n_params']:,} on {t.get('device')}" if "n_params" in t else "")
        + ". Local model: $0 per request (compute only).",
        "",
        "> With ~300 dev claims a 95% CI is several F1 points wide: differences smaller than the CI are not evidence.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, results: dict, preds: list, outputs: list):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        by_id = {o["id"]: o for o in outputs}
        for p in preds:
            f.write(json.dumps({**p, "docs": by_id[p["id"]]["docs"], "ranking": by_id[p["id"]]["ranking"][:10],
                                "gold_docs": list(by_id[p["id"]]["gold_docs"])}) + "\n")
    (out_dir / "summary.md").write_text(summary_md(results), encoding="utf-8")
    print(summary_md(results))
    print(f"[saved] {out_dir}")


def unit_tag(config: dict) -> str:
    unit = config.get("evidence_unit", "window")
    return "" if unit == "window" else f" [{unit}]"


def write_latest(out_root: Path):
    rows = []
    for rj in sorted(out_root.glob("*/results.json")):
        r = json.loads(rj.read_text(encoding="utf-8"))
        rows.append(r)
    rows.sort(key=lambda r: r["created_utc"])
    lines = [
        "# All evaluation runs",
        "",
        "| run | split | retriever | verifier | tau | macro-F1 [95% CI] | false-SUPPORT | recall@5 | flags |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cl, rt, c = r["claim_level"], r["retrieval"], r["config"]
        flags = ", ".join(x for x, on in (("SYNTHETIC", r["synthetic_fixture"]), ("partial", r["partial"]),
                                          (f"stage2:{r.get('stage2', {}).get('mode')}", "stage2" in r)) if on) or "-"
        lines.append(
            f"| {r['name']} | {r['split']} | {c['retriever']} | {c['verifier']}{unit_tag(c)} | {c['tau']} | {cl['macro_f1']:.3f} {_fmt_ci(cl['macro_f1_ci95'])} "
            f"| {cl['false_support_rate']:.3f} | {rt['recall@5']:.3f} | {flags} |"
        )
    (out_root / "LATEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def promote(out_root: Path, results: dict, args):
    if results["synthetic_fixture"] or results["partial"] or results["split"] != "dev":
        sys.exit("Only a full, real-data DEV run can be promoted to deployed_config.json.")
    cfg = {
        "run_name": results["name"],
        "retriever": args.retriever,
        "embed_model": args.embed_model,
        "verifier": args.verifier if not args.model else f"nli:{args.model}",
        "k": args.k,
        "sentences_per_paper": args.sentences,
        "evidence_unit": args.evidence_unit,
        "tau": results["config"]["tau"],
        "margin": args.margin,
        "dev_macro_f1": results["claim_level"]["macro_f1"],
        "dev_macro_f1_ci95": results["claim_level"]["macro_f1_ci95"],
        "dev_false_support_rate": results["claim_level"]["false_support_rate"],
        "data_sha256_prefix": results["data"]["sha256_prefix"],
        "git_commit": results["provenance"]["git_commit"],
        "created_utc": results["created_utc"],
    }
    (out_root / "deployed_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[promoted] {out_root / 'deployed_config.json'}  <- the app will now run exactly this configuration")


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
