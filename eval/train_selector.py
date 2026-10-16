"""Train the evidence-sentence selector on SciFact TRAIN.   Owner: Modeling.

    python -m eval.train_selector --out models/citecheck-selector
    python -m eval.run_eval --name finetuned_nli_selector --retriever bm25 --verifier nli --model models/citecheck-nli-ft \\
        --sentence-scorer cross:models/citecheck-selector --oracle --compare-with eval/results/finetuned_nli

Why: given the gold sentences the fine-tuned verifier scores 0.849 macro-F1, given the 3 sentences our word-overlap
selector picks from the gold paper 0.727.  A cross-encoder reads claim and sentence together, so it can match
"lowers LDL" to "reduced low-density lipoprotein" where IDF overlap cannot.

Data (train split only; the claims run_eval tunes tau on are held out exactly as in eval.finetune_nli):
  * positive: every gold rationale sentence of a gold paper (label 1)
  * negative: every other sentence of that paper, plus the --neg-sentences sentences that look most like the claim
              in the papers an NEI claim cites and in the top retrieved non-gold papers (label 0)
Validation: on the held-out claims, how often the top-m sentences contain a complete gold rationale; the best epoch
is kept.  The weights go to models/ (git-ignored); the run's record is train_log.json next to them.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from eval.run_eval import tau_tuning_claims
from src.data_io import find_scifact_dir, load_claims, load_corpus, sha256_file
from src.evidence import LexicalSentenceScorer, select_sentences
from src.retrieval import BM25Retriever
from src.schema import NEI

DEFAULT_BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_selector_pairs(claims, corpus_by_id, retriever, lexical, neg_sentences=3, neg_docs=1):
    """-> list of (claim, sentence, label 0/1)."""
    pairs = []
    for c in claims:
        gold = set(c.gold_docs)
        for doc_id in c.gold_docs:
            doc = corpus_by_id.get(doc_id)
            if doc is None:
                continue
            rationale = set(c.rationale_sentence_idxs(doc_id))
            pairs += [(c.claim, s, int(i in rationale)) for i, s in enumerate(doc.sentences)]
        others = [corpus_by_id.get(d) for d in c.cited_doc_ids] if c.gold_label == NEI else []
        if retriever is not None and neg_docs > 0:
            others += [h.doc for h in retriever.search(c.claim, neg_docs + len(gold) + 3) if h.doc.doc_id not in gold][:neg_docs]
        for doc in others:
            if doc is not None and doc.sentences:
                for i in select_sentences(c.claim, doc.sentences, neg_sentences, lexical):
                    pairs.append((c.claim, doc.sentences[i], 0))
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE, help="a sequence-classification checkpoint (1 logit is ideal)")
    ap.add_argument("--data-dir", default="data/scifact")
    ap.add_argument("--out", default="models/citecheck-selector")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--neg-sentences", type=int, default=3)
    ap.add_argument("--neg-docs", type=int, default=1)
    ap.add_argument("--m", type=int, default=3, help="sentences selected per paper (validation metric)")
    ap.add_argument("--tune-n", type=int, default=200, help="held-out claims, same as run_eval --tune-n")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    data_dir = find_scifact_dir(a.data_dir)
    if data_dir is None:
        raise SystemExit("SciFact not found. Run: python -m data.fetch_scifact")
    corpus = load_corpus(data_dir / "corpus.jsonl")
    by_id = {d.doc_id: d for d in corpus}
    train_path = data_dir / "claims_train.jsonl"
    all_claims = load_claims(train_path)
    heldout = tau_tuning_claims(all_claims, a.tune_n, a.seed) if a.tune_n > 0 else []
    held_ids = {c.id for c in heldout}
    claims = [c for c in all_claims if c.id not in held_ids]
    lexical = LexicalSentenceScorer.from_docs(corpus)
    pairs = build_selector_pairs(claims, by_id, BM25Retriever(corpus), lexical, a.neg_sentences, a.neg_docs)
    n_pos = sum(p[2] for p in pairs)
    print(f"{len(claims)} training claims -> {len(pairs)} pairs ({n_pos} rationale sentences); {len(heldout)} held-out claims")

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=1, ignore_mismatched_sizes=True)
    model = model.float().to(dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(pairs) - n_pos) / max(n_pos, 1), device=dev))

    def encode(batch):
        return tok([p[0] for p in batch], [p[1] for p in batch], truncation="only_second", max_length=a.max_length,
                   padding=True, return_tensors="pt").to(dev)

    class _Scorer:  # the model being trained, behind the interface select_sentences() expects
        def score(self, claim, sentences):
            with torch.no_grad():
                return model(**encode([(claim, s, 0) for s in sentences])).logits[:, 0].float().cpu().tolist()

    def validate():
        hits = []
        model.eval()
        for c in heldout:
            for d in c.gold_docs:
                sets = [set(r) for r in c.rationale_sets(d) if r]
                if d in by_id and sets:
                    chosen = set(select_sentences(c.claim, by_id[d].sentences, a.m, _Scorer()))
                    hits.append(any(r <= chosen for r in sets))
        model.train()
        return sum(hits) / len(hits) if hits else None

    lexical_recall = None
    if heldout:
        hits = [any(set(r) <= set(select_sentences(c.claim, by_id[d].sentences, a.m, lexical)) for r in c.rationale_sets(d) if r)
                for c in heldout for d in c.gold_docs if d in by_id and any(c.rationale_sets(d))]
        lexical_recall = sum(hits) / len(hits) if hits else None
        print(f"held-out rationale recall@{a.m}: word overlap {lexical_recall:.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = min(((len(pairs) + a.batch_size - 1) // a.batch_size) * a.epochs, a.max_steps or 10**9)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    out, log, val_log, best, step, t0 = Path(a.out), [], [], None, 0, time.time()
    model.train()
    for epoch in range(a.epochs):
        order = list(range(len(pairs)))
        random.shuffle(order)
        for s in range(0, len(order), a.batch_size):
            b = [pairs[i] for i in order[s : s + a.batch_size]]
            y = torch.tensor([float(p[2]) for p in b], device=dev)
            loss = loss_fn(model(**encode(b)).logits[:, 0].float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"epoch {epoch + 1} step {step}/{total} loss {loss.item():.4f}  ({time.time() - t0:.0f}s)")
                log.append({"step": step, "loss": float(loss.item())})
            if a.max_steps and step >= a.max_steps:
                break
        r = validate()
        val_log.append({"epoch": epoch + 1, "step": step, f"val_rationale_recall@{a.m}": r})
        print(f"epoch {epoch + 1}: held-out rationale recall@{a.m} {r}")
        if best is None or r is None or r > best[1]:
            best = (epoch + 1, r)
            model.save_pretrained(out)
            tok.save_pretrained(out)
        if a.max_steps and step >= a.max_steps:
            break
    (out / "train_log.json").write_text(json.dumps({
        "kind": "sentence_selector", "base": a.base, "args": vars(a), "n_train_claims": len(claims), "n_pairs": len(pairs),
        "n_rationale_pairs": n_pos, "heldout_claim_ids": sorted(held_ids), "lexical_val_recall": lexical_recall,
        "val_log": val_log, "best_epoch": best[0], "loss_log": log, "train_file_sha256": sha256_file(train_path)[:16],
        "seconds": time.time() - t0, "device": dev,
        "note": "trained on claims_train.jsonl minus heldout_claim_ids; use with run_eval --sentence-scorer cross:<this folder>",
    }, indent=2), encoding="utf-8")
    print(f"[saved] {out} (best epoch {best[0]}, held-out recall {best[1]})")


if __name__ == "__main__":
    main()
