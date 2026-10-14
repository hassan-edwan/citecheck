"""Fine-tune the NLI verifier on SciFact TRAIN.   Owner: Data & Evaluation (Sahil).   (STRETCH goal)

    python -m eval.finetune_nli --out models/citecheck-nli-ft --epochs 3 --lr 1e-5 --batch-size 8
    python -m eval.run_eval --name finetuned_nli --retriever bm25 --verifier nli --model models/citecheck-nli-ft \\
        --oracle --compare-with eval/results/zero_shot_nli

What it trains on (train split only; dev/test are never read):
  * positives : gold rationale sentences of each evidence paper -> SUPPORT / CONTRADICT
  * inference-style positives (--match-inference, on by default): the top-m sentences our own sentence
                selector picks from the gold paper, when they include a rationale sentence.  At inference the
                verifier sees selector output, not gold rationales, so it should be trained on that too.
  * NEI       : for claims with no evidence, the best sentences of the papers the claim cited -> neutral
  * hard negatives: papers OUR retriever ranks high that are not gold -> neutral (teaches the model to
                    abstain on plausible-looking but irrelevant papers, exactly what happens at inference)
  * (optional, --gold-doc-neutral N) the N non-rationale sentence windows of a gold paper that look most
                like the claim -> neutral

Evidence units (--unit; evaluate with the same run_eval --evidence-unit):
  * window   : the pairs above; the verifier reads the top-m selected sentences joined (Session 5 design)
  * abstract : the verifier reads the whole abstract; no sentence selection (build_abstract_pairs)
  * sentence : the verifier reads one sentence at a time and the paper takes its most decisive sentence
               (build_sentence_pairs; aggregation in src/evidence.aggregate_sentence_probs)

Held-out claims: the train claims eval.run_eval tunes tau on (run_eval.tau_tuning_claims, same --tune-n and
--seed) are LEFT OUT of training.  They are used for per-epoch validation and the best epoch is kept, and
run_eval refuses to tune tau on any claim the checkpoint was trained on.

On a GPU, mixed precision is used automatically; --grad-accum and --gradient-checkpointing let a large
model fit on a free Colab T4.  CPU works for a base model but is slow (~15-30 min/epoch).  Model weights go
to models/ which is git-ignored: do NOT commit them (record the run in eval/results instead).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from eval.run_eval import tau_tuning_claims
from src.data_io import MIXED, find_scifact_dir, load_claims, load_corpus, sha256_file
from src.evidence import LexicalSentenceScorer, build_premise, select_sentences
from src.nli import DEFAULT_NLI_MODEL, resolve_label_indices
from src.retrieval import BM25Retriever
from src.schema import NEI, SUPPORT


CLASSES = ("support", "neutral", "contradict")


def build_training_pairs(claims, corpus_by_id, retriever, scorer, m=3, neg_per_claim=1, match_inference=False,
                         gold_doc_neutral=0):
    """Returns list of (premise, claim, class) with class in {"support","neutral","contradict"}."""
    pairs = []
    for c in claims:
        gold = set(c.gold_docs)
        for doc_id, lab in c.gold_doc_labels.items():
            doc = corpus_by_id.get(doc_id)
            if lab == MIXED or doc is None:
                continue
            cls = "support" if lab == SUPPORT else "contradict"
            premises = set()
            for rset in c.rationale_sets(doc_id):
                premise = " ".join(doc.sentences[i] for i in rset if i < len(doc.sentences))
                if premise:
                    premises.add(premise)
                    pairs.append((premise, c.claim, cls))
            rationale = set(c.rationale_sentence_idxs(doc_id))
            if match_inference:
                idxs = select_sentences(c.claim, doc.sentences, m, scorer)
                premise = build_premise(doc.sentences, idxs)
                if rationale & set(idxs) and premise not in premises:
                    pairs.append((premise, c.claim, cls))
            if gold_doc_neutral > 0:
                rest = [i for i in range(len(doc.sentences)) if i not in rationale]
                scores = scorer.score(c.claim, [doc.sentences[i] for i in rest])
                ranked = [i for _, i in sorted(zip(scores, rest), key=lambda t: (-t[0], t[1]))]
                for start in range(0, min(len(ranked), gold_doc_neutral * m), m):
                    window = sorted(ranked[start : start + m])
                    pairs.append((build_premise(doc.sentences, window), c.claim, "neutral"))
        if c.gold_label == NEI:
            for doc_id in c.cited_doc_ids:
                doc = corpus_by_id.get(doc_id)
                if doc is None or not doc.sentences:
                    continue
                pairs.append((build_premise(doc.sentences, select_sentences(c.claim, doc.sentences, m, scorer)), c.claim, "neutral"))
        if retriever is not None and neg_per_claim > 0:
            taken = 0
            for hit in retriever.search(c.claim, neg_per_claim + len(gold) + 3):
                if hit.doc.doc_id in gold or not hit.doc.sentences:
                    continue
                pairs.append((build_premise(hit.doc.sentences, select_sentences(c.claim, hit.doc.sentences, m, scorer)), c.claim, "neutral"))
                taken += 1
                if taken >= neg_per_claim:
                    break
    return pairs


def _hard_negative_docs(claim, gold, retriever, n):
    if retriever is None or n <= 0:
        return []
    docs = [h.doc for h in retriever.search(claim.claim, n + len(gold) + 3) if h.doc.doc_id not in gold and h.doc.sentences]
    return docs[:n]


def build_abstract_pairs(claims, corpus_by_id, retriever, scorer, neg_per_claim=1, **_):
    """unit=abstract: the WHOLE abstract is the premise, exactly what the verifier reads at inference.

    Gold paper -> its label; cited papers of NEI claims and retrieved non-gold papers -> neutral.  The gold
    rationale alone is kept as an extra positive so the model still learns which sentences carry the verdict.
    """
    pairs = []
    for c in claims:
        gold = set(c.gold_docs)
        for doc_id, lab in c.gold_doc_labels.items():
            doc = corpus_by_id.get(doc_id)
            if lab == MIXED or doc is None or not doc.sentences:
                continue
            cls = "support" if lab == SUPPORT else "contradict"
            pairs.append((" ".join(doc.sentences), c.claim, cls))
            for rset in c.rationale_sets(doc_id):
                premise = " ".join(doc.sentences[i] for i in rset if i < len(doc.sentences))
                if premise:
                    pairs.append((premise, c.claim, cls))
        if c.gold_label == NEI:
            for doc_id in c.cited_doc_ids:
                doc = corpus_by_id.get(doc_id)
                if doc is not None and doc.sentences:
                    pairs.append((" ".join(doc.sentences), c.claim, "neutral"))
        for doc in _hard_negative_docs(c, gold, retriever, neg_per_claim):
            pairs.append((" ".join(doc.sentences), c.claim, "neutral"))
    return pairs


def build_sentence_pairs(claims, corpus_by_id, retriever, scorer, neg_per_claim=1, neutral_per_doc=4, **_):
    """unit=sentence: ONE sentence is the premise, as at inference.

    Rationale sentence -> the paper's label (a multi-sentence rationale is also kept joined, and its single
    sentences are skipped: alone, each may only partly support the claim).  Neutral: the non-rationale
    sentences of gold papers and the sentences of cited-NEI / retrieved non-gold papers, taking the
    neutral_per_doc that look MOST like the claim (the hard cases the aggregator must not fire on).
    """
    pairs = []

    def hardest(claim, doc, exclude=()):
        rest = [i for i in range(len(doc.sentences)) if i not in exclude]
        scores = scorer.score(claim, [doc.sentences[i] for i in rest])
        return [i for _, i in sorted(zip(scores, rest), key=lambda t: (-t[0], t[1]))][:neutral_per_doc]

    for c in claims:
        gold = set(c.gold_docs)
        for doc_id, lab in c.gold_doc_labels.items():
            doc = corpus_by_id.get(doc_id)
            if doc is None or not doc.sentences:
                continue
            rationale = set(c.rationale_sentence_idxs(doc_id))
            if lab != MIXED:
                cls = "support" if lab == SUPPORT else "contradict"
                for rset in c.rationale_sets(doc_id):
                    rset = [i for i in rset if i < len(doc.sentences)]
                    if rset:
                        pairs.append((build_premise(doc.sentences, rset), c.claim, cls))
            for i in hardest(c.claim, doc, exclude=rationale):
                pairs.append((doc.sentences[i], c.claim, "neutral"))
        neutral_docs = [corpus_by_id.get(d) for d in c.cited_doc_ids] if c.gold_label == NEI else []
        neutral_docs += _hard_negative_docs(c, gold, retriever, neg_per_claim)
        for doc in neutral_docs:
            if doc is not None and doc.sentences:
                for i in hardest(c.claim, doc):
                    pairs.append((doc.sentences[i], c.claim, "neutral"))
    return pairs


PAIR_BUILDERS = {"window": build_training_pairs, "abstract": build_abstract_pairs, "sentence": build_sentence_pairs}
MAX_LENGTH = {"window": 320, "abstract": 512, "sentence": 160}  # abstracts: median ~190 words, p99 ~450


def class_counts(pairs):
    return {k: sum(1 for p in pairs if p[2] == k) for k in CLASSES}


def pair_macro_f1(gold, pred):
    """Pair-level 3-class macro-F1 (validation only; the reported metric is claim-level, from run_eval)."""
    f1s = []
    for k in set(gold):
        tp = sum(g == k and p == k for g, p in zip(gold, pred))
        fp = sum(g != k and p == k for g, p in zip(gold, pred))
        fn = sum(g == k and p != k for g, p in zip(gold, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_NLI_MODEL)
    ap.add_argument("--data-dir", default="data/scifact")
    ap.add_argument("--out", default="models/citecheck-nli-ft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1, help="optimizer step every N batches (effective batch = N x batch-size)")
    ap.add_argument("--unit", default="window", choices=list(PAIR_BUILDERS),
                    help="evidence unit to train for; evaluate with run_eval --evidence-unit <same>")
    ap.add_argument("--max-length", type=int, default=None, help="default: 320 window, 512 abstract, 160 sentence")
    ap.add_argument("--neg-per-claim", type=int, default=1)
    ap.add_argument("--sentences", type=int, default=3)
    ap.add_argument("--match-inference", action=argparse.BooleanOptionalAction, default=True,
                    help="also train on the sentences our selector picks from gold papers (default on)")
    ap.add_argument("--gold-doc-neutral", type=int, default=0, help="window unit: non-rationale windows per gold paper -> neutral")
    ap.add_argument("--neutral-per-doc", type=int, default=4, help="sentence unit: neutral sentences taken per paper")
    ap.add_argument("--tune-n", type=int, default=200, help="hold out the claims run_eval tunes tau on (same as its --tune-n)")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke test)")
    ap.add_argument("--class-weights", default="auto", choices=["auto", "none"])
    ap.add_argument("--warmup", type=float, default=0.1, help="warmup share of optimizer steps")
    ap.add_argument("--gradient-checkpointing", action="store_true", help="less GPU memory, ~30%% slower (large models)")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision on GPU")
    ap.add_argument("--seed", type=int, default=0, help="training seed AND the held-out sample (same as run_eval --seed)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    a.max_length = a.max_length or MAX_LENGTH[a.unit]

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
    train_path = data_dir / "claims_train.jsonl"  # dev and test are deliberately never opened here
    all_claims = load_claims(train_path)
    heldout = tau_tuning_claims(all_claims, a.tune_n, a.seed) if a.tune_n > 0 else []
    heldout_ids = {c.id for c in heldout}
    claims = [c for c in all_claims if c.id not in heldout_ids]
    if not claims:
        raise SystemExit(f"--tune-n {a.tune_n} holds out all {len(all_claims)} train claims; lower it.")

    retriever, scorer = BM25Retriever(corpus), LexicalSentenceScorer.from_docs(corpus)
    build = dict(m=a.sentences, neg_per_claim=a.neg_per_claim, match_inference=a.match_inference,
                 gold_doc_neutral=a.gold_doc_neutral, neutral_per_doc=a.neutral_per_doc)
    builder = PAIR_BUILDERS[a.unit]
    if a.unit == "window":
        build.pop("neutral_per_doc")
    pairs = builder(claims, by_id, retriever, scorer, **build)
    val_pairs = builder(heldout, by_id, retriever, scorer, **build)
    counts = class_counts(pairs)
    print(f"[unit={a.unit}, max_length={a.max_length}] {len(claims)} training claims -> {len(pairs)} pairs {counts}; "
          f"{len(heldout)} held-out claims -> {len(val_pairs)} validation pairs")

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = dev.startswith("cuda") and not a.no_amp
    # bf16 only on Ampere+ (A100/L4); a T4 reports emulated bf16 support but runs it slowly, so it gets fp16
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    tok = AutoTokenizer.from_pretrained(a.base)
    # fp32 master weights: transformers >= 5 keeps a checkpoint's saved dtype (the MoritzLaurer models are fp16),
    # and fp16 weights cannot be trained (GradScaler: "Attempting to unscale FP16 gradients"); autocast does the fp16 math
    model = AutoModelForSequenceClassification.from_pretrained(a.base).float().to(dev)
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    idx = resolve_label_indices(model.config.id2label)
    inv = {v: k for k, v in idx.items()}
    labels = [idx[p[2]] for p in pairs]

    if a.class_weights == "auto":
        w = torch.zeros(model.config.num_labels)
        for k, v in counts.items():
            w[idx[k]] = len(pairs) / (3 * max(v, 1))
        loss_fn = torch.nn.CrossEntropyLoss(weight=w.to(dev))
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    def encode(batch):
        return tok([p[0] for p in batch], [p[1] for p in batch], truncation="only_first",
                   max_length=a.max_length, padding=True, return_tensors="pt").to(dev)

    def validate():
        if not val_pairs:
            return None
        model.eval()
        preds = []
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            for s in range(0, len(val_pairs), a.batch_size * 2):
                preds += model(**encode(val_pairs[s : s + a.batch_size * 2])).logits.argmax(-1).tolist()
        model.train()
        return pair_macro_f1([p[2] for p in val_pairs], [inv[i] for i in preds])

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    batches_per_epoch = (len(pairs) + a.batch_size - 1) // a.batch_size
    total = min(-(-batches_per_epoch // a.grad_accum) * a.epochs, a.max_steps or 10**9)
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    out = Path(a.out)  # created by the first save, so a crashed run leaves no empty checkpoint folder behind
    log, val_log, best, step, t0 = [], [], None, 0, time.time()
    model.train()
    for epoch in range(a.epochs):
        order = list(range(len(pairs)))
        random.shuffle(order)
        for bi, s in enumerate(range(0, len(order), a.batch_size)):
            b = order[s : s + a.batch_size]
            y = torch.tensor([labels[i] for i in b]).to(dev)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(**encode([pairs[i] for i in b])).logits
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss / a.grad_accum).backward()
            if (bi + 1) % a.grad_accum and s + a.batch_size < len(order):
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 20 == 0 or step == 1:
                print(f"epoch {epoch + 1} step {step}/{total} loss {loss.item():.4f}  ({time.time() - t0:.0f}s)")
                log.append({"step": step, "loss": float(loss.item())})
            if a.max_steps and step >= a.max_steps:
                break
        f1 = validate()
        val_log.append({"epoch": epoch + 1, "step": step, "val_pair_macro_f1": f1})
        print(f"epoch {epoch + 1} done: held-out pair macro-F1 {f1 if f1 is None else round(f1, 4)}")
        # keep the best epoch (by held-out train claims, never dev); without validation keep the last one
        if best is None or f1 is None or f1 > best["val_pair_macro_f1"]:
            best = val_log[-1]
            model.save_pretrained(out)
            tok.save_pretrained(out)
        if a.max_steps and step >= a.max_steps:
            break

    (out / "train_log.json").write_text(json.dumps({
        "base": a.base, "args": vars(a), "n_train_claims": len(claims), "n_pairs": len(pairs), "class_counts": counts,
        "n_val_pairs": len(val_pairs), "heldout_claim_ids": sorted(heldout_ids), "val_log": val_log, "best_epoch": best,
        "loss_log": log, "train_file_sha256": sha256_file(train_path)[:16], "seconds": time.time() - t0,
        "device": dev, "amp": str(amp_dtype) if use_amp else None,
        "note": "trained on claims_train.jsonl minus heldout_claim_ids; evaluate with eval.run_eval on dev",
    }, indent=2), encoding="utf-8")
    print(f"[saved] {out} (best epoch {best['epoch']})")


if __name__ == "__main__":
    main()
