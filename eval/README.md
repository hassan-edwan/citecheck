# eval/ : the evaluation harness

Owner: Data & Evaluation (Sahil).  **One command produces every number we report.**

```bash
# 0. one-time: get SciFact (writes data/scifact/*.jsonl + MANIFEST.json)
python -m data.fetch_scifact

# 1. the baseline ladder, on DEV. Run in this order; each compares itself with the previous one.
python -m eval.run_eval --name majority      --retriever bm25 --verifier majority
python -m eval.run_eval --name lexical       --retriever bm25 --verifier lexical --compare-with eval/results/majority
python -m eval.run_eval --name zero_shot_nli --retriever bm25 --verifier nli --oracle \
                        --compare-with eval/results/majority --compare-with eval/results/lexical --promote

# 2. what went wrong, and the figures
python -m eval.error_analysis --run eval/results/zero_shot_nli
python -m eval.plots --run zero_shot_nli

# 3. improvements (each is one row in the ablation table)
python -m eval.run_eval --name hybrid_nli --retriever hybrid --verifier nli --compare-with eval/results/zero_shot_nli
python -m eval.finetune_nli --out models/citecheck-nli-ft            # GPU recommended: notebooks/finetune_colab.ipynb
python -m eval.run_eval --name finetuned_nli --retriever bm25 --verifier nli --model models/citecheck-nli-ft \
                        --oracle --compare-with eval/results/zero_shot_nli
```

**Evidence units** (`--evidence-unit`, and `--unit` when fine-tuning): what the verifier reads per paper.
`window` (default, Session 5) = the top `--sentences` sentences joined; `abstract` = the whole abstract, no selection;
`sentence` = each sentence on its own, the paper takes its most decisive sentence and those sentences are the evidence
shown. Promoting a run records its unit in `deployed_config.json`, so the app reads papers the same way.

**Stage 2, hybrid reasoner** (`--reasoner <local LLM> --reasoner-mode hybrid|checks|llm --escalate not-confident|all`):
claims stage 1 is not confident about are re-judged by `src/reasoning.py` (a small local LLM extracts sentences,
quantities, p-values and CIs as JSON; Python grounds every number in the text and checks significance and quantities;
a failed check can make a verdict CONTRADICT, a check never creates a SUPPORT). LLM outputs are cached in
`eval/cache/`, so the three modes of one model share one generation pass. Not promotable yet (the app has no stage 2).

**Cited-paper setting**: with `--oracle`, `results.json["cited_paper"]` and `predictions_cited_paper.jsonl` hold the
metric for the product's own question (the user supplies the paper), with a CI and paired comparisons.

Smoke test with 20 claims first (`--limit 20`; the run is flagged *partial* and cannot be promoted).
The first NLI run downloads the model from Hugging Face (~700 MB) and the first dense/hybrid run embeds the
corpus once (cached in `data/cache/`).

## What is measured

| Question | Metric | Where |
|---|---|---|
| Does the system judge citations correctly? (**north star**) | claim-level 3-class **macro-F1** on dev, with 95% bootstrap CI | `claim_level` |
| Is it dangerously over-confident? | **false-SUPPORT rate** (gold is not SUPPORT, we said SUPPORT), SUPPORT precision, coverage | `claim_level` |
| Do we find the right paper? | Recall@k, Precision@k (capped near 1/k), MRR, nDCG@10 | `retrieval` |
| Can we compare with the literature? | abstract-level P/R/F1, label-only and label+rationale (SciFact-leaderboard style, **not** the official scorer) | `abstract_level` |
| Where do errors come from? | end-to-end vs oracle-paper vs oracle-paper+sentences | `error_attribution` |
| Is A really better than B? | paired bootstrap over claims: difference, 95% CI, P(A better) | `paired_comparisons` |
| What does a request cost? | ms per claim (retrieval, verification), model parameters, device; local model = $0 | `timing` |

## Rules that keep us honest

* **tau (the abstain threshold) is tuned on `train`, reported on `dev`.** Tuning on dev is refused unless `--allow-leak`.
* **A fine-tuned model never sees the claims tau is tuned on.** `eval.finetune_nli` holds out the same 200 train claims
  `run_eval` tunes tau on (`run_eval.tau_tuning_claims`, same `--tune-n`/`--seed`) and lists them in the checkpoint's
  `train_log.json`; `run_eval --model <local dir>` refuses to tune on any other train claim unless `--allow-leak`.
* `claims_test.jsonl` has no public labels. We never use it.
* Runs on `tests/fixtures/**` are flagged `synthetic_fixture` and can never be promoted or quoted.
* `--promote` writes `eval/results/deployed_config.json`; the app loads exactly that configuration and shows its dev score.
  That is how "the model we report is the model in the demo" stays true.
* With n=300, differences smaller than the CI are noise. Say so in the report; do not claim a win the CI does not support.
* Publish negative results: if the fine-tuned model loses to zero-shot, that is a result (the guidelines credit it).

## Sanity checks before you trust a number

* BM25 recall@10 on SciFact should be clearly above 0.5. If it is near 0, the loader or tokenizer is broken.
* `majority` macro-F1 should be low (it only ever predicts NEI; ~0.2 at best).
* `oracle_doc` should be >= `end_to_end` unless retrieval already finds every gold paper. `oracle_rationale` is usually >= `oracle_doc` for a good verifier, but a weak or quirky one can score lower on gold rationales (the lexical baseline does); a LARGE unexplained drop is a red flag for a bug in sentence handling.
* Re-run the same command twice: the numbers must be identical (seeded; the cache makes this fast).

## Files written per run (`eval/results/<name>/`)

`results.json` (everything, with provenance), `predictions.jsonl` (per-claim predictions and probabilities),
`summary.md` (human-readable), later `errors.md` / `errors.csv`.  `eval/results/LATEST.md` is the table of all runs.
**Commit `eval/results/` (small) but not `eval/cache/` or `models/`.**
