# Run `finetuned_nli_selector_v2_capped`  (dev split, n=300)

> Real data.

- retriever `bm25`, verifier `nli:hassanedwan/citecheck-nli-ft-base-sel`, k=3, sentences/paper=3, evidence unit=window, tau=0.5
- tau tuning: {'method': 'grid_max_macro_f1', 'split': 'train', 'n_claims': 200, 'tuned_macro_f1': 0.7612621925592022, 'tuned_false_support_rate': 0.08620689655172414, 'fs_cap': 0.1, 'met_fs_cap': True, 'leak': False, 'n_tuning_claims_seen_in_finetuning': 0}

## Claim-level (3-class: SUPPORT / CONTRADICT / NEI)
- **macro-F1 0.736**  95% CI [0.684, 0.784]  accuracy 0.747
- **false-SUPPORT rate 0.091**  SUPPORT precision 0.849  coverage 0.533
- claims excluded as MIXED gold: 0

| class | precision | recall | F1 | gold n |
|---|---|---|---|---|
| SUPPORT | 0.849 | 0.726 | 0.783 | 124 |
| CONTRADICT | 0.741 | 0.625 | 0.678 | 64 |
| NEI | 0.671 | 0.839 | 0.746 | 112 |

## Retrieval (claims with gold evidence: n=188, avg gold papers/claim 1.11)
| k | recall@k | precision@k |
|---|---|---|
| 1 | 0.722 | 0.734 |
| 3 | 0.841 | 0.294 |
| 5 | 0.882 | 0.186 |
| 10 | 0.941 | 0.102 |
| 20 | 0.958 | 0.052 |

MRR 0.810, nDCG@10 0.838. (precision@k is capped near 1/k when ~1 paper is relevant.)

## Abstract-level (SciFact-leaderboard style; NOT the official scorer)
- label-only    P 0.598 R 0.598 F1 0.598
- label+rationale P 0.569 R 0.569 F1 0.569

## Where do the errors come from? (macro-F1 with progressively more oracle help)
- end-to-end 0.736 -> oracle paper 0.782 -> oracle paper+sentences 0.866
- lost to retrieval +0.046; lost to sentence selection +0.085; remaining gap (verifier) 0.134

**vs `finetuned_nli`**: macro-F1 difference +0.042, 95% CI [-0.003, +0.084], P(this run better) = 0.97  (n=300)

## Sentence selection (`cross:hassanedwan/citecheck-sentence-selector`)
- a complete gold rationale is among the 3 selected sentences for 0.938 of 209 gold papers

## Cited-paper setting (the product: the user gives the paper; = oracle_doc)
- macro-F1 0.782  95% CI [0.728, 0.832]  false-SUPPORT 0.062
- vs `finetuned_nli`: +0.055, 95% CI [+0.012, +0.101] (n=300)

## Cost / latency
retrieval 10.0 ms/claim, verification 73.5 ms/claim, model params 184,424,451 on cuda. Local model: $0 per request (compute only).

> With ~300 dev claims a 95% CI is several F1 points wide: differences smaller than the CI are not evidence.
