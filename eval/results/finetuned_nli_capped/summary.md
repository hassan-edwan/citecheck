# Run `finetuned_nli_capped`  (dev split, n=300)

> Real data.

- retriever `bm25`, verifier `nli:hassanedwan/citecheck-nli-ft-base`, k=3, sentences/paper=3, evidence unit=window, tau=0.5
- tau tuning: {'method': 'grid_max_macro_f1', 'split': 'train', 'n_claims': 200, 'tuned_macro_f1': 0.8097543482158867, 'tuned_false_support_rate': 0.05172413793103448, 'fs_cap': 0.1, 'met_fs_cap': True, 'leak': False, 'n_tuning_claims_seen_in_finetuning': 0}

## Claim-level (3-class: SUPPORT / CONTRADICT / NEI)
- **macro-F1 0.693**  95% CI [0.636, 0.749]  accuracy 0.713
- **false-SUPPORT rate 0.068**  SUPPORT precision 0.875  coverage 0.463
- claims excluded as MIXED gold: 0

| class | precision | recall | F1 | gold n |
|---|---|---|---|---|
| SUPPORT | 0.875 | 0.677 | 0.764 | 124 |
| CONTRADICT | 0.744 | 0.500 | 0.598 | 64 |
| NEI | 0.609 | 0.875 | 0.718 | 112 |

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
- label-only    P 0.626 R 0.522 F1 0.569
- label+rationale P 0.580 R 0.483 F1 0.527

## Where do the errors come from? (macro-F1 with progressively more oracle help)
- end-to-end 0.693 -> oracle paper 0.727 -> oracle paper+sentences 0.849
- lost to retrieval +0.034; lost to sentence selection +0.122; remaining gap (verifier) 0.151

**vs `finetuned_nli`**: macro-F1 difference +0.000, 95% CI [+0.000, +0.000], P(this run better) = 0.00  (n=300)

## Sentence selection (`lexical`)
- a complete gold rationale is among the 3 selected sentences for 0.794 of 209 gold papers

## Cited-paper setting (the product: the user gives the paper; = oracle_doc)
- macro-F1 0.727  95% CI [0.668, 0.782]  false-SUPPORT 0.045
- vs `finetuned_nli`: +0.000, 95% CI [+0.000, +0.000] (n=300)

## Cost / latency
retrieval 10.5 ms/claim, verification 37.7 ms/claim, model params 184,424,451 on cuda. Local model: $0 per request (compute only).

> With ~300 dev claims a 95% CI is several F1 points wide: differences smaller than the CI are not evidence.
