# Run `zero_shot_nli_large`  (dev split, n=300)

> Real data.

- retriever `bm25`, verifier `nli:MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, k=3, sentences/paper=3, tau=0.85
- tau tuning: {'method': 'grid_max_macro_f1', 'split': 'train', 'n_claims': 200, 'tuned_macro_f1': 0.7147481170868254, 'leak': False, 'n_tuning_claims_seen_in_finetuning': 0}

## Claim-level (3-class: SUPPORT / CONTRADICT / NEI)
- **macro-F1 0.625**  95% CI [0.567, 0.681]  accuracy 0.627
- **false-SUPPORT rate 0.074**  SUPPORT precision 0.829  coverage 0.450
- claims excluded as MIXED gold: 0

| class | precision | recall | F1 | gold n |
|---|---|---|---|---|
| SUPPORT | 0.829 | 0.508 | 0.630 | 124 |
| CONTRADICT | 0.644 | 0.594 | 0.618 | 64 |
| NEI | 0.527 | 0.777 | 0.628 | 112 |

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
- label-only    P 0.557 R 0.445 F1 0.495
- label+rationale P 0.527 R 0.421 F1 0.468

## Where do the errors come from? (macro-F1 with progressively more oracle help)
- end-to-end 0.625 -> oracle paper 0.685 -> oracle paper+sentences 0.677
- lost to retrieval +0.060; lost to sentence selection -0.008; remaining gap (verifier) 0.323

**vs `zero_shot_nli`**: macro-F1 difference +0.023, 95% CI [-0.031, +0.078], P(this run better) = 0.80  (n=300)

## Cost / latency
retrieval 10.0 ms/claim, verification 33.1 ms/claim, model params 435,064,835 on cuda. Local model: $0 per request (compute only).

> With ~300 dev claims a 95% CI is several F1 points wide: differences smaller than the CI are not evidence.
