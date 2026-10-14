# Run `zero_shot_nli_abstract`  (dev split, n=300)

> Real data.

- retriever `bm25`, verifier `nli:MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, k=3, sentences/paper=3, evidence unit=abstract, tau=0.85
- tau tuning: {'method': 'grid_max_macro_f1', 'split': 'train', 'n_claims': 200, 'tuned_macro_f1': 0.6145484949832776, 'leak': False, 'n_tuning_claims_seen_in_finetuning': 0}

## Claim-level (3-class: SUPPORT / CONTRADICT / NEI)
- **macro-F1 0.601**  95% CI [0.538, 0.659]  accuracy 0.610
- **false-SUPPORT rate 0.131**  SUPPORT precision 0.742  coverage 0.463
- claims excluded as MIXED gold: 0

| class | precision | recall | F1 | gold n |
|---|---|---|---|---|
| SUPPORT | 0.742 | 0.532 | 0.620 | 124 |
| CONTRADICT | 0.640 | 0.500 | 0.561 | 64 |
| NEI | 0.528 | 0.759 | 0.623 | 112 |

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
- label-only    P 0.481 R 0.416 F1 0.446
- label+rationale P 0.133 R 0.115 F1 0.123

## Where do the errors come from? (macro-F1 with progressively more oracle help)
- end-to-end 0.601 -> oracle paper 0.635 -> oracle paper+sentences 0.587
- lost to retrieval +0.033; lost to sentence selection -0.047; remaining gap (verifier) 0.413

**vs `zero_shot_nli`**: macro-F1 difference -0.001, 95% CI [-0.058, +0.046], P(this run better) = 0.48  (n=300)

## Cost / latency
retrieval 10.3 ms/claim, verification 28.4 ms/claim, model params 184,424,451 on cuda. Local model: $0 per request (compute only).

> With ~300 dev claims a 95% CI is several F1 points wide: differences smaller than the CI are not evidence.
