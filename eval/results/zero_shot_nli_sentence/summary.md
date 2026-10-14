# Run `zero_shot_nli_sentence`  (dev split, n=300)

> Real data.

- retriever `bm25`, verifier `nli:MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, k=3, sentences/paper=3, evidence unit=sentence, tau=0.3
- tau tuning: {'method': 'grid_max_macro_f1', 'split': 'train', 'n_claims': 200, 'tuned_macro_f1': 0.46376199569134346, 'leak': False, 'n_tuning_claims_seen_in_finetuning': 0}

## Claim-level (3-class: SUPPORT / CONTRADICT / NEI)
- **macro-F1 0.454**  95% CI [0.401, 0.509]  accuracy 0.460
- **false-SUPPORT rate 0.091**  SUPPORT precision 0.778  coverage 0.833
- claims excluded as MIXED gold: 0

| class | precision | recall | F1 | gold n |
|---|---|---|---|---|
| SUPPORT | 0.778 | 0.452 | 0.571 | 124 |
| CONTRADICT | 0.303 | 0.844 | 0.446 | 64 |
| NEI | 0.560 | 0.250 | 0.346 | 112 |

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
- label-only    P 0.190 R 0.450 F1 0.267
- label+rationale P 0.147 R 0.349 F1 0.207

## Where do the errors come from? (macro-F1 with progressively more oracle help)
- end-to-end 0.454 -> oracle paper 0.574 -> oracle paper+sentences 0.576
- lost to retrieval +0.119; lost to sentence selection +0.002; remaining gap (verifier) 0.424

**vs `zero_shot_nli`**: macro-F1 difference -0.148, 95% CI [-0.207, -0.095], P(this run better) = 0.00  (n=300)

## Cost / latency
retrieval 9.7 ms/claim, verification 52.2 ms/claim, model params 184,424,451 on cuda. Local model: $0 per request (compute only).

> With ~300 dev claims a 95% CI is several F1 points wide: differences smaller than the CI are not evidence.
