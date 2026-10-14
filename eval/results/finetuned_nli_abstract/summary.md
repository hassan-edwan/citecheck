# Run `finetuned_nli_abstract`  (dev split, n=300)

> Real data.

- retriever `bm25`, verifier `nli:/content/drive/MyDrive/citecheck-models/ft-base-abstract`, k=3, sentences/paper=3, evidence unit=abstract, tau=0.45
- tau tuning: {'method': 'grid_max_macro_f1', 'split': 'train', 'n_claims': 200, 'tuned_macro_f1': 0.7792651405653866, 'leak': False, 'n_tuning_claims_seen_in_finetuning': 0}

## Claim-level (3-class: SUPPORT / CONTRADICT / NEI)
- **macro-F1 0.679**  95% CI [0.625, 0.733]  accuracy 0.687
- **false-SUPPORT rate 0.131**  SUPPORT precision 0.781  coverage 0.543
- claims excluded as MIXED gold: 0

| class | precision | recall | F1 | gold n |
|---|---|---|---|---|
| SUPPORT | 0.781 | 0.661 | 0.716 | 124 |
| CONTRADICT | 0.672 | 0.609 | 0.639 | 64 |
| NEI | 0.620 | 0.759 | 0.683 | 112 |

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
- label-only    P 0.513 R 0.550 F1 0.531
- label+rationale P 0.129 R 0.139 F1 0.134

## Where do the errors come from? (macro-F1 with progressively more oracle help)
- end-to-end 0.679 -> oracle paper 0.741 -> oracle paper+sentences 0.870
- lost to retrieval +0.062; lost to sentence selection +0.129; remaining gap (verifier) 0.130

**vs `finetuned_nli`**: macro-F1 difference -0.014, 95% CI [-0.067, +0.044], P(this run better) = 0.29  (n=300)

**vs `zero_shot_nli`**: macro-F1 difference +0.077, 95% CI [+0.017, +0.130], P(this run better) = 0.99  (n=300)

## Cost / latency
retrieval 10.1 ms/claim, verification 108.8 ms/claim, model params 184,424,451 on cuda. Local model: $0 per request (compute only).

> With ~300 dev claims a 95% CI is several F1 points wide: differences smaller than the CI are not evidence.
