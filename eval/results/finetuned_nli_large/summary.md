# Run `finetuned_nli_large`  (dev split, n=300)

> Real data.

- retriever `bm25`, verifier `nli:/content/drive/MyDrive/citecheck-models/ft-large`, k=3, sentences/paper=3, tau=0.55
- tau tuning: {'method': 'grid_max_macro_f1', 'split': 'train', 'n_claims': 200, 'tuned_macro_f1': 0.8185872674625231, 'leak': False, 'n_tuning_claims_seen_in_finetuning': 0}

## Claim-level (3-class: SUPPORT / CONTRADICT / NEI)
- **macro-F1 0.720**  95% CI [0.663, 0.774]  accuracy 0.727
- **false-SUPPORT rate 0.125**  SUPPORT precision 0.812  coverage 0.573
- claims excluded as MIXED gold: 0

| class | precision | recall | F1 | gold n |
|---|---|---|---|---|
| SUPPORT | 0.812 | 0.766 | 0.788 | 124 |
| CONTRADICT | 0.745 | 0.641 | 0.689 | 64 |
| NEI | 0.641 | 0.732 | 0.683 | 112 |

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
- label-only    P 0.577 R 0.589 F1 0.583
- label+rationale P 0.540 R 0.550 F1 0.545

## Where do the errors come from? (macro-F1 with progressively more oracle help)
- end-to-end 0.720 -> oracle paper 0.777 -> oracle paper+sentences 0.888
- lost to retrieval +0.056; lost to sentence selection +0.112; remaining gap (verifier) 0.112

**vs `zero_shot_nli`**: macro-F1 difference +0.118, 95% CI [+0.058, +0.177], P(this run better) = 1.00  (n=300)

**vs `zero_shot_nli_large`**: macro-F1 difference +0.095, 95% CI [+0.045, +0.145], P(this run better) = 1.00  (n=300)

## Cost / latency
retrieval 12.5 ms/claim, verification 131.4 ms/claim, model params 435,064,835 on cuda. Local model: $0 per request (compute only).

> With ~300 dev claims a 95% CI is several F1 points wide: differences smaller than the CI are not evidence.
