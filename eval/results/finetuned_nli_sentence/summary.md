# Run `finetuned_nli_sentence`  (dev split, n=300)

> Real data.

- retriever `bm25`, verifier `nli:/content/drive/MyDrive/citecheck-models/ft-base-sentence`, k=3, sentences/paper=3, evidence unit=sentence, tau=0.1
- tau tuning: {'method': 'grid_max_macro_f1', 'split': 'train', 'n_claims': 200, 'tuned_macro_f1': 0.7672961138478379, 'leak': False, 'n_tuning_claims_seen_in_finetuning': 0}

## Claim-level (3-class: SUPPORT / CONTRADICT / NEI)
- **macro-F1 0.696**  95% CI [0.640, 0.748]  accuracy 0.707
- **false-SUPPORT rate 0.227**  SUPPORT precision 0.712  coverage 0.667
- claims excluded as MIXED gold: 0

| class | precision | recall | F1 | gold n |
|---|---|---|---|---|
| SUPPORT | 0.712 | 0.798 | 0.753 | 124 |
| CONTRADICT | 0.672 | 0.641 | 0.656 | 64 |
| NEI | 0.720 | 0.643 | 0.679 | 112 |

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
- label-only    P 0.420 R 0.612 F1 0.498
- label+rationale P 0.367 R 0.536 F1 0.436

## Where do the errors come from? (macro-F1 with progressively more oracle help)
- end-to-end 0.696 -> oracle paper 0.781 -> oracle paper+sentences 0.765
- lost to retrieval +0.085; lost to sentence selection -0.016; remaining gap (verifier) 0.235

**vs `finetuned_nli`**: macro-F1 difference +0.003, 95% CI [-0.052, +0.060], P(this run better) = 0.55  (n=300)

**vs `finetuned_nli_abstract`**: macro-F1 difference +0.017, 95% CI [-0.043, +0.080], P(this run better) = 0.71  (n=300)

**vs `zero_shot_nli`**: macro-F1 difference +0.093, 95% CI [+0.028, +0.157], P(this run better) = 1.00  (n=300)

## Cost / latency
retrieval 10.4 ms/claim, verification 193.4 ms/claim, model params 184,424,451 on cuda. Local model: $0 per request (compute only).

> With ~300 dev claims a 95% CI is several F1 points wide: differences smaller than the CI are not evidence.
