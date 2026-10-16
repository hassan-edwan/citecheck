# Run `finetuned_nli_selector_v2`  (dev split, n=300)

> Real data.

- retriever `bm25`, verifier `nli:/content/drive/MyDrive/citecheck-models/ft-base-sel`, k=3, sentences/paper=3, evidence unit=window, tau=0.1
- tau tuning: {'method': 'grid_max_macro_f1', 'split': 'train', 'n_claims': 200, 'tuned_macro_f1': 0.7719159427768699, 'leak': False, 'n_tuning_claims_seen_in_finetuning': 0}

## Claim-level (3-class: SUPPORT / CONTRADICT / NEI)
- **macro-F1 0.729**  95% CI [0.676, 0.780]  accuracy 0.743
- **false-SUPPORT rate 0.142**  SUPPORT precision 0.800  coverage 0.630
- claims excluded as MIXED gold: 0

| class | precision | recall | F1 | gold n |
|---|---|---|---|---|
| SUPPORT | 0.800 | 0.806 | 0.803 | 124 |
| CONTRADICT | 0.656 | 0.656 | 0.656 | 64 |
| NEI | 0.730 | 0.723 | 0.726 | 112 |

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
- label-only    P 0.514 R 0.636 F1 0.568
- label+rationale P 0.490 R 0.608 F1 0.543

## Where do the errors come from? (macro-F1 with progressively more oracle help)
- end-to-end 0.729 -> oracle paper 0.783 -> oracle paper+sentences 0.842
- lost to retrieval +0.055; lost to sentence selection +0.059; remaining gap (verifier) 0.158

**vs `finetuned_nli`**: macro-F1 difference +0.035, 95% CI [-0.014, +0.083], P(this run better) = 0.91  (n=300)

**vs `finetuned_nli_selector`**: macro-F1 difference +0.016, 95% CI [-0.025, +0.057], P(this run better) = 0.79  (n=300)

## Sentence selection (`cross:/content/drive/MyDrive/citecheck-models/selector`)
- a complete gold rationale is among the 3 selected sentences for 0.938 of 209 gold papers

## Cited-paper setting (the product: the user gives the paper; = oracle_doc)
- macro-F1 0.783  95% CI [0.732, 0.832]  false-SUPPORT 0.097
- vs `finetuned_nli`: +0.056, 95% CI [+0.005, +0.103] (n=300)
- vs `finetuned_nli_selector`: +0.037, 95% CI [-0.006, +0.077] (n=300)

## Cost / latency
retrieval 16.5 ms/claim, verification 89.6 ms/claim, model params 184,424,451 on cuda. Local model: $0 per request (compute only).

> With ~300 dev claims a 95% CI is several F1 points wide: differences smaller than the CI are not evidence.
