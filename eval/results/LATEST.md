# All evaluation runs

| run | split | retriever | verifier | tau | macro-F1 [95% CI] | false-SUPPORT | recall@5 | flags |
|---|---|---|---|---|---|---|---|---|
| majority | dev | bm25 | majority | 0.5 | 0.181 [0.162, 0.200] | 0.000 | 0.882 | - |
| lexical | dev | bm25 | lexical | 0.65 | 0.520 [0.456, 0.580] | 0.222 | 0.882 | - |
| zero_shot_nli | dev | bm25 | nli:MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | 0.6 | 0.603 [0.545, 0.660] | 0.108 | 0.882 | - |
| hybrid_nli | dev | hybrid(bm25+dense) | nli:MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | 0.6 | 0.587 [0.526, 0.641] | 0.148 | 0.909 | - |
| zero_shot_nli_large | dev | bm25 | nli:MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli | 0.85 | 0.625 [0.567, 0.681] | 0.074 | 0.882 | - |
| finetuned_nli | dev | bm25 | nli:/content/drive/MyDrive/citecheck-models/ft-base | 0.5 | 0.693 [0.636, 0.749] | 0.068 | 0.882 | - |
| finetuned_nli_large | dev | bm25 | nli:/content/drive/MyDrive/citecheck-models/ft-large | 0.55 | 0.720 [0.663, 0.774] | 0.125 | 0.882 | - |
