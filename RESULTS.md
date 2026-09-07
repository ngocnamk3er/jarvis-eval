# jarvis-eval — current scores

_Generated from `datasets/baseline.json` · agent model `deepseek/deepseek-v4-flash` · 2026-09-07T17:33:57Z_

> embeddings: openai/text-embedding-3-large @1024 dims (was baai/bge-m3, 2026-09-07). beir_* full test sets; hotpotqa retrieval 300 sampled Qs + answers 47/50.

## beir_scifact

_reference: text-embedding-3-large ndcg@10 ≈ 0.77 (bge-m3 was 0.64)_

| metric | value |
|---|---|
| ndcg@10 | 0.769 |
| recall@10 | 0.895 |
| recall@100 | 0.977 |
| mrr@10 | 0.736 |
| _cases_ | 300 (0 errored) |

## beir_nfcorpus

_reference: text-embedding-3-large ndcg@10 ≈ 0.42 (bge-m3 was 0.34)_

| metric | value |
|---|---|
| ndcg@10 | 0.410 |
| recall@10 | 0.201 |
| recall@100 | 0.385 |
| mrr@10 | 0.621 |
| _cases_ | 323 (0 errored) |

## hotpotqa

_reference: distractor-setting retrieval; strong RAG answer_f1 ≈ 0.6-0.8_

| metric | value |
|---|---|
| support_recall@2 | 0.733 |
| support_recall@5 | 0.887 |
| both@2 | 0.513 |
| both@5 | 0.777 |
| answer_em | 0.620 |
| answer_f1 | 0.753 |
| support_read | 0.960 |
| latency_s | 56.832 |
| usd | 0.004 |
| latency_p95 | 184.540 |
| usd_total | 0.213 |
| _cases_ | 300 (3 errored) |

