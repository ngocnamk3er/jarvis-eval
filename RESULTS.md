# jarvis-eval — current scores

_Generated from `datasets/baseline.json` · agent model `deepseek/deepseek-v4-flash` · 2026-09-07T16:21:03Z_

> embeddings: openai/text-embedding-3-large @1024 dims (was baai/bge-m3). beir_* full test sets; hotpotqa retrieval on 300 sampled Qs. 2026-09-07.

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
| answer_em | 0.580 |
| answer_f1 | 0.710 |
| answer_n | 45 |
| retrieval_n_cases | 300 |
| latency_p95_s | 282 |
| usd_agent_total | 0.300 |
| _cases_ | 300 (5 errored) |

_answers from a bge-m3-era 45/50 agent run; retrieval since re-measured with text-embedding-3-large_

