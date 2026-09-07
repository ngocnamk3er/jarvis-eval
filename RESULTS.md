# jarvis-eval — current scores

_Generated from `datasets/baseline.json` · agent model `deepseek/deepseek-v4-flash` · 2026-09-07T04:42:06Z_

> beir_*: file-service bge-m3 dense retrieval. hotpotqa: retrieval on 300 sampled Qs, answers on 45/50 (5 agent timeouts). 2026-09-07.

## beir_scifact

_reference: bge-m3 dense ≈ ndcg@10 0.64_

| metric | value |
|---|---|
| ndcg@10 | 0.654 |
| recall@10 | 0.797 |
| recall@100 | 0.908 |
| mrr@10 | 0.619 |
| _cases_ | 300 (0 errored) |

## beir_nfcorpus

_reference: bge-m3 dense ≈ ndcg@10 0.34_

| metric | value |
|---|---|
| ndcg@10 | 0.316 |
| recall@10 | 0.150 |
| recall@100 | 0.282 |
| mrr@10 | 0.524 |
| _cases_ | 323 (0 errored) |

## hotpotqa

_reference: distractor-setting retrieval; strong RAG answer_f1 ≈ 0.6-0.8_

| metric | value |
|---|---|
| support_recall@2 | 0.723 |
| support_recall@5 | 0.855 |
| both@2 | 0.493 |
| both@5 | 0.720 |
| answer_em | 0.580 |
| answer_f1 | 0.710 |
| answer_n | 45 |
| retrieval_n_cases | 300 |
| latency_p95_s | 282 |
| usd_agent_total | 0.300 |
| _cases_ | 300 (5 errored) |

