"""Standard third-party benchmarks, loaded from HuggingFace at seed time.

Each entry: seed(suite) builds the corpus in a dedicated `eval-<suite>`
workspace; run(cases, repeats, suite) scores it. `cases` is unused (the
dataset comes from HF), kept for a uniform runner signature.
"""
from functools import partial

from jarvis_eval.benchmarks import beir, gaia, hotpotqa

# suite -> (seed_fn, run_fn)
BENCHMARKS = {
    "beir_scifact": (partial(beir.seed, "beir_scifact"), partial(beir.run, suite="beir_scifact")),
    "beir_nfcorpus": (partial(beir.seed, "beir_nfcorpus"), partial(beir.run, suite="beir_nfcorpus")),
    "hotpotqa": (hotpotqa.seed, hotpotqa.run),
    "gaia": (gaia.seed, gaia.run),
}

# for report.py aggregation — which metrics to surface per suite, and the
# published reference point (bge-m3 dense / strong-agent numbers)
BENCH_METRICS = {
    "beir_scifact": (["ndcg@10", "recall@10", "recall@100", "mrr@10"],
                     "bge-m3 dense ≈ ndcg@10 0.64"),
    "beir_nfcorpus": (["ndcg@10", "recall@10", "recall@100", "mrr@10"],
                      "bge-m3 dense ≈ ndcg@10 0.34"),
    "hotpotqa": (["support_recall@2", "support_recall@5", "both@2", "both@5",
                  "answer_em", "answer_f1", "support_read", "latency_s", "usd"],
                 "distractor-setting retrieval; strong RAG answer_f1 ≈ 0.6-0.8"),
    "gaia": (["score", "turns", "latency_s", "usd"],
             "level-1, no-attachment subset; SOTA overall ≈ 0.5-0.7"),
}
