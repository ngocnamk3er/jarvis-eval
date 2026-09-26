"""The benchmark registry.

Each benchmark module exposes two functions with a uniform signature:

    seed(suite) -> dict          download the dataset from HF, upload its
                                 corpus into the `eval-<suite>` workspace,
                                 wait for embedding. One-off.
    run(_cases, repeats, suite)  score it against the seeded workspace,
                 -> list[dict]   returning one {suite, case_id, metrics, meta}
                                 per query.  (`_cases` is unused — the data
                                 comes from HF, not a local file — it's just
                                 there so cli.py can call every runner the
                                 same way.)

`BENCHMARKS` maps a name to (seed_fn, run_fn); `cli.py` dispatches on it.
`BENCH_METRICS` tells `report.py` which metrics to put in the table and what
published number to show beside them.
"""
from functools import partial

from jarvis_eval.benchmarks import beir, gaia, hotpotqa, spreadsheetbench

# name -> (seed_fn, run_fn).  beir handles two suites, so its functions are
# partial-applied with the suite name.
BENCHMARKS = {
    "beir_scifact": (partial(beir.seed, "beir_scifact"), partial(beir.run, suite="beir_scifact")),
    "beir_nfcorpus": (partial(beir.seed, "beir_nfcorpus"), partial(beir.run, suite="beir_nfcorpus")),
    "hotpotqa": (hotpotqa.seed, hotpotqa.run),
    "gaia": (gaia.seed, gaia.run),
    # run_fn is None: this suite only seeds so far — scoring a workbook
    # needs a runner that copies it into the sandbox, which does not exist yet.
    "spreadsheetbench": (spreadsheetbench.seed, None),
}

# suite -> (metrics to show in the report, "reference: <published number>").
# Jarvis is on openai/text-embedding-3-large @1024 dims — compare to that
# model's published BEIR numbers (bge-m3 dense was ~0.64 / ~0.34).
BENCH_METRICS = {
    "beir_scifact": (["ndcg@10", "recall@10", "recall@100", "mrr@10"],
                     "text-embedding-3-large ndcg@10 ≈ 0.77 (bge-m3 was 0.64)"),
    "beir_nfcorpus": (["ndcg@10", "recall@10", "recall@100", "mrr@10"],
                      "text-embedding-3-large ndcg@10 ≈ 0.42 (bge-m3 was 0.34)"),
    "hotpotqa": (["support_recall@2", "support_recall@5", "both@2", "both@5",
                  "answer_em", "answer_f1", "support_read", "latency_s", "usd"],
                 "distractor-setting retrieval; strong RAG answer_f1 ≈ 0.6-0.8"),
    "gaia": (["score", "turns", "latency_s", "usd"],
             "level-1, no-attachment subset; SOTA overall ≈ 0.5-0.7"),
}
