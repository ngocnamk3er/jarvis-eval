"""Retrieval suite — component test, no LLM.

Each case: a natural-language query + the file path(s) that actually answer
it. We run file-service semantic search (top_k=10) and score the ranking.
Cheap and fast, so this is the tightest feedback loop for RAG work.

Case shape (datasets/retrieval.jsonl):
  {"id": "...", "query": "...", "gold_paths": ["/research/x.md"], "note": "..."}
"""
import time

from jarvis_eval.clients import files
from jarvis_eval.metrics import retrieval_case_metrics

TOP_K = 10


def run(cases: list[dict], repeats: int = 1) -> list[dict]:
    results: list[dict] = []
    for case in cases:
        for rep in range(repeats):
            t0 = time.time()
            err = None
            try:
                hits = files.search_vector(case["query"], top_k=TOP_K)
                ranked = [h["path"] for h in hits]
                metrics = retrieval_case_metrics(ranked, case["gold_paths"])
                metrics["top1_score"] = round(hits[0]["score"], 4) if hits else 0.0
            except Exception as e:  # noqa: BLE001 - record, don't abort the suite
                err, ranked, metrics = f"{type(e).__name__}: {e}", [], {}
            results.append({
                "suite": "retrieval",
                "case_id": case["id"],
                "repeat": rep,
                "metrics": metrics,
                "meta": {
                    "query": case["query"],
                    "gold_paths": case["gold_paths"],
                    "ranked_paths": ranked[:TOP_K],
                    "latency_s": round(time.time() - t0, 3),
                    "error": err,
                },
            })
    return results
