"""BEIR retrieval benchmarks (SciFact, NFCorpus) run against file-service.

Load the (small) corpus as one file per document, then run the standard
queries through /search/vector and score with the standard IR metrics.
Numbers are directly comparable to the bge-m3 rows on the MTEB / BEIR
leaderboards (SciFact nDCG@10 ≈ 0.64, NFCorpus ≈ 0.34 for bge-m3 dense).
"""
import math
from pathlib import PurePosixPath

from jarvis_eval.benchmarks import hf
from jarvis_eval.clients import files
from jarvis_eval.config import settings

SPECS = {
    "beir_scifact": {"ds": "BeIR/scifact", "qrels": "BeIR/scifact-qrels", "split": "test"},
    "beir_nfcorpus": {"ds": "BeIR/nfcorpus", "qrels": "BeIR/nfcorpus-qrels", "split": "test"},
}
FOLDER = "/bench"
TOP_K = 100          # unique docs scored
FETCH = 250          # chunks fetched before de-duping to docs
KS = (1, 5, 10, 100)


def user_for(suite: str) -> str:
    return f"{settings.EVAL_USERNAME}-{suite.replace('_', '-')}"


def _corpus(suite: str) -> list[dict]:
    return hf.load_rows(SPECS[suite]["ds"], "corpus", "corpus")


def _queries(suite: str) -> dict[str, str]:
    rows = hf.load_rows(SPECS[suite]["ds"], "queries", "queries")
    return {str(r["_id"]): r["text"] for r in rows}


def _qrels(suite: str) -> dict[str, dict[str, int]]:
    rows = hf.load_rows(SPECS[suite]["qrels"], "default", SPECS[suite]["split"])
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        qid, did, score = str(r["query-id"]), str(r["corpus-id"]), int(r["score"])
        out.setdefault(qid, {})[did] = score
    return out


# ---- seed -------------------------------------------------------------------
def seed(suite: str) -> dict:
    user = user_for(suite)
    corpus = _corpus(suite)
    if len(corpus) > settings.BENCH_MAX_DOCS:
        corpus = corpus[: settings.BENCH_MAX_DOCS]

    files.wipe(user=user)
    files.ensure_folder(f"{FOLDER}/{suite}", user=user)
    docs = []
    for row in corpus:
        did = str(row["_id"])
        body = (row.get("title", "") + "\n\n" + row.get("text", "")).strip()
        docs.append((f"{did}.txt", body or did))
    n = files.bulk_upload(docs, parent_path=f"{FOLDER}/{suite}", user=user)
    done = files.wait_for_indexing(timeout=max(300.0, n * 0.4), user=user)
    statuses: dict[str, int] = {}
    for e in done:
        statuses[e["indexing_status"]] = statuses.get(e["indexing_status"], 0) + 1
    return {"docs": n, "statuses": statuses}


# ---- metrics --------------------------------------------------------------
def _dcg(rels: list[int], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def _score_query(ranked_ids: list[str], rel: dict[str, int]) -> dict:
    gains = [rel.get(d, 0) for d in ranked_ids]
    ideal = sorted(rel.values(), reverse=True)
    n_rel = sum(1 for v in rel.values() if v > 0)
    out = {}
    for k in KS:
        idcg = _dcg(ideal, k)
        out[f"ndcg@{k}"] = (_dcg(gains, k) / idcg) if idcg else 0.0
        hit_k = sum(1 for g in gains[:k] if g > 0)
        out[f"recall@{k}"] = hit_k / n_rel if n_rel else 0.0
    first = next((i for i, g in enumerate(gains, 1) if g > 0), None)
    out["mrr@10"] = 1.0 / first if first and first <= 10 else 0.0
    return out


# ---- run ----------------------------------------------------------------
def run(_cases, repeats: int = 1, suite: str = "beir_scifact") -> list[dict]:
    user = user_for(suite)
    queries = _queries(suite)
    qrels = _qrels(suite)
    loaded = {PurePosixPath(e["path"]).stem for e in files.list_all(user=user) if e["type"] == "file"}

    results: list[dict] = []
    for qid, rel in qrels.items():
        if qid not in queries:
            continue
        # if the corpus was capped, skip queries whose gold docs aren't present
        if not any(d in loaded for d, s in rel.items() if s > 0):
            continue
        try:
            hits = files.search_vector(queries[qid], top_k=FETCH, user=user)
            # collapse multiple chunks of the same doc to its first appearance
            seen: set[str] = set()
            ranked = [s for h in hits
                      if (s := PurePosixPath(h["path"]).stem) not in seen and not seen.add(s)][:TOP_K]
            m = _score_query(ranked, rel)
            err = None
        except Exception as e:  # noqa: BLE001
            m, err = {}, f"{type(e).__name__}: {e}"
        results.append({
            "suite": suite, "case_id": qid, "repeat": 0,
            "metrics": m,
            "meta": {"query": queries[qid], "n_rel": sum(1 for v in rel.values() if v > 0),
                     "error": err},
        })
    return results
