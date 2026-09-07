"""BEIR retrieval benchmarks: SciFact and NFCorpus.

Pure retrieval — no agent, no LLM. Load the corpus as one file per document,
then for each standard query call file-service semantic search and score the
ranking with the standard IR metrics. The numbers line up with the bge-m3
*dense* rows on the MTEB / BEIR leaderboards (SciFact nDCG@10 ≈ 0.64,
NFCorpus ≈ 0.34), so a big gap would mean a bug in this file, not in Jarvis.

Three HF pieces per benchmark:
  corpus  {_id, title, text}          the documents
  queries {_id, text}                 all queries (train + test)
  qrels   {query-id, corpus-id, score} which doc is relevant to which query (test split)
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
TOP_K = 100          # unique docs actually scored
FETCH = 250          # chunks fetched from Qdrant before collapsing to docs
KS = (1, 5, 10, 100)


def user_for(suite: str) -> str:
    return f"{settings.EVAL_USERNAME}-{suite.replace('_', '-')}"      # eval-beir-scifact


def _corpus(suite: str) -> list[dict]:
    return hf.load_rows(SPECS[suite]["ds"], "corpus", "corpus")


def _queries(suite: str) -> dict[str, str]:
    return {str(r["_id"]): r["text"] for r in hf.load_rows(SPECS[suite]["ds"], "queries", "queries")}


def _qrels(suite: str) -> dict[str, dict[str, int]]:
    """{query_id: {doc_id: relevance}} for the test split. relevance > 0 == relevant."""
    rows = hf.load_rows(SPECS[suite]["qrels"], "default", SPECS[suite]["split"])
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        out.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    return out


# =========================================================================
# seed  —  jeval seed beir_scifact
# =========================================================================
def seed(suite: str) -> dict:
    user = user_for(suite)
    corpus = _corpus(suite)
    if len(corpus) > settings.BENCH_MAX_DOCS:
        corpus = corpus[: settings.BENCH_MAX_DOCS]

    files.wipe(user=user)
    files.ensure_folder(f"{FOLDER}/{suite}", user=user)
    # one file per doc, named "<doc_id>.txt", content = "title\n\ntext"
    docs = [(f'{row["_id"]}.txt',
             (row.get("title", "") + "\n\n" + row.get("text", "")).strip() or str(row["_id"]))
            for row in corpus]
    n = files.bulk_upload(docs, parent_path=f"{FOLDER}/{suite}", user=user)
    done = files.wait_for_indexing(timeout=max(300.0, n * 0.4), user=user)

    statuses: dict[str, int] = {}
    for e in done:
        statuses[e["indexing_status"]] = statuses.get(e["indexing_status"], 0) + 1
    return {"docs": n, "statuses": statuses}


# =========================================================================
# metrics
# =========================================================================
def _dcg(rels: list[int], k: int) -> float:
    """Discounted Cumulative Gain: sum of gains, each divided by log2(rank+1)."""
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def _score_query(ranked_ids: list[str], rel: dict[str, int]) -> dict:
    """One query's metrics. `ranked_ids` = retrieved doc ids best-first,
    `rel` = {doc_id: relevance} from qrels."""
    gains = [rel.get(d, 0) for d in ranked_ids]          # relevance at each retrieved position
    ideal = sorted(rel.values(), reverse=True)           # best possible ordering
    n_rel = sum(1 for v in rel.values() if v > 0)
    out = {}
    for k in KS:
        idcg = _dcg(ideal, k)
        out[f"ndcg@{k}"] = (_dcg(gains, k) / idcg) if idcg else 0.0
        out[f"recall@{k}"] = (sum(1 for g in gains[:k] if g > 0) / n_rel) if n_rel else 0.0
    first = next((i for i, g in enumerate(gains, 1) if g > 0), None)
    out["mrr@10"] = 1.0 / first if first and first <= 10 else 0.0
    return out


# =========================================================================
# run  —  jeval run --suite beir_scifact
# =========================================================================
def run(_cases, repeats: int = 1, suite: str = "beir_scifact") -> list[dict]:
    user = user_for(suite)
    queries = _queries(suite)
    qrels = _qrels(suite)
    # doc ids actually present in the workspace (in case BENCH_MAX_DOCS capped the corpus)
    loaded = {PurePosixPath(e["path"]).stem for e in files.list_all(user=user) if e["type"] == "file"}

    results: list[dict] = []
    for qid, rel in qrels.items():
        if qid not in queries:
            continue
        if not any(d in loaded for d, s in rel.items() if s > 0):
            continue    # this query's gold doc isn't loaded — skip, don't punish
        try:
            hits = files.search_vector(queries[qid], top_k=FETCH, user=user)
            # a doc can appear as several chunks — keep its first hit only, take top TOP_K docs
            seen: set[str] = set()
            ranked = [s for h in hits
                      if (s := PurePosixPath(h["path"]).stem) not in seen and not seen.add(s)][:TOP_K]
            m, err = _score_query(ranked, rel), None
        except Exception as e:  # noqa: BLE001
            m, err = {}, f"{type(e).__name__}: {e}"
        results.append({
            "suite": suite, "case_id": qid, "repeat": 0, "metrics": m,
            "meta": {"query": queries[qid], "n_rel": sum(1 for v in rel.values() if v > 0),
                     "error": err},
        })
    return results
