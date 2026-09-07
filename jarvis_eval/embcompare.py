"""`jeval embcompare <model>` — score a candidate embedding model on BEIR
*offline*, without touching the cluster.

Use it before swapping jarvis-file-service's EMBEDDING_MODEL: embed the
SciFact / NFCorpus corpus + test queries with the candidate, brute-force
cosine, and print nDCG@10 / recall@10 / MRR@10 next to the committed
baseline. If it doesn't beat the baseline there's no point deploying it.

Reuses the BEIR data loaders + scoring from `benchmarks/beir.py`; the only
new thing is talking to an /embeddings endpoint directly (OpenRouter by
default — set EMBEDDING_API_KEY in .env).
"""
from __future__ import annotations

import time

import httpx
import numpy as np
from rich.console import Console

from jarvis_eval.benchmarks import beir
from jarvis_eval.config import settings
from jarvis_eval.report import _load_baseline

console = Console()
_BATCH = 128   # inputs per /embeddings call (keeps request size sane)


def _embed_all(model: str, texts: list[str], dims: int | None) -> np.ndarray:
    """(len(texts), D) L2-normalised float32 matrix."""
    if not settings.EMBEDDING_API_KEY:
        raise RuntimeError("set EMBEDDING_API_KEY in .env (an OpenRouter key)")
    out: list[list[float]] = []
    with httpx.Client(base_url=settings.EMBEDDING_BASE_URL,
                      headers={"Authorization": f"Bearer {settings.EMBEDDING_API_KEY}"},
                      timeout=120) as c:
        for i in range(0, len(texts), _BATCH):
            batch = texts[i:i + _BATCH]
            body: dict = {"model": model, "input": batch}
            if dims:
                body["dimensions"] = dims
            for attempt in range(4):
                try:
                    r = c.post("/embeddings", json=body)
                    r.raise_for_status()
                    out.extend(d["embedding"] for d in r.json()["data"])
                    break
                except httpx.HTTPError:
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
            console.print(f"  embedded {min(i + _BATCH, len(texts))}/{len(texts)}", end="\r")
    console.print(" " * 40, end="\r")
    m = np.asarray(out, dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def run(model: str, dims: int | None, suites: list[str]) -> None:
    baseline = (_load_baseline() or {}).get("suites", {})
    for suite in suites:
        if suite not in beir.SPECS:
            console.print(f"[yellow]skip {suite}[/] — embcompare only does BEIR suites")
            continue
        console.rule(f"{suite}  ·  {model}" + (f"  (dims={dims})" if dims else ""))

        corpus = beir._corpus(suite)
        if len(corpus) > settings.BENCH_MAX_DOCS:
            corpus = corpus[: settings.BENCH_MAX_DOCS]
        doc_ids = [str(r["_id"]) for r in corpus]
        doc_texts = [(r.get("title", "") + "\n\n" + r.get("text", "")).strip() or i
                     for r, i in zip(corpus, doc_ids)]
        queries = beir._queries(suite)
        qrels = beir._qrels(suite)
        loaded = set(doc_ids)
        test_qids = [q for q, rel in qrels.items()
                     if q in queries and any(d in loaded for d, s in rel.items() if s > 0)]

        console.print(f"  corpus {len(corpus)} · test queries {len(test_qids)} · embedding…")
        started = time.time()
        dmat = _embed_all(model, doc_texts, dims)                       # (N, D)
        qmat = _embed_all(model, [queries[q] for q in test_qids], dims)  # (Q, D)

        # cosine (already normalised) -> top-100 doc ids per query -> BEIR score
        per_q: list[dict] = []
        sims = qmat @ dmat.T                                            # (Q, N)
        top = np.argsort(-sims, axis=1)[:, :100]
        for row, qid in zip(top, test_qids):
            ranked = [doc_ids[j] for j in row]
            per_q.append(beir._score_query(ranked, qrels[qid]))

        agg = {k: round(float(np.mean([p[k] for p in per_q])), 4)
               for k in ("ndcg@10", "recall@10", "recall@100", "mrr@10")}
        b = baseline.get(suite, {})
        console.print(f"  [dim]{time.time() - started:.0f}s[/]")
        console.print("\n  | metric | candidate | baseline (bge-m3) | Δ |")
        console.print("  |---|---|---|---|")
        for k, v in agg.items():
            bv = b.get(k)
            d = f"{v - bv:+.3f}" if isinstance(bv, (int, float)) else "—"
            console.print(f"  | {k} | [bold]{v:.3f}[/] | {bv if bv is not None else '—'} | {d} |")
        gate = agg["ndcg@10"] - (b.get("ndcg@10") or 0)
        verdict = "[green]worth deploying[/]" if gate > 0.02 else "[yellow]marginal / not worth it[/]"
        console.print(f"\n  → nDCG@10 {gate:+.3f} vs baseline — {verdict}")
