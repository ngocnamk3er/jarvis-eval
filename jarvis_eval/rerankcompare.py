"""`jeval rerankcompare <reranker>` — measure what a reranker adds, offline.

The retrieval half of jarvis-eval today is a single dense bi-encoder pass
(`openai/text-embedding-3-large` @1024, cosine in Qdrant). A *reranker* is the
standard next move: take the top-N of that dense list and re-score every
`(query, passage)` pair with a model that looks at the two *together*, then
re-sort.

Why it should help — it attacks the one thing a bi-encoder structurally
cannot: a passage that sits close to the query in embedding space but is the
*wrong entity*. HotpotQA's hard distractors are exactly this — for "when did
the Tivoli park open?" the pool holds "Tivoli One (a live album recorded at
the Tivolis Koncertsal)", which a bi-encoder ranks near the top on surface
match alone.

Two backends:

  --backend openrouter   (default)  listwise LLM reranking, RankGPT-style:
        hand the model the query + a numbered window of candidates, ask for
        the permutation `[3] > [1] > ...`. One /chat/completions call per
        query (per window). Fits Jarvis's stack — no model service to run,
        just one more OpenRouter call in the search path. Costs money; the
        run prints a live $ total and stops at --max-cost.

  --backend local                   a CPU cross-encoder via `transformers`
        (see requirements-rerank.txt). `cross-encoder/ms-marco-MiniLM-L-6-v2`
        is the canonical BEIR reranker (22M params, ~300 pairs/s). Free,
        offline, but English-only and another artefact to deploy.

Like `embcompare`, the *dense* stage never touches the cluster: embed the
cached corpus + queries with the production model, cosine, take the top
FETCH — cached to disk so trying another reranker is free.

Suites: beir_scifact / beir_nfcorpus (nDCG@10 …) · hotpotqa (support_recall@k,
both@k — retrieval half only; answer EM/F1 needs the agent, that's
`jeval run --suite hotpotqa`).
"""
from __future__ import annotations

import os
import re
import time

import httpx
import numpy as np
from rich.console import Console

from jarvis_eval.benchmarks import beir, hf, hotpotqa
from jarvis_eval.config import settings
from jarvis_eval.embcompare import _embed_all
from jarvis_eval.report import _load_baseline

console = Console()

PROD_MODEL = "openai/text-embedding-3-large"  # what the cluster embeds with — keep in sync
PROD_DIMS = 1024
_EMB_CACHE = hf.CACHE / "rerank_emb"          # <suite>.npz: doc_ids, qids, dmat, qmat

# defaults differ by backend: an LLM call reranking 20 passages is one thing,
# a cross-encoder chewing 100 pairs is cheap.
_FETCH_DEFAULT = {"openrouter": 20, "local": 100}
_WINDOW = 20                                  # listwise window size (RankGPT)


# =========================================================================
# per-suite data  ->  (doc_id -> text), (qid -> text), (qid -> {gold doc ids})
# =========================================================================
def _beir_data(suite: str):
    corpus = beir._corpus(suite)
    if len(corpus) > settings.BENCH_MAX_DOCS:
        corpus = corpus[: settings.BENCH_MAX_DOCS]
    docs = {str(r["_id"]): (r.get("title", "") + "\n\n" + r.get("text", "")).strip() or str(r["_id"])
            for r in corpus}
    queries = beir._queries(suite)
    qrels = beir._qrels(suite)
    loaded = set(docs)
    gold = {q: {d for d, s in rel.items() if s > 0}
            for q, rel in qrels.items()
            if q in queries and any(d in loaded for d, s in rel.items() if s > 0)}
    return docs, {q: queries[q] for q in gold}, gold, qrels


def _hotpot_data():
    """Merge every sampled question's 10-paragraph pool into one corpus,
    de-duped by title (doc_id == article title) — exactly hotpotqa.seed()."""
    rows = hotpotqa._sample()
    paragraphs: dict[str, str] = {}
    queries: dict[str, str] = {}
    gold: dict[str, set[str]] = {}
    for row in rows:
        ctx = row["context"]
        titles = ctx["title"] if isinstance(ctx, dict) else [t for t, _ in ctx]
        sents = ctx["sentences"] if isinstance(ctx, dict) else [s for _, s in ctx]
        for t, s in zip(titles, sents):
            paragraphs.setdefault(t, hotpotqa._para_text(t, list(s)))
        queries[row["id"]] = row["question"]
        gold[row["id"]] = set(hotpotqa._support_titles(row))
    return paragraphs, queries, gold


def _score_hotpot(ranked_ids: list[str], gold_titles: set[str]) -> dict:
    out = {}
    for k in (2, 5, 10):
        got = set(ranked_ids[:k]) & gold_titles
        out[f"support_recall@{k}"] = len(got) / len(gold_titles) if gold_titles else 0.0
        out[f"both@{k}"] = 1.0 if gold_titles and len(got) == len(gold_titles) else 0.0
    return out


# =========================================================================
# stage 1 — dense retrieval (cached, offline)
# =========================================================================
def _dense(suite: str, docs: dict[str, str], queries: dict[str, str], fetch: int):
    _EMB_CACHE.mkdir(parents=True, exist_ok=True)
    cache = _EMB_CACHE / f"{suite}.npz"
    doc_ids, qids = list(docs), list(queries)

    dmat = qmat = None
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["doc_ids"]) == doc_ids and list(z["qids"]) == qids:
            dmat, qmat = z["dmat"], z["qmat"]
            console.print(f"  dense embeddings: cache hit ([dim]{cache.name}[/])")
    if dmat is None:
        console.print(f"  embedding {len(doc_ids)} docs + {len(qids)} queries "
                      f"[dim]({PROD_MODEL} @{PROD_DIMS})[/]…")
        dmat = _embed_all(PROD_MODEL, [docs[d] for d in doc_ids], PROD_DIMS)
        qmat = _embed_all(PROD_MODEL, [queries[q] for q in qids], PROD_DIMS)
        np.savez(cache, doc_ids=np.array(doc_ids, dtype=object),
                 qids=np.array(qids, dtype=object), dmat=dmat, qmat=qmat)

    sims = qmat @ dmat.T                                  # (Q, N) cosine
    top = np.argsort(-sims, axis=1)[:, :fetch]
    return doc_ids, qids, top


# =========================================================================
# stage 2a — listwise LLM reranker (OpenRouter, RankGPT-style)
# =========================================================================
class LLMReranker:
    def __init__(self, model: str, window: int = _WINDOW, passage_words: int = 110,
                 max_cost: float = 3.0):
        if not settings.OPENROUTER_API_KEY:
            raise RuntimeError("set OPENROUTER_API_KEY in .env (a chat-capable OpenRouter key)")
        self.model = model
        self.window = window
        self.passage_words = passage_words
        self.max_cost = max_cost
        self.calls = 0
        self.cost = 0.0
        self.c = httpx.Client(base_url=settings.OPENROUTER_BASE_URL,
                              headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
                              timeout=120)

    def _permutation(self, query: str, passages: list[str]) -> list[int]:
        """Ask the model to order `passages`; return 0-based indices best-first."""
        numbered = "\n\n".join(
            f"[{i + 1}] {' '.join(p.split()[:self.passage_words])}" for i, p in enumerate(passages))
        user = (
            f"I will give you {len(passages)} passages, each with an identifier in [].\n"
            f"Rank them by relevance to the search query.\n\n{numbered}\n\n"
            f"Search Query: {query}\n\n"
            f"Rank all {len(passages)} passages from most to least relevant. Respond with ONLY "
            f"the identifiers in order, e.g. [2] > [5] > [1] > … — no prose, no explanation."
        )
        for attempt in range(4):
            try:
                r = self.c.post("/chat/completions", json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are RankGPT, an expert at ranking "
                         "passages by relevance to a query."},
                        {"role": "user", "content": user}],
                    "temperature": 0, "max_tokens": 800,
                    "reasoning": {"enabled": False},
                })
                r.raise_for_status()
                j = r.json()
                break
            except httpx.HTTPError:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        self.calls += 1
        self.cost += float((j.get("usage") or {}).get("cost", 0) or 0)
        txt = j["choices"][0]["message"]["content"] or ""
        order, seen = [], set()
        for n in re.findall(r"\[(\d+)\]", txt):
            i = int(n) - 1
            if 0 <= i < len(passages) and i not in seen:
                order.append(i)
                seen.add(i)
        order += [i for i in range(len(passages)) if i not in seen]   # keep any the model dropped
        return order

    def rerank(self, query: str, cand_ids: list[str], cand_texts: list[str]) -> list[str]:
        if self.cost > self.max_cost:
            return cand_ids                                          # budget hit — pass through
        idx = list(range(len(cand_ids)))
        w = self.window
        if len(idx) <= w:
            perm = self._permutation(query, cand_texts)
            return [cand_ids[i] for i in perm]
        # RankGPT sliding window, back to front, 50% overlap
        step = w // 2
        end = len(idx)
        while True:
            start = max(0, end - w)
            win = idx[start:end]
            perm = self._permutation(query, [cand_texts[i] for i in win])
            idx[start:end] = [win[p] for p in perm]
            if start == 0:
                break
            end -= step
        return [cand_ids[i] for i in idx]


# =========================================================================
# stage 2b — local CPU cross-encoder (transformers)
# =========================================================================
class CrossEncoder:
    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(min(16, os.cpu_count() or 4))
        console.print(f"  loading reranker [bold]{name}[/] (CPU)…")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.calls = 0
        self.cost = 0.0

    def _scores(self, query: str, passages: list[str], max_length=320, batch=64) -> list[float]:
        out: list[float] = []
        for i in range(0, len(passages), batch):
            chunk = passages[i:i + batch]
            enc = self.tok([query] * len(chunk), chunk, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt")
            with self.torch.no_grad():
                logits = self.mdl(**enc).logits
            logits = logits.squeeze(-1) if logits.shape[-1] == 1 else logits[:, -1]
            out.extend(logits.tolist())
        return out

    def rerank(self, query: str, cand_ids: list[str], cand_texts: list[str]) -> list[str]:
        self.calls += 1
        sc = self._scores(query, cand_texts)
        return [c for _, c in sorted(zip(sc, cand_ids), key=lambda t: -t[0])]


# =========================================================================
# report
# =========================================================================
def _agg(per_q: list[dict], keys) -> dict:
    return {k: round(float(np.mean([p.get(k, 0.0) for p in per_q])), 4) for k in keys}


def _print_table(suite, keys, dense, rr, base):
    console.print("\n  | metric | dense | reranked | Δ rerank | baseline |")
    console.print("  |---|---|---|---|---|")
    for k in keys:
        d, r = dense[k], rr[k]
        bv = base.get(k)
        tag = " ✅" if r - d > 0.002 else (" ⚠️" if r - d < -0.002 else "")
        bcell = f"{bv:.3f}" if isinstance(bv, (int, float)) else "—"
        console.print(f"  | {k} | {d:.3f} | [bold]{r:.3f}[/] | {r - d:+.3f}{tag} | {bcell} |")
    head = "ndcg@10" if suite.startswith("beir_") else "both@5"
    lift = rr[head] - dense[head]
    verdict = ("[green]clear win[/]" if lift > 0.02 else
               "[yellow]marginal[/]" if lift > 0.005 else "[red]no help[/]")
    console.print(f"\n  → {head} {lift:+.3f} vs dense — {verdict}")


def run(reranker: str, suites: list[str], fetch: int | None = None,
        backend: str = "openrouter", max_cost: float = 3.0) -> None:
    fetch = fetch or _FETCH_DEFAULT[backend]
    baseline = (_load_baseline() or {}).get("suites", {})

    if backend == "openrouter":
        rr = LLMReranker(reranker, max_cost=max_cost)
        console.print(f"[dim]backend: OpenRouter listwise · model {reranker} · "
                      f"top-{fetch} · budget ${max_cost}[/]")
    else:
        rr = CrossEncoder(reranker)
        console.print(f"[dim]backend: local cross-encoder · top-{fetch}[/]")

    for suite in suites:
        console.rule(f"{suite}  ·  rerank top-{fetch}")
        if suite.startswith("beir_"):
            docs, queries, gold, qrels = _beir_data(suite)
            score = lambda ranked, qid: beir._score_query(ranked, qrels[qid])   # noqa: E731
            keys = ("ndcg@10", "recall@10", "recall@100", "mrr@10")
        elif suite == "hotpotqa":
            docs, queries, gold = _hotpot_data()
            score = lambda ranked, qid: _score_hotpot(ranked, gold[qid])        # noqa: E731
            keys = ("support_recall@2", "support_recall@5", "both@2", "both@5")
        else:
            console.print(f"[yellow]skip {suite}[/] — rerankcompare does beir_* / hotpotqa")
            continue

        console.print(f"  corpus {len(docs)} · queries {len(queries)}")
        doc_ids, qids, top = _dense(suite, docs, queries, fetch)
        dense_rank = {qid: [doc_ids[j] for j in top[qi]] for qi, qid in enumerate(qids)}

        t0 = time.time()
        reranked: dict[str, list[str]] = {}
        for qi, qid in enumerate(qids):
            cand = dense_rank[qid]
            reranked[qid] = rr.rerank(queries[qid], cand, [docs[c] for c in cand])
            if backend == "openrouter" and qi % 25 == 0:
                console.print(f"  [dim]{qi + 1}/{len(qids)}  ${rr.cost:.2f}  {rr.calls} calls[/]",
                              end="\r")
            if rr.cost > max_cost:
                console.print(f"\n  [red]budget ${max_cost} hit after {qi + 1} queries — "
                              f"rest pass through unranked[/]")
                break
        console.print(" " * 60, end="\r")
        console.print(f"  reranked {len(qids)} queries in {time.time() - t0:.0f}s"
                      + (f"  ·  ${rr.cost:.2f}  ({rr.calls} calls)" if rr.cost else ""))

        dense_agg = _agg([score(dense_rank[q], q) for q in qids], keys)
        rr_agg = _agg([score(reranked[q], q) for q in qids], keys)
        _print_table(suite, keys, dense_agg, rr_agg, baseline.get(suite, {}))

    if rr.cost:
        console.print(f"\n[bold]total OpenRouter spend: ${rr.cost:.2f}[/] ({rr.calls} calls)")
