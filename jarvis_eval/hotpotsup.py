"""`jeval hotpotsup <model>` — Sup EM / F1 for HotpotQA, the way the
leaderboard scores it (sentence level), on the 300-question sample.

Jarvis has no supporting-fact head, so this bolts one on as an LLM call:
give a model the question, the gold answer, and a set of paragraphs with
every sentence numbered; ask which sentences are the evidence; map the reply
back to `(title, sent_id)` pairs and score with the official `update_sp`
(EM = exact set match, F1 = sentence precision/recall).

`--context` picks which paragraphs the selector sees:

  distractor   the question's own 10 paragraphs (2 gold + 8 distractors) —
               the leaderboard's input, so this number is directly
               comparable to Beam Retrieval's Sup EM ≈ 0.64 / F1 ≈ 0.90.
  dense        the top-`--topk` paragraphs from dense retrieval
               (text-embedding-3-large @1024) — Jarvis's real pipeline.
  rerank       the top-`--topk` after listwise LLM reranking (`--rerank-model`)
               of the dense top-20 — the pipeline with a reranker.
  all          run all three.

dense / rerank cap Sup recall at retrieval recall (can't cite a sentence in a
paragraph you never retrieved) — that's the point: it shows how much the
reranker's better paragraph recall carries through to supporting facts.

Offline, like embcompare / rerankcompare. Prints a live $ total and stops at
--max-cost.
"""
from __future__ import annotations

import re
import time

import numpy as np
from rich.console import Console

from jarvis_eval.benchmarks import hotpotqa
from jarvis_eval.config import settings
from jarvis_eval.rerankcompare import LLMReranker, _dense, _hotpot_data

console = Console()

_SYS = ("You identify which sentences in a set of Wikipedia paragraphs are the "
        "evidence for a given answer to a question. You reply with JSON only.")


def update_sp(pred: set, gold: set) -> tuple[float, float, float, float]:
    """Official HotpotQA supporting-fact scoring for one question.
    Returns (em, precision, recall, f1). EM = 1 only on an exact set match."""
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    em = 1.0 if fp == 0 and fn == 0 else 0.0
    return em, prec, rec, f1


class SupSelector:
    """One LLM call: question + answer + numbered paragraphs -> chosen
    (paragraph_idx, sentence_idx) pairs."""

    def __init__(self, model: str, max_cost: float = 2.0):
        if not settings.OPENROUTER_API_KEY:
            raise RuntimeError("set OPENROUTER_API_KEY in .env")
        import httpx
        self.model = model
        self.max_cost = max_cost
        self.calls = 0
        self.cost = 0.0
        self.c = httpx.Client(base_url=settings.OPENROUTER_BASE_URL,
                              headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
                              timeout=120)

    def pick(self, question: str, answer: str,
             paras: list[tuple[str, list[str]]]) -> set[tuple[int, int]]:
        blocks = []
        for pi, (title, sents) in enumerate(paras):
            lines = "\n".join(f"    [{si}] {s.strip()}" for si, s in enumerate(sents))
            blocks.append(f"[P{pi}] {title}\n{lines}")
        user = (
            f"Question: {question}\nAnswer: {answer}\n\n"
            f"Paragraphs (each sentence prefixed with its number):\n\n"
            + "\n\n".join(blocks)
            + "\n\nList every sentence that is directly needed to justify the answer — "
              "usually 2-3 sentences spread across 2 paragraphs. Respond with ONLY a JSON "
              "array of [paragraph, sentence] pairs, e.g. [[0,1],[3,0]]. Nothing else."
        )
        for attempt in range(4):
            try:
                r = self.c.post("/chat/completions", json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": _SYS},
                                 {"role": "user", "content": user}],
                    "temperature": 0, "max_tokens": 300, "reasoning": {"enabled": False},
                })
                r.raise_for_status()
                j = r.json()
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    return set()
                time.sleep(2 * (attempt + 1))
        self.calls += 1
        self.cost += float((j.get("usage") or {}).get("cost", 0) or 0)
        txt = j["choices"][0]["message"]["content"] or ""
        out: set[tuple[int, int]] = set()
        for a, b in re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", txt):
            pi, si = int(a), int(b)
            if 0 <= pi < len(paras) and 0 <= si < len(paras[pi][1]):
                out.add((pi, si))
        return out


# =========================================================================
def _para_map(rows):
    """title -> [sentences]  (union of every sampled question's paragraphs)."""
    pm: dict[str, list[str]] = {}
    for row in rows:
        ctx = row["context"]
        for t, s in zip(ctx["title"], ctx["sentences"]):
            pm.setdefault(t, list(s))
    return pm


def _contexts_for(context: str, rows, topk: int, rerank_model: str, max_cost: float):
    """qid -> ordered list of (title, [sentences]) the selector will see."""
    if context == "distractor":
        return {row["id"]: list(zip(row["context"]["title"],
                                    [list(s) for s in row["context"]["sentences"]]))
                for row in rows}

    docs, queries, _gold = _hotpot_data()                 # title -> paragraph text
    fetch = 20 if context == "rerank" else topk
    doc_ids, qids, top = _dense("hotpotqa", docs, queries, fetch)
    pm = _para_map(rows)
    order: dict[str, list[str]] = {}

    if context == "dense":
        for qi, qid in enumerate(qids):
            order[qid] = [doc_ids[j] for j in top[qi]][:topk]
    else:  # rerank
        rr = LLMReranker(rerank_model, max_cost=max_cost)
        for qi, qid in enumerate(qids):
            cand = [doc_ids[j] for j in top[qi]]
            ranked = rr.rerank(queries[qid], cand, [docs[c] for c in cand])
            order[qid] = ranked[:topk]
            if qi % 25 == 0:
                console.print(f"  [dim]rerank {qi + 1}/{len(qids)}  ${rr.cost:.2f}[/]", end="\r")
        console.print(" " * 60, end="\r")
        console.print(f"  rerank done · ${rr.cost:.2f}")

    return {qid: [(t, pm[t]) for t in titles if t in pm] for qid, titles in order.items()}


def run(model: str, contexts: list[str], topk: int = 5,
        rerank_model: str = "google/gemini-2.5-flash", max_cost: float = 2.0) -> None:
    rows = hotpotqa._sample()
    gold = {row["id"]: set(zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]))
            for row in rows}

    sel = SupSelector(model, max_cost=max_cost)
    console.print(f"[dim]selector {model} · {len(rows)} questions · gold answer shown[/]\n")

    summary = []
    for context in contexts:
        console.rule(f"context = {context}" + ("" if context == "distractor" else f"  (top-{topk})"))
        per_q = _contexts_for(context, rows, topk, rerank_model, max_cost)

        ems, f1s, precs, recs, capped = [], [], [], [], 0
        for i, row in enumerate(rows):
            qid = row["id"]
            paras = per_q.get(qid, [])
            title_at = [t for t, _ in paras]
            g = gold[qid]
            # retrieval ceiling: gold sentences whose paragraph wasn't retrieved
            if not {t for t, _ in g} <= set(title_at):
                capped += 1
            chosen = sel.pick(row["question"], row["answer"], paras) if paras else set()
            pred = {(title_at[pi], si) for pi, si in chosen}
            em, p, r, f1 = update_sp(pred, g)
            ems.append(em); precs.append(p); recs.append(r); f1s.append(f1)
            if i % 25 == 0:
                console.print(f"  [dim]{i + 1}/{len(rows)}  ${sel.cost:.2f}[/]", end="\r")
            if sel.cost > max_cost:
                console.print(f"\n  [red]budget ${max_cost} hit at {i + 1}[/]")
                break
        console.print(" " * 60, end="\r")

        row_out = dict(context=context,
                       sup_em=round(float(np.mean(ems)), 4),
                       sup_f1=round(float(np.mean(f1s)), 4),
                       sup_p=round(float(np.mean(precs)), 4),
                       sup_r=round(float(np.mean(recs)), 4),
                       gold_para_missing=capped)
        summary.append(row_out)
        console.print(f"  Sup EM {row_out['sup_em']:.3f} · F1 {row_out['sup_f1']:.3f} "
                      f"(P {row_out['sup_p']:.3f} / R {row_out['sup_r']:.3f})  "
                      f"[dim]· {capped}/{len(rows)} q missing a gold paragraph[/]")

    console.rule("summary")
    console.print("| context | Sup EM | Sup F1 | Sup P | Sup R | gold-para missing |")
    console.print("|---|---|---|---|---|---|")
    for s in summary:
        console.print(f"| {s['context']} | {s['sup_em']:.3f} | {s['sup_f1']:.3f} | "
                      f"{s['sup_p']:.3f} | {s['sup_r']:.3f} | {s['gold_para_missing']}/{len(rows)} |")
    console.print("\n[dim]reference — Beam Retrieval (leaderboard, distractor): Sup EM 0.663 / F1 0.901[/]")
    if sel.cost:
        console.print(f"[bold]OpenRouter spend: ${sel.cost:.2f}[/] ({sel.calls} calls)")
