"""Scoring maths, shared by the benchmark runners.

Two groups:
  * IR ranking metrics (recall@k, MRR, nDCG) — for retrieval.
  * QA answer scoring (Exact Match, F1) with the official SQuAD/HotpotQA
    text normalisation — for HotpotQA / GAIA answers.
Plus `mean` / `pct` helpers used by the report aggregation.

Throughout, a "ranked" list is document ids best-first, and "gold" is the
set of ids that are actually relevant to the query. Relevance is binary.
"""
from __future__ import annotations

import math
import re
import statistics
import string


# =========================================================================
# IR ranking metrics
# =========================================================================
def _first_hit_rank(ranked: list[str], gold: set[str]) -> int | None:
    """1-based position of the first relevant doc, or None if none are found."""
    for i, p in enumerate(ranked, start=1):
        if p in gold:
            return i
    return None


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """Fraction of the gold docs that appear in the top k."""
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """Fraction of the top k that are relevant."""
    if k == 0:
        return 0.0
    return len(set(ranked[:k]) & gold) / k


def mrr(ranked: list[str], gold: set[str]) -> float:
    """Mean Reciprocal Rank for one query = 1 / (rank of first relevant hit).
    1.0 if the first result is relevant, 0.5 if the second, ... 0 if none."""
    r = _first_hit_rank(ranked, gold)
    return 1.0 / r if r else 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain @ k, binary relevance.
    DCG rewards relevant docs more the higher they rank (1/log2(rank+1));
    dividing by the ideal DCG (all gold docs packed at the top) normalises
    it to 0..1."""
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(ranked[:k], start=1) if p in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


# =========================================================================
# QA answer scoring — the SQuAD / HotpotQA normalisation (official)
# =========================================================================
def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, drop the articles a/an/the, squeeze
    whitespace — so "The White House." matches "white house"."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def answer_em(pred: str, gold: str) -> float:
    """Exact Match: 1.0 if the normalised strings are identical, else 0."""
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def answer_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 between prediction and gold (bag of words).
    Partial credit — "Chief Protocol Officer" vs "Chief of Protocol" scores
    ~0.8 rather than 0."""
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pt or not gt:
        return float(pt == gt)          # both empty -> 1.0, one empty -> 0.0
    # count tokens shared, without exceeding either side's multiplicity
    common: dict[str, int] = {}
    for w in pt:
        if w in gt:
            common[w] = common.get(w, 0) + 1
    num_same = sum(min(c, gt.count(w)) for w, c in common.items())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return 2 * precision * recall / (precision + recall)


def extract_final_answer(text: str, *, strict: bool = False) -> str:
    """Pull the bare answer out of the agent's prose. Prefers an explicit
    'FINAL ANSWER: x' / 'Answer: x' line, which the prompt asks for.

    `strict` controls what happens when that line is absent. The default
    falls back to the last non-empty line, which is worth keeping where the
    metric gives partial credit — a stray sentence containing the answer
    still earns some F1.

    Under exact match it earns nothing, and it actively lies: an agent that
    ran out of budget mid-thought gets its last thought scored as if it were
    an answer, so "never finished" is indistinguishable from "answered
    wrongly". Measured over the 2026-09-23 GAIA run the fallback fired 29
    times and was right 0 of them, while hiding 6 cases that had produced no
    answer at all. strict=True returns "" instead, making that visible.
    """
    if not text:
        return ""
    # `**` around the label: models format the line as **FINAL ANSWER**: x
    # often enough that not allowing it silently loses correct answers to the
    # fallback below, which then returns the whole bolded line.
    #
    # And the *last* match, not the first: agents reason out loud, so "the
    # answer: maybe Paris?" appears mid-thought and the real line comes at the
    # end. Taking the first match scores the guess the agent then discarded.
    hits = re.findall(r"(?:\*\*)?(?:final answer|answer)(?:\*\*)?\s*[:\-]\s*(.+)", text, re.I)
    if hits:
        return hits[-1].strip().strip("*`.")
    if strict:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1].strip("*`.") if lines else text.strip()


# =========================================================================
# aggregate helpers (report.py)
# =========================================================================
def mean(xs) -> float:
    """Average, ignoring None (e.g. a metric a suite doesn't produce). 0 if empty."""
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else 0.0


def pct(xs, p: float) -> float:
    """The p-th percentile (nearest-rank), e.g. pct(latencies, 95) = p95."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return 0.0
    idx = min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))
    return round(xs[idx], 3)
