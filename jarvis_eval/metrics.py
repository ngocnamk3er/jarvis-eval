"""Standard IR ranking primitives + SQuAD/HotpotQA answer scoring + small
aggregate helpers. Relevance is binary and graded by document id."""
from __future__ import annotations

import math
import re
import statistics
import string


def _first_hit_rank(ranked: list[str], gold: set[str]) -> int | None:
    for i, p in enumerate(ranked, start=1):
        if p in gold:
            return i
    return None


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(ranked[:k])
    return len(top & gold) / len(gold)


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(ranked[:k]) & gold) / k


def mrr(ranked: list[str], gold: set[str]) -> float:
    r = _first_hit_rank(ranked, gold)
    return 1.0 / r if r else 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(ranked[:k], start=1) if p in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


# --------------------------------------------------------------------------
# QA answer scoring — the SQuAD / HotpotQA normalisation (official)
# --------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def answer_em(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def answer_f1(pred: str, gold: str) -> float:
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pt or not gt:
        return float(pt == gt)
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


def extract_final_answer(text: str) -> str:
    """Pull the answer out of an agent's prose. Honours an explicit
    'FINAL ANSWER: x' / 'Answer: x' line, else uses the last non-empty line."""
    if not text:
        return ""
    m = re.search(r"(?:final answer|answer)\s*[:\-]\s*(.+)", text, re.I)
    if m:
        return m.group(1).strip().strip("*`.")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1].strip("*`.") if lines else text.strip()


def mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else 0.0


def pct(xs, p: float) -> float:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return 0.0
    idx = min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))
    return round(xs[idx], 3)
