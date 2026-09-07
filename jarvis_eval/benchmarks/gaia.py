"""GAIA (validation, level 1) — real multi-step assistant tasks, exact-match
scored with GAIA's own normalisation.

GAIA is a *gated* HF dataset: set HF_TOKEN (https://huggingface.co/settings/
tokens) and accept the terms once at
https://huggingface.co/datasets/gaia-benchmark/GAIA .

Many GAIA questions ship a file attachment (xlsx/pdf/image) the agent must
open. Jarvis has no way to ingest those, so this suite runs only the
level-1 questions with **no attachment** — expect a low score; it partly
measures "Jarvis is not a browsing agent", which is useful signal on its own.
"""
import re

from jarvis_eval.benchmarks import hf
from jarvis_eval.clients import chat
from jarvis_eval.config import settings

DS, SPLIT = "gaia-benchmark/GAIA", "validation"
_INSTRUCTION = (
    "Finish with a line 'FINAL ANSWER: <answer>'. The answer is a number "
    "OR as few words as possible OR a comma-separated list. No units unless "
    "asked, no thousands separators, no articles."
)


def _norm_str(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s.%-]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _norm_num(s: str) -> str:
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return str(float(s))
    except ValueError:
        return s


def gaia_score(pred: str, gold: str) -> float:
    pred = (pred or "").strip()
    if "," in gold:  # list
        p = [x.strip() for x in pred.split(",")]
        g = [x.strip() for x in gold.split(",")]
        if len(p) != len(g):
            return 0.0
        return 1.0 if all(_norm_str(a) == _norm_str(b) or _norm_num(a) == _norm_num(b)
                          for a, b in zip(p, g)) else 0.0
    if re.fullmatch(r"[-+]?[\d,.]+%?\$?", gold.strip()):
        return 1.0 if _norm_num(pred) == _norm_num(gold) else 0.0
    return 1.0 if _norm_str(pred) == _norm_str(gold) else 0.0


def _rows() -> list[dict]:
    if not settings.HF_TOKEN:
        raise RuntimeError("GAIA is gated — set HF_TOKEN and accept the terms on HF.")
    for cfg in ("2023_level1", "2023_all"):
        try:
            return hf.load_rows(DS, cfg, SPLIT)
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("could not load GAIA (config/split mismatch or no access)")


def _cases() -> list[dict]:
    out = []
    for r in _rows():
        if str(r.get("Level")) != "1":
            continue
        if (r.get("file_name") or "").strip():
            continue  # needs an attachment Jarvis can't open
        out.append({"id": r["task_id"], "question": r["Question"], "gold": r["Final answer"]})
    return out


def seed(_suite: str = "gaia") -> dict:
    return {"note": "GAIA needs no corpus (open web); nothing to seed.",
            "cases": len(_cases())}


def run(_cases, repeats: int = 1, suite: str = "gaia") -> list[dict]:
    results = []
    for case in _cases()[: settings.BENCH_AGENT_SAMPLE]:
        tr = chat.run_agent(f"{case['question']}\n\n({_INSTRUCTION})", case["id"],
                            web_search=True)
        from jarvis_eval.metrics import extract_final_answer
        pred = extract_final_answer(tr.final_text)
        results.append({
            "suite": "gaia", "case_id": case["id"], "repeat": 0,
            "metrics": {"score": gaia_score(pred, case["gold"]),
                        "latency_s": tr.wall_seconds, "usd": tr.usd, "turns": tr.turns},
            "meta": {"question": case["question"], "gold": case["gold"], "pred": pred[:300],
                     "tool_sequence": tr.tool_names, "stopped": tr.stopped, "error": tr.error},
        })
    return results
