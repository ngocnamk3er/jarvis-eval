"""GAIA (validation, level 1) — real multi-step assistant tasks.

Unlike the other suites there's no corpus: GAIA questions are answered from
the open web (the agent uses web_search / web_fetch). Scored by exact match
after GAIA's normalisation (numbers ignore commas/units, lists compare
element-wise, strings ignore case/articles/punctuation).

Two caveats, both baked in here:
  * GAIA is a *gated* HF dataset — set HF_TOKEN in .env and accept the terms
    once at https://huggingface.co/datasets/gaia-benchmark/GAIA .
  * Many GAIA questions attach a file (xlsx/pdf/image) the agent must open.
    Jarvis can't ingest those, so `_sample()` keeps only level-1 questions
    with `file_name == ""`. Expect a low score — it partly measures "Jarvis
    is not a full browsing/file agent", which is itself useful signal.
"""
import re

from jarvis_eval.benchmarks import hf
from jarvis_eval.clients import chat
from jarvis_eval.config import settings
from jarvis_eval.metrics import extract_final_answer

DS, SPLIT = "gaia-benchmark/GAIA", "validation"
_INSTRUCTION = (
    "Finish with a line 'FINAL ANSWER: <answer>'. The answer is a number "
    "OR as few words as possible OR a comma-separated list. No units unless "
    "asked, no thousands separators, no articles."
)


# --- GAIA's official scoring normalisation ------------------------------
def _norm_str(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s.%-]", "", s)          # drop punctuation (keep . % -)
    s = re.sub(r"\b(a|an|the)\b", " ", s)     # drop articles
    return " ".join(s.split())


def _norm_num(s: str) -> str:
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return str(float(s))                 # "1,000" and "1000.0" compare equal
    except ValueError:
        return s


def gaia_score(pred: str, gold: str) -> float:
    """1.0 if `pred` matches `gold` under GAIA's rules, else 0.0."""
    pred = (pred or "").strip()
    if "," in gold:                                   # comma-separated list
        p = [x.strip() for x in pred.split(",")]
        g = [x.strip() for x in gold.split(",")]
        if len(p) != len(g):
            return 0.0
        return 1.0 if all(_norm_str(a) == _norm_str(b) or _norm_num(a) == _norm_num(b)
                          for a, b in zip(p, g)) else 0.0
    if re.fullmatch(r"[-+]?[\d,.]+%?\$?", gold.strip()):   # numeric
        return 1.0 if _norm_num(pred) == _norm_num(gold) else 0.0
    return 1.0 if _norm_str(pred) == _norm_str(gold) else 0.0   # string


def _rows() -> list[dict]:
    if not settings.HF_TOKEN:
        raise RuntimeError("GAIA is gated — set HF_TOKEN and accept the terms on HF.")
    for cfg in ("2023_level1", "2023_all"):           # config name varies by dataset version
        try:
            return hf.load_rows(DS, cfg, SPLIT)
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("could not load GAIA (config/split mismatch or no access)")


def _sample() -> list[dict]:
    """Level-1 questions with no file attachment.

    Named _sample() to match hotpotqa's: run() below takes an unused `_cases`
    placeholder as part of the shared calling convention (the CLI invokes
    every benchmark as run(None, repeats=1)), so a module function of that
    same name is shadowed by the parameter and calls None().
    """
    out = []
    for r in _rows():
        if str(r.get("Level")) != "1":
            continue
        if (r.get("file_name") or "").strip():
            continue
        out.append({"id": r["task_id"], "question": r["Question"], "gold": r["Final answer"]})
    return out


def seed(_suite: str = "gaia") -> dict:
    """Nothing to seed — GAIA answers from the open web."""
    return {"note": "GAIA needs no corpus (open web); nothing to seed.",
            "cases": len(_sample())}


def run(_cases, repeats: int = 1, suite: str = "gaia") -> list[dict]:
    results = []
    for case in _sample()[: settings.BENCH_AGENT_SAMPLE]:
        tr = chat.run_agent(f"{case['question']}\n\n({_INSTRUCTION})", case["id"],
                            web_search=True)             # GAIA needs the web
        pred = extract_final_answer(tr.final_text, strict=True)
        results.append({
            "suite": "gaia", "case_id": case["id"], "repeat": 0,
            "metrics": {"score": gaia_score(pred, case["gold"]),
                        "latency_s": tr.wall_seconds, "usd": tr.usd, "turns": tr.turns},
            "meta": {"question": case["question"], "gold": case["gold"], "pred": pred[:300],
                     "tool_sequence": tr.tool_names, "stopped": tr.stopped, "error": tr.error},
        })
    return results
