"""LLM-as-judge over OpenRouter.

One chat/completions call per grade, `temperature=0`, JSON-object response
format, 2 retries on transient errors, and a disk cache keyed by
(model, rubric, prompt) so re-running `report` or a partially-failed run
doesn't re-pay.
"""
import hashlib
import json
import re
import time

import httpx

from jarvis_eval.config import settings, JUDGE_CACHE_FILE

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
_cache: dict | None = None


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(JUDGE_CACHE_FILE.read_text())
        except (FileNotFoundError, ValueError):
            _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is not None:
        JUDGE_CACHE_FILE.write_text(json.dumps(_cache, indent=2, sort_keys=True))


def _key(rubric: str, prompt: str) -> str:
    return hashlib.sha256(f"{settings.JUDGE_MODEL}\x00{rubric}\x00{prompt}".encode()).hexdigest()


def _call(system: str, user: str) -> str:
    if not settings.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set — the judge needs it.")
    body = {
        "model": settings.JUDGE_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    last: Exception | None = None
    for attempt in range(3):
        try:
            r = httpx.post(
                f"{settings.OPENROUTER_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
                json=body, timeout=90.0,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError) as e:  # noqa: PERF203
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"judge call failed after 3 tries: {last}")


def judge(rubric: str, system: str, user: str) -> dict:
    cache = _load_cache()
    k = _key(rubric, user)
    if k in cache:
        return cache[k]
    try:
        raw = _call(system, user)
    except RuntimeError as e:
        # a judge outage must not abort the whole suite — score it None,
        # surfaced in the report as an error, and don't cache
        return {"score": None, "pass": None, "reasoning": f"judge unavailable: {e}"}
    m = _JSON_BLOCK.search(raw)
    try:
        parsed = json.loads(m.group(0) if m else raw)
    except ValueError:
        parsed = {"score": None, "pass": None, "reasoning": f"unparseable judge output: {raw[:300]}"}
    cache[k] = parsed
    _save_cache()
    return parsed


_SYS = ("You are a strict, fair evaluator of an AI assistant's output. "
        "Respond with a single JSON object and nothing else.")


def score_answer(question: str, gold: str, acceptable: list[str], answer: str) -> dict:
    user = f"""Grade the assistant's ANSWER to a question against the reference.

QUESTION:
{question}

REFERENCE ANSWER:
{gold}

ALSO ACCEPTABLE POINTS (any subset is fine, not all required):
{json.dumps(acceptable, ensure_ascii=False)}

ASSISTANT ANSWER:
{answer or "(empty)"}

Score 1-5 for factual correctness & completeness vs the reference:
5 = fully correct and complete
4 = correct, minor omission
3 = partially correct or missing a key point
2 = mostly wrong or largely incomplete
1 = wrong, empty, or refused
Return JSON: {{"score": <int 1-5>, "reasoning": "<one sentence>"}}"""
    return judge("answer_correctness", _SYS, user)


def score_groundedness(answer: str, context: str) -> dict:
    user = f"""Judge whether the ASSISTANT ANSWER is grounded in the CONTEXT it was
given (retrieved file snippets). Penalise claims not supported by the context
(hallucinations); do not penalise correct omissions.

CONTEXT:
{context or "(no retrieval happened)"}

ASSISTANT ANSWER:
{answer or "(empty)"}

Score 1-5:
5 = every substantive claim is supported by the context
3 = mostly supported, one unsupported claim
1 = largely unsupported / fabricated, or no answer
Return JSON: {{"score": <int 1-5>, "reasoning": "<one sentence>"}}"""
    return judge("groundedness", _SYS, user)


def score_tool_choice(task: str, tool_sequence: list[str], final_text: str) -> dict:
    user = f"""An agent was given the TASK below. It made this sequence of tool calls:
{json.dumps(tool_sequence)}

Its final answer:
{(final_text or "(empty)")[:1500]}

TASK:
{task}

Rate how appropriate the tool usage was (right tools, not wildly too many or
too few, no obviously missing step). Score 1-5 (5 = ideal, 1 = clearly wrong
tools or thrashing). Return JSON: {{"score": <int 1-5>, "reasoning": "<one sentence>"}}"""
    return judge("tool_choice", _SYS, user)


def check_freeform(rubric: str, content: str) -> dict:
    user = f"""Check whether the CONTENT satisfies this requirement.

REQUIREMENT:
{rubric}

CONTENT:
{content[:4000] if content else "(empty)"}

Return JSON: {{"pass": <true|false>, "reasoning": "<one sentence>"}}"""
    return judge(f"freeform::{rubric[:60]}", _SYS, user)
