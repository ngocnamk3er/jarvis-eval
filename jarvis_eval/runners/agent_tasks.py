"""Agent-task suite — full agent run, scored by a per-case `check` plus a
judge on tool usage.

Case shape (datasets/agent_tasks.jsonl):
  {"id","prompt","category","gold_hint","check": {...}, "web_search": false}

check kinds:
  {"type":"final_contains","all":["3.14"],"any":[...]}  deterministic substring asserts
  {"type":"present_file","name_matches":"\\.docx$"}      a `file` event whose name matches
  {"type":"sandbox_file","name":"out.csv","rubric":"has 10 data rows"}
                                                        fetch the bytes, judge them
  {"type":"judge","rubric":"..."}                        judge final_text + tool trace
"""
import re

from jarvis_eval.clients import chat, conversations
from jarvis_eval import judge


def _check(case: dict, trace) -> tuple[float, str]:
    chk = case["check"]
    kind = chk["type"]

    if kind == "final_contains":
        text = trace.final_text or ""
        miss = [s for s in chk.get("all", []) if s.lower() not in text.lower()]
        any_ok = (not chk.get("any")) or any(s.lower() in text.lower() for s in chk["any"])
        ok = not miss and any_ok
        return (1.0 if ok else 0.0), ("" if ok else f"missing {miss or chk.get('any')}")

    if kind == "present_file":
        pat = re.compile(chk["name_matches"])
        hit = next((f for f in trace.file_outputs if pat.search(f.get("name", ""))), None)
        return (1.0 if hit else 0.0), ("" if hit else f"no file matching /{chk['name_matches']}/")

    if kind == "sandbox_file":
        want = chk["name"]
        f = next((f for f in trace.file_outputs if f.get("name") == want or want in f.get("path", "")), None)
        if not f:
            return 0.0, f"agent never present_file'd {want}"
        status, content, ctype = conversations.sandbox_file(trace.thread_id, f["path"])
        if status != 200:
            return 0.0, f"download {want} -> HTTP {status}"
        preview = content[:3000].decode("utf-8", "replace")
        v = judge.check_freeform(chk["rubric"], f"file {want} ({ctype}, {len(content)} bytes):\n{preview}")
        return (1.0 if v.get("pass") else 0.0), v.get("reasoning", "")

    if kind == "judge":
        blob = f"FINAL ANSWER:\n{trace.final_text}\n\nTOOLS CALLED: {trace.tool_names}"
        v = judge.check_freeform(chk["rubric"], blob)
        return (1.0 if v.get("pass") else 0.0), v.get("reasoning", "")

    return 0.0, f"unknown check type {kind!r}"


def run(cases: list[dict], repeats: int = 1) -> list[dict]:
    results: list[dict] = []
    for case in cases:
        for rep in range(repeats):
            trace = chat.run_agent(
                case["prompt"], case["id"], web_search=case.get("web_search", False))
            success, why = _check(case, trace)
            tc = judge.score_tool_choice(case["prompt"], trace.tool_names, trace.final_text)

            results.append({
                "suite": "agent_tasks",
                "case_id": case["id"],
                "repeat": rep,
                "metrics": {
                    "success": success,
                    "tool_choice": tc.get("score"),
                    "latency_s": trace.wall_seconds,
                    "usd": trace.usd,
                    "turns": trace.turns,
                    "hitl_rounds": trace.hitl_rounds,
                },
                "meta": {
                    "category": case.get("category"),
                    "prompt": case["prompt"],
                    "check": case["check"],
                    "check_detail": why,
                    "final_text": (trace.final_text or "")[:1500],
                    "tool_sequence": trace.tool_names,
                    "file_outputs": trace.file_outputs,
                    "tool_choice_reasoning": tc.get("reasoning"),
                    "stopped": trace.stopped,
                    "error": trace.error,
                },
            })
    return results
