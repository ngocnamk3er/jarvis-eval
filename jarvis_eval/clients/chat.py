"""Drive one agent turn end-to-end through the deployed backend and collect a
RunTrace.

`/api/v1/chat/stream` is SSE (`data: {json}\\n\\n`). A `bash` call pauses the
run with a `hitl_request` event and ends the stream; we POST
`/api/v1/chat/resume` with `decision=approve` and keep reading, up to
`MAX_HITL_ROUNDS` times. Everything the agent might have wanted approved is a
`bash` call, and eval tasks are trusted, so blanket-approve.
"""
import json
import re
import time

import httpx

from jarvis_eval.clients import conversations
from jarvis_eval.clients.auth import access_token
from jarvis_eval.config import settings
from jarvis_eval.trace import RunTrace, ToolCall

_SEARCH_FILES_LINE = re.compile(r"^(/\S+) \(score=", re.M)
_GREP_LINE = re.compile(r"^(/\S+) \(", re.M)


def _client(user: str | None = None) -> httpx.Client:
    return httpx.Client(
        base_url=settings.api_base,
        headers={**settings.api_headers, "Authorization": f"Bearer {access_token(user)}"},
        timeout=httpx.Timeout(settings.RUN_TIMEOUT, connect=15.0),
    )


def _parse_retrieval(tool_name: str, output: str) -> list[str]:
    if tool_name == "search_files":
        return _SEARCH_FILES_LINE.findall(output or "")
    if tool_name == "grep_files":
        return [m for m in _GREP_LINE.findall(output or "") if m.startswith("/")]
    return []


def _pump(resp: httpx.Response, trace: RunTrace) -> list[dict]:
    """Consume one SSE stream into `trace`; return the raw event list."""
    events: list[dict] = []
    open_tools: dict[str, dict] = {}  # run_id -> {name, input, task_run_id}
    for raw in resp.iter_lines():
        if not raw or not raw.startswith("data: "):
            continue
        ev = json.loads(raw[6:])
        events.append(ev)
        t = ev.get("type")
        if t == "usage":
            trace.input_tokens += ev.get("input_tokens", 0) or 0
            trace.output_tokens += ev.get("output_tokens", 0) or 0
        elif t == "tool_start":
            open_tools[ev.get("run_id", "")] = {
                "name": ev["name"], "input": ev.get("input") or {},
                "task_run_id": ev.get("task_run_id"),
            }
        elif t == "tool_end":
            meta = open_tools.pop(ev.get("run_id", ""), None) or {
                "name": ev.get("name", "?"), "input": {}, "task_run_id": ev.get("task_run_id")}
            out = ev.get("output", "") or ""
            trace.tool_calls.append(ToolCall(
                name=meta["name"], input=meta["input"], output=out[:4000],
                task_run_id=meta["task_run_id"],
            ))
            trace.retrieval_paths.extend(_parse_retrieval(meta["name"], out))
        elif t == "file":
            f = {k: ev[k] for k in ("name", "mime", "size", "path") if k in ev}
            trace.file_outputs.append(f)
            # present_file renders as a `file` event (not a tool_end), so
            # record it in the call list too — the tool-choice judge needs it.
            trace.tool_calls.append(ToolCall(
                name="present_file", input={"path": f.get("path", "")},
                output=f"delivered {f.get('name', '')}", task_run_id=ev.get("task_run_id")))
        elif t == "stopped":
            trace.stopped = True
        elif t == "error":
            trace.error = ev.get("message", "stream error")
    return events


def run_agent(prompt: str, case_id: str, *, web_search: bool = True,
              model: str | None = None, user: str | None = None) -> RunTrace:
    model = model or settings.RUNNER_MODEL
    trace = RunTrace(case_id=case_id, model=model)
    started = time.time()
    try:
        tid = conversations.create(f"[eval] {case_id}", user=user)
        trace.thread_id = tid
        body = {
            "thread_id": tid, "content": prompt, "model": model,
            "thinking_effort": settings.RUNNER_THINKING_EFFORT, "web_search": web_search,
        }
        with _client(user) as c:
            with c.stream("POST", "/api/v1/chat/stream", json=body) as resp:
                resp.raise_for_status()
                events = _pump(resp, trace)

            rounds = 0
            while any(e.get("type") == "hitl_request" for e in events) and rounds < settings.MAX_HITL_ROUNDS:
                rounds += 1
                rb = {"thread_id": tid, "decision": "approve", "model": model, "web_search": web_search}
                with c.stream("POST", "/api/v1/chat/resume", json=rb) as resp:
                    resp.raise_for_status()
                    events = _pump(resp, trace)
            trace.hitl_rounds = rounds
            if rounds >= settings.MAX_HITL_ROUNDS and any(e.get("type") == "hitl_request" for e in events):
                trace.stopped = True

        # final assistant text: last assistant message's text parts (more
        # reliable than concatenating token deltas)
        try:
            msgs = conversations.messages(tid, user=user)
            trace.messages = msgs
            seen = {(f.get("name"), f.get("path")) for f in trace.file_outputs}
            for m in reversed(msgs):
                if m.get("role") == "assistant":
                    texts = [p["content"] for p in m.get("parts", [])
                             if p.get("type") == "text" and p.get("content", "").strip()]
                    if texts:
                        trace.final_text = "\n".join(texts).strip()
                    for p in m.get("parts", []):
                        if p.get("type") == "file" and (p.get("name"), p.get("path")) not in seen:
                            trace.file_outputs.append(
                                {k: p[k] for k in ("name", "mime", "size", "path") if k in p})
                    break
        except httpx.HTTPError:
            pass
    except httpx.HTTPError as e:
        trace.error = f"{type(e).__name__}: {e}"
    finally:
        trace.wall_seconds = round(time.time() - started, 2)
    return trace
