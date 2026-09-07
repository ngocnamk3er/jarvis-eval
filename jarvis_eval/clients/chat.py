"""Run the real Jarvis agent once and capture what it did.

The entry point is `run_agent(prompt)` -> RunTrace. Used by the hotpotqa and
gaia suites (BEIR never touches this — it only does retrieval).

How the backend streams a turn:
  POST /api/v1/chat/stream  -> Server-Sent Events, one per line: `data: {json}`
  event types we care about: token/usage, tool_start, tool_end, file, error.
  When the agent wants to run `bash` it emits a `hitl_request` (human-in-the
  -loop) event and the stream ends. We then POST /api/v1/chat/resume with
  decision="approve" and read the next stream segment — up to MAX_HITL_ROUNDS
  times. Eval tasks are trusted, so we blanket-approve every bash call.

`_pump` folds one stream segment into the trace; `run_agent` loops it over
the resume rounds, then reads the conversation back for the final answer.
"""
import json
import re
import time

import httpx

from jarvis_eval.clients import conversations
from jarvis_eval.clients.auth import access_token
from jarvis_eval.config import settings
from jarvis_eval.trace import RunTrace, ToolCall

# search_files prints results as "/path/to/doc.txt (score=0.83): <snippet>"
_SEARCH_FILES_LINE = re.compile(r"^(/\S+) \(score=", re.M)
# grep_files prints "/path/to/doc.txt (12.3KB)"
_GREP_LINE = re.compile(r"^(/\S+) \(", re.M)


def _client(user: str | None = None) -> httpx.Client:
    """httpx client for jarvis-backend, authed as `user`, with the full
    RUN_TIMEOUT budget (agent turns are slow)."""
    return httpx.Client(
        base_url=settings.api_base,
        headers={**settings.api_headers, "Authorization": f"Bearer {access_token(user)}"},
        timeout=httpx.Timeout(settings.RUN_TIMEOUT, connect=15.0),
    )


def _parse_retrieval(tool_name: str, output: str) -> list[str]:
    """Pull the file paths a retrieval tool returned out of its text output —
    this is how we know what the agent actually retrieved."""
    if tool_name == "search_files":
        return _SEARCH_FILES_LINE.findall(output or "")
    if tool_name == "grep_files":
        return [m for m in _GREP_LINE.findall(output or "") if m.startswith("/")]
    return []


def _pump(resp: httpx.Response, trace: RunTrace) -> list[dict]:
    """Read one SSE stream to completion, folding events into `trace`.
    Returns the raw event list (so the caller can spot a `hitl_request`)."""
    events: list[dict] = []
    open_tools: dict[str, dict] = {}   # run_id -> tool meta, matched up at tool_end
    for raw in resp.iter_lines():
        if not raw or not raw.startswith("data: "):
            continue
        ev = json.loads(raw[6:])       # strip the "data: " prefix
        events.append(ev)
        t = ev.get("type")
        if t == "usage":                                  # token counts, one per LLM call
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
                task_run_id=meta["task_run_id"]))
            trace.retrieval_paths.extend(_parse_retrieval(meta["name"], out))
        elif t == "file":                                 # a present_file result
            f = {k: ev[k] for k in ("name", "mime", "size", "path") if k in ev}
            trace.file_outputs.append(f)
            # present_file renders as a `file` event, not a tool_end — record
            # it as a call too so tool_names shows it.
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
    """Send `prompt` to the agent, auto-approve any bash calls, return a
    RunTrace (final_text, tool_calls, retrieval_paths, tokens, $, latency).
    Never raises — transport errors land in trace.error."""
    model = model or settings.RUNNER_MODEL
    trace = RunTrace(case_id=case_id, model=model)
    started = time.time()
    try:
        tid = conversations.create(f"[eval] {case_id}", user=user)   # throwaway conversation
        trace.thread_id = tid
        body = {
            "thread_id": tid, "content": prompt, "model": model,
            "thinking_effort": settings.RUNNER_THINKING_EFFORT, "web_search": web_search,
        }
        with _client(user) as c:
            # first stream segment
            with c.stream("POST", "/api/v1/chat/stream", json=body) as resp:
                resp.raise_for_status()
                events = _pump(resp, trace)

            # each bash approval resumes the run; loop until no more hitl_request
            rounds = 0
            while any(e.get("type") == "hitl_request" for e in events) and rounds < settings.MAX_HITL_ROUNDS:
                rounds += 1
                rb = {"thread_id": tid, "decision": "approve", "model": model, "web_search": web_search}
                with c.stream("POST", "/api/v1/chat/resume", json=rb) as resp:
                    resp.raise_for_status()
                    events = _pump(resp, trace)
            trace.hitl_rounds = rounds
            if rounds >= settings.MAX_HITL_ROUNDS and any(e.get("type") == "hitl_request" for e in events):
                trace.stopped = True   # agent kept asking for bash; gave up

        # read the conversation back for the final answer (more reliable than
        # stitching token deltas from the stream)
        try:
            msgs = conversations.messages(tid, user=user)
            trace.messages = msgs
            seen = {(f.get("name"), f.get("path")) for f in trace.file_outputs}
            for m in reversed(msgs):                       # last assistant message
                if m.get("role") == "assistant":
                    texts = [p["content"] for p in m.get("parts", [])
                             if p.get("type") == "text" and p.get("content", "").strip()]
                    if texts:
                        trace.final_text = "\n".join(texts).strip()
                    for p in m.get("parts", []):           # catch file parts we missed
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
