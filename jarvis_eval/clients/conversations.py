"""Conversation endpoints on jarvis-backend, used by the agent suites
(hotpotqa / gaia). Bearer token = the given eval user; default is `eval`.

Each agent run creates a throwaway conversation, streams into it (see
chat.py), then reads the messages back to get the final answer.
"""
import httpx

from jarvis_eval.clients.auth import access_token
from jarvis_eval.config import settings


def _client(user: str | None = None) -> httpx.Client:
    """httpx client aimed at jarvis-backend, authenticated as `user`."""
    return httpx.Client(
        base_url=settings.api_base,
        headers={**settings.api_headers, "Authorization": f"Bearer {access_token(user)}"},
        timeout=30.0,
    )


def create(title: str, user: str | None = None) -> str:
    """Make a new conversation, return its id (== the thread_id)."""
    with _client(user) as c:
        r = c.post("/api/v1/conversations", json={"title": title})
        r.raise_for_status()
        return r.json()["id"]


def delete(thread_id: str, user: str | None = None) -> None:
    """Best-effort cleanup — never raise."""
    try:
        with _client(user) as c:
            c.delete(f"/api/v1/conversations/{thread_id}")
    except httpx.HTTPError:
        pass


def messages(thread_id: str, user: str | None = None) -> list[dict]:
    """The serialized message list — each message is {role, parts: [...]}.
    We read `final_text` and any `file` parts out of this."""
    with _client(user) as c:
        r = c.get(f"/api/v1/conversations/{thread_id}/messages")
        r.raise_for_status()
        data = r.json()
    return data["messages"] if isinstance(data, dict) else data


def sandbox_file(thread_id: str, name: str, user: str | None = None) -> tuple[int, bytes, str]:
    """Download a file the agent produced in its bash sandbox (via present_file).
    Returns (http_status, bytes, content_type)."""
    with _client(user) as c:
        r = c.get("/api/v1/chat/sandbox-file", params={"thread_id": thread_id, "name": name})
    return r.status_code, r.content, r.headers.get("content-type", "")
