"""Conversation CRUD through the backend API (bearer = eval user)."""
import httpx

from jarvis_eval.clients.auth import access_token
from jarvis_eval.config import settings


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=settings.api_base,
        headers={**settings.api_headers, "Authorization": f"Bearer {access_token()}"},
        timeout=30.0,
    )


def create(title: str) -> str:
    with _client() as c:
        r = c.post("/api/v1/conversations", json={"title": title})
        r.raise_for_status()
        return r.json()["id"]


def delete(thread_id: str) -> None:
    try:
        with _client() as c:
            c.delete(f"/api/v1/conversations/{thread_id}")
    except httpx.HTTPError:
        pass  # best-effort cleanup


def messages(thread_id: str) -> list[dict]:
    with _client() as c:
        r = c.get(f"/api/v1/conversations/{thread_id}/messages")
        r.raise_for_status()
        data = r.json()
    return data["messages"] if isinstance(data, dict) else data


def sandbox_file(thread_id: str, name: str) -> tuple[int, bytes, str]:
    """GET /api/v1/chat/sandbox-file — bytes of a file the agent surfaced
    with present_file. Returns (status, content, content_type)."""
    with _client() as c:
        r = c.get("/api/v1/chat/sandbox-file", params={"thread_id": thread_id, "name": name})
    return r.status_code, r.content, r.headers.get("content-type", "")
