"""Conversation CRUD through the backend API (bearer = the given eval user)."""
import httpx

from jarvis_eval.clients.auth import access_token
from jarvis_eval.config import settings


def _client(user: str | None = None) -> httpx.Client:
    return httpx.Client(
        base_url=settings.api_base,
        headers={**settings.api_headers, "Authorization": f"Bearer {access_token(user)}"},
        timeout=30.0,
    )


def create(title: str, user: str | None = None) -> str:
    with _client(user) as c:
        r = c.post("/api/v1/conversations", json={"title": title})
        r.raise_for_status()
        return r.json()["id"]


def delete(thread_id: str, user: str | None = None) -> None:
    try:
        with _client(user) as c:
            c.delete(f"/api/v1/conversations/{thread_id}")
    except httpx.HTTPError:
        pass


def messages(thread_id: str, user: str | None = None) -> list[dict]:
    with _client(user) as c:
        r = c.get(f"/api/v1/conversations/{thread_id}/messages")
        r.raise_for_status()
        data = r.json()
    return data["messages"] if isinstance(data, dict) else data


def sandbox_file(thread_id: str, name: str, user: str | None = None) -> tuple[int, bytes, str]:
    with _client(user) as c:
        r = c.get("/api/v1/chat/sandbox-file", params={"thread_id": thread_id, "name": name})
    return r.status_code, r.content, r.headers.get("content-type", "")
