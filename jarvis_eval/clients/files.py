"""File workspace access for the eval harness — all straight to file-service
(`FILE_SERVICE_URL` + `X-Internal-Api-Key`), so run `make port-forward` first.

`user` is a Keycloak username; we resolve it to its `sub` (the file-service
`user_id`, and the same id the agent runs under) via a token exchange.
"""
import concurrent.futures as cf
import time

import httpx

from jarvis_eval.clients.auth import user_sub
from jarvis_eval.config import settings

_TERMINAL_STATUS = {"done", "skipped", "failed"}


def _svc() -> httpx.Client:
    return httpx.Client(
        base_url=settings.FILE_SERVICE_URL,
        headers={"X-Internal-Api-Key": settings.INTERNAL_API_KEY},
        timeout=60.0,
    )


# --------------------------------------------------------------------------
# listing / seeding
# --------------------------------------------------------------------------
def list_all(path: str = "/", user: str | None = None) -> list[dict]:
    uid = user_sub(user)
    out: list[dict] = []
    with _svc() as c:
        def walk(p: str):
            r = c.get("/api/v1/files/tree", params={"user_id": uid, "path": p})
            r.raise_for_status()
            for n in r.json():
                full = (p.rstrip("/") + "/" + n["name"])
                out.append({"path": full, "type": n["type"],
                            "indexing_status": n.get("indexing_status"), "id": n["id"]})
                if n["type"] == "folder":
                    walk(full)
        walk(path)
    return out


def wipe(user: str | None = None) -> int:
    uid = user_sub(user)
    with _svc() as c:
        r = c.get("/api/v1/files/tree", params={"user_id": uid, "path": "/"})
        r.raise_for_status()
        nodes = r.json()
        for n in nodes:
            c.delete(f"/api/v1/files/nodes/{n['id']}", params={"user_id": uid})
    return len(nodes)


def ensure_folder(path: str, user: str | None = None) -> None:
    uid = user_sub(user)
    parts = [p for p in path.strip("/").split("/") if p]
    with _svc() as c:
        for i in range(len(parts)):
            parent = "/" + "/".join(parts[:i])
            r = c.post("/api/v1/files/folders",
                       json={"user_id": uid, "parent_path": parent, "name": parts[i]})
            if r.status_code >= 400 and "exist" not in r.text.lower():
                r.raise_for_status()


def bulk_upload(docs: list[tuple[str, str]], parent_path: str = "/",
                user: str | None = None, concurrency: int = 16) -> int:
    """Upload many (filename, text) docs concurrently. Folder must exist."""
    uid = user_sub(user)

    def one(item):
        name, text = item
        for attempt in range(3):
            try:
                with _svc() as c:
                    r = c.post("/api/v1/files/files",
                               data={"user_id": uid, "parent_path": parent_path},
                               files={"file": (name, text.encode("utf-8"), "text/plain")})
                    r.raise_for_status()
                return True
            except httpx.HTTPError:
                time.sleep(1 + attempt)
        return False

    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        return sum(1 for ok in ex.map(one, docs) if ok)


def wait_for_indexing(timeout: float = 180.0, poll: float = 3.0,
                      user: str | None = None) -> list[dict]:
    deadline = time.time() + timeout
    while True:
        entries = [e for e in list_all(user=user) if e["type"] == "file"]
        pending = [e for e in entries if e["indexing_status"] not in _TERMINAL_STATUS]
        if not pending:
            return entries
        if time.time() > deadline:
            raise TimeoutError(f"{len(pending)}/{len(entries)} still indexing after {timeout:.0f}s")
        time.sleep(poll)


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------
def search_vector(query: str, top_k: int = 10, user: str | None = None) -> list[dict]:
    with _svc() as c:
        r = c.post("/api/v1/files/search/vector",
                   json={"user_id": user_sub(user), "query": query, "top_k": top_k})
        r.raise_for_status()
        return r.json()


def search_grep(query: str, path: str = "/", user: str | None = None) -> list[dict]:
    with _svc() as c:
        r = c.get("/api/v1/files/search/grep",
                  params={"user_id": user_sub(user), "query": query, "path": path})
        r.raise_for_status()
        return r.json()
