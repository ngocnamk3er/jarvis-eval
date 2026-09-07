"""The file workspace — talk straight to jarvis-file-service.

Why direct (not via jarvis-backend): the backend doesn't proxy the
`/search/*` routes, and bulk-uploading thousands of docs through it is slow.
file-service is ClusterIP-only though, so you must `make port-forward` first
(it exposes it on FILE_SERVICE_URL = http://localhost:18002) and set
INTERNAL_API_KEY in `.env`.

Every call takes a Keycloak username in `user`; `user_sub()` turns that into
the `sub` that file-service expects as `user_id`.

Used by:
  * benchmark seed()  — ensure_folder + bulk_upload + wait_for_indexing
  * benchmark run()   — search_vector (this is what the agent's `search_files`
                        tool calls under the hood, so scoring it here == scoring
                        Jarvis's semantic search)
"""
import concurrent.futures as cf
import time

import httpx

from jarvis_eval.clients.auth import user_sub
from jarvis_eval.config import settings

# an indexing_status a file won't move out of — stop waiting once every file is here
_TERMINAL_STATUS = {"done", "skipped", "failed"}


def _svc() -> httpx.Client:
    """httpx client for file-service (internal API key, no OIDC)."""
    return httpx.Client(
        base_url=settings.FILE_SERVICE_URL,
        headers={"X-Internal-Api-Key": settings.INTERNAL_API_KEY},
        timeout=60.0,
    )


# =========================================================================
# listing / seeding
# =========================================================================
def list_all(path: str = "/", user: str | None = None) -> list[dict]:
    """Recursively walk the workspace tree. Each entry:
    {path, type, indexing_status, id}."""
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
    """Delete every top-level node (cascades). Called at the start of seed()
    so a re-seed is clean. Returns how many were removed."""
    uid = user_sub(user)
    with _svc() as c:
        r = c.get("/api/v1/files/tree", params={"user_id": uid, "path": "/"})
        r.raise_for_status()
        nodes = r.json()
        for n in nodes:
            c.delete(f"/api/v1/files/nodes/{n['id']}", params={"user_id": uid})
    return len(nodes)


def ensure_folder(path: str, user: str | None = None) -> None:
    """mkdir -p for the workspace, e.g. "/bench/hotpotqa"."""
    uid = user_sub(user)
    parts = [p for p in path.strip("/").split("/") if p]
    with _svc() as c:
        for i in range(len(parts)):
            parent = "/" + "/".join(parts[:i])
            r = c.post("/api/v1/files/folders",
                       json={"user_id": uid, "parent_path": parent, "name": parts[i]})
            if r.status_code >= 400 and "exist" not in r.text.lower():
                r.raise_for_status()   # tolerate "already exists", raise anything else


def bulk_upload(docs: list[tuple[str, str]], parent_path: str = "/",
                user: str | None = None, concurrency: int = 16) -> int:
    """Upload many (filename, text) docs into `parent_path` (must exist),
    `concurrency` at a time. file-service then chunks + embeds each one in a
    background task. Returns how many uploaded OK.

    Each upload retries up to 3x (file-service is a single small pod and can
    briefly 5xx under load)."""
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
                time.sleep(1 + attempt)   # linear backoff
        return False

    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        return sum(1 for ok in ex.map(one, docs) if ok)


def wait_for_indexing(timeout: float = 180.0, poll: float = 3.0,
                      user: str | None = None) -> list[dict]:
    """Block until every file has finished embedding (indexing_status is
    terminal). Raises TimeoutError with a count if it takes too long.
    Returns the final file list."""
    deadline = time.time() + timeout
    while True:
        entries = [e for e in list_all(user=user) if e["type"] == "file"]
        pending = [e for e in entries if e["indexing_status"] not in _TERMINAL_STATUS]
        if not pending:
            return entries
        if time.time() > deadline:
            raise TimeoutError(f"{len(pending)}/{len(entries)} still indexing after {timeout:.0f}s")
        time.sleep(poll)


# =========================================================================
# search  (what the agent's search_files / grep_files tools call)
# =========================================================================
def search_vector(query: str, top_k: int = 10, user: str | None = None) -> list[dict]:
    """Semantic search. Returns [{file_id, path, chunk_text, score}] best-first."""
    with _svc() as c:
        r = c.post("/api/v1/files/search/vector",
                   json={"user_id": user_sub(user), "query": query, "top_k": top_k})
        r.raise_for_status()
        return r.json()


def search_grep(query: str, path: str = "/", user: str | None = None) -> list[dict]:
    """Keyword (substring) search over file names + extracted text."""
    with _svc() as c:
        r = c.get("/api/v1/files/search/grep",
                  params={"user_id": user_sub(user), "query": query, "path": path})
        r.raise_for_status()
        return r.json()
