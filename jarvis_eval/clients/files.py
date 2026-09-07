"""File workspace access for the eval harness.

Seeding + listing go through the backend proxy (`/api/v1/files/*`, bearer =
eval user). Semantic / keyword search go straight to file-service
(`FILE_SERVICE_URL`, `X-Internal-Api-Key`) because the backend does not
proxy `/search/*` — run `make port-forward` first.
"""
import time
from pathlib import Path

import httpx

from jarvis_eval.clients.auth import access_token, eval_user_sub
from jarvis_eval.config import settings

_TERMINAL_STATUS = {"done", "skipped", "failed"}


def _proxy() -> httpx.Client:
    return httpx.Client(
        base_url=settings.api_base,
        headers={**settings.api_headers, "Authorization": f"Bearer {access_token()}"},
        timeout=60.0,
    )


def _file_svc() -> httpx.Client:
    return httpx.Client(
        base_url=settings.FILE_SERVICE_URL,
        headers={"X-Internal-Api-Key": settings.INTERNAL_API_KEY},
        timeout=30.0,
    )


# --------------------------------------------------------------------------
# listing / seeding (backend proxy)
# --------------------------------------------------------------------------
def list_all(path: str = "/") -> list[dict]:
    """Recursive walk. Each entry: {path, type, indexing_status, id}."""
    out: list[dict] = []
    with _proxy() as c:
        def walk(p: str):
            r = c.get("/api/v1/files/tree", params={"path": p})
            r.raise_for_status()
            for n in r.json():
                full = (p.rstrip("/") + "/" + n["name"])
                out.append({"path": full, "type": n["type"],
                            "indexing_status": n.get("indexing_status"), "id": n["id"]})
                if n["type"] == "folder":
                    walk(full)
        walk(path)
    return out


def wipe() -> int:
    """Delete every top-level node in the eval user's workspace (cascades)."""
    with _proxy() as c:
        r = c.get("/api/v1/files/tree", params={"path": "/"})
        r.raise_for_status()
        nodes = r.json()
        for n in nodes:
            c.delete(f"/api/v1/files/nodes/{n['id']}")
    return len(nodes)


def seed_corpus(corpus_dir: Path | None = None) -> dict:
    from jarvis_eval.config import CORPUS_DIR
    corpus_dir = corpus_dir or CORPUS_DIR

    files = sorted(p for p in corpus_dir.rglob("*") if p.is_file())
    made_dirs: set[str] = set()
    uploaded = 0
    with _proxy() as c:
        for f in files:
            rel = f.relative_to(corpus_dir)
            # create each ancestor folder top-down (idempotent-ish; 4xx = exists)
            parts = rel.parts[:-1]
            for i in range(len(parts)):
                folder_path = "/" + "/".join(parts[: i + 1])
                if folder_path in made_dirs:
                    continue
                parent = "/" + "/".join(parts[:i])
                resp = c.post("/api/v1/files/folders",
                              json={"parent_path": parent, "name": parts[i]})
                if resp.status_code >= 400 and "exist" not in resp.text.lower():
                    resp.raise_for_status()
                made_dirs.add(folder_path)

            parent_path = "/" + "/".join(parts) if parts else "/"
            resp = c.post("/api/v1/files/upload",
                          data={"parent_path": parent_path},
                          files={"file": (f.name, f.read_bytes(), "text/plain")})
            resp.raise_for_status()
            uploaded += 1
    return {"folders": len(made_dirs), "files": uploaded}


def wait_for_indexing(timeout: float = 180.0, poll: float = 3.0) -> list[dict]:
    """Block until every file reaches a terminal indexing_status. Returns the
    final file list; raises on timeout."""
    deadline = time.time() + timeout
    while True:
        entries = [e for e in list_all() if e["type"] == "file"]
        pending = [e for e in entries if e["indexing_status"] not in _TERMINAL_STATUS]
        if not pending:
            return entries
        if time.time() > deadline:
            raise TimeoutError(
                f"{len(pending)} file(s) still indexing after {timeout}s: "
                + ", ".join(e["path"] for e in pending[:5])
            )
        time.sleep(poll)


# --------------------------------------------------------------------------
# search (file-service direct)
# --------------------------------------------------------------------------
def search_vector(query: str, top_k: int = 10) -> list[dict]:
    """[{file_id, path, chunk_text, score}] ordered best-first."""
    with _file_svc() as c:
        r = c.post("/api/v1/files/search/vector",
                   json={"user_id": eval_user_sub(), "query": query, "top_k": top_k})
        r.raise_for_status()
        return r.json()


def search_grep(query: str, path: str = "/") -> list[dict]:
    with _file_svc() as c:
        r = c.get("/api/v1/files/search/grep",
                  params={"user_id": eval_user_sub(), "query": query, "path": path})
        r.raise_for_status()
        return r.json()
