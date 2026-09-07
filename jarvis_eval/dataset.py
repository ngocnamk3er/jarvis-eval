"""Load the JSONL suites under datasets/."""
import json

from jarvis_eval.config import DATASETS

SUITES = ("retrieval", "rag_qa", "agent_tasks")


def load(suite: str) -> list[dict]:
    path = DATASETS / f"{suite}.jsonl"
    rows: list[dict] = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as e:
            raise ValueError(f"{path.name}:{i} is not valid JSON: {e}") from e
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: duplicate case ids")
    return rows


def smoke(suite: str, n: int = 3) -> list[dict]:
    return load(suite)[:n]
