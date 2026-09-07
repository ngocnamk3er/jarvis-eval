"""Download a HuggingFace dataset split as a list of dicts.

Deliberately does NOT use the `datasets` library (heavy, pulls arrow/pandas
+ its own cache). Instead it hits HF's parquet REST API, downloads the
parquet file(s), reads them with pyarrow, and caches the rows as JSONL under
`~/.cache/jarvis-eval/hf/`.

  load_rows("BeIR/scifact", "corpus", "queries")
   dataset ─┘               config ─┘  split ─┘

`_headers()` adds the HF token only when set — the 3 public benchmarks work
without one; GAIA is gated and needs it.
"""
import io
import json
import os
from pathlib import Path

import httpx
import pyarrow.parquet as pq

from jarvis_eval.config import settings

CACHE = Path(os.environ.get("JARVIS_EVAL_HF_CACHE",
                            Path.home() / ".cache" / "jarvis-eval" / "hf"))


def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.HF_TOKEN}"} if settings.HF_TOKEN else {}


def parquet_urls(dataset: str, config: str, split: str) -> list[str]:
    """Ask HF where a split's parquet file(s) live.
    GET /api/datasets/<ds>/parquet -> {config: {split: [url, ...]}}."""
    r = httpx.get(f"https://huggingface.co/api/datasets/{dataset}/parquet",
                  headers=_headers(), timeout=30, follow_redirects=True)
    r.raise_for_status()
    tree = r.json()
    try:
        return tree[config][split]
    except KeyError as e:
        raise KeyError(
            f"{dataset}: no {config!r}/{split!r} — available: "
            f"{ {c: list(s) for c, s in tree.items()} }") from e


def load_rows(dataset: str, config: str, split: str, limit: int | None = None) -> list[dict]:
    """Every row of the split as a list of dicts (one per parquet record).
    Cached to disk after the first call."""
    CACHE.mkdir(parents=True, exist_ok=True)
    slug = f"{dataset}__{config}__{split}".replace("/", "_")
    cached = CACHE / f"{slug}.jsonl"
    if cached.exists():
        rows = [json.loads(x) for x in cached.read_text().splitlines()]
        return rows[:limit] if limit else rows

    rows: list[dict] = []
    for url in parquet_urls(dataset, config, split):
        resp = httpx.get(url, headers=_headers(), timeout=120, follow_redirects=True)
        resp.raise_for_status()
        table = pq.read_table(io.BytesIO(resp.content))   # parquet bytes -> arrow table
        rows.extend(table.to_pylist())                    # -> list[dict]
        if limit and len(rows) >= limit:
            break
    cached.write_text("\n".join(json.dumps(r, default=str) for r in rows))
    return rows[:limit] if limit else rows
