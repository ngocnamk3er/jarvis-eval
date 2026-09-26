"""SpreadsheetBench — real .xlsx files, instructions written by their users.

The other suites in this harness all reduce to text: BEIR and HotpotQA score
retrieval over passages, GAIA scores a short string. None of them can say
whether the agent can work with a spreadsheet, because flattening a workbook
to text is exactly what destroys the thing being tested — the row/column
relationships, the sheet boundaries, the cell types.

These tasks come from real spreadsheet help requests, so the files carry the
mess that goes with that: a group label alone on row 1, the real header on
row 2, tables starting at arbitrary offsets. `pd.read_excel` with default
arguments produces `Unnamed: 0` columns on most of them. That is the point —
an agent has to look at the file before it can read it.

Scoring needs no judge and no embeddings: each task names a cell range
(`answer_position`, e.g. "A3:D32"), and the agent's output workbook is
compared to the golden one over exactly that range.

Unlike the HF-parquet suites this one ships 800 .xlsx files in a tarball, so
it unpacks under datasets/ (gitignored) rather than into the shared row cache.
"""
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import httpx

REPO = "KAKA22/SpreadsheetBench"
# The 400-task subset whose answers the authors verified by hand. The full
# 912-task archive is the other file in the same repo, if a wider run is ever
# wanted; nothing here depends on which one is unpacked.
ARCHIVE = "spreadsheetbench_verified_400.tar.gz"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{ARCHIVE}"

ROOT = Path(os.environ.get("JARVIS_EVAL_SSB_DIR",
                           Path(__file__).resolve().parents[2] / "datasets" / "spreadsheetbench"))


def _dataset_file() -> Path:
    return ROOT / "dataset.json"


def download(force: bool = False) -> int:
    """Fetch and unpack the archive. Returns the task count.

    Skips the download when dataset.json is already there, so `jeval seed` is
    cheap to re-run — the archive is a fixed release, not something that
    changes under us.
    """
    if _dataset_file().exists() and not force:
        return len(json.loads(_dataset_file().read_text()))

    ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / ARCHIVE
        with httpx.stream("GET", URL, follow_redirects=True, timeout=300) as r:
            r.raise_for_status()
            with tarball.open("wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        with tarfile.open(tarball) as tf:
            tf.extractall(tmp)          # noqa: S202 — fixed archive from a pinned URL
        # The tarball wraps everything in one directory; lift its contents up
        # so ROOT/dataset.json holds regardless of what that directory is named.
        inner = next(p for p in Path(tmp).iterdir() if p.is_dir() and (p / "dataset.json").exists())
        for item in inner.iterdir():
            dest = ROOT / item.name
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(item), str(dest))
    return len(json.loads(_dataset_file().read_text()))


def tasks() -> list[dict]:
    """Every task, with absolute paths to its input and golden workbooks.

    Most task directories hold `<n>_<id>_init.xlsx` and `<n>_<id>_golden.xlsx`,
    but five of the 400 use the bare names `initial.xlsx` / `golden.xlsx`, so
    matching only the prefixed form silently drops them. Matching on the
    suffix covers both; the instruction in dataset.json is the same text as
    the prompt file, so only the workbooks are resolved here.
    """
    if not _dataset_file().exists():
        raise RuntimeError("SpreadsheetBench not downloaded — run `jeval seed spreadsheetbench`")
    out = []
    for t in json.loads(_dataset_file().read_text()):
        d = ROOT / "spreadsheet" / str(t["id"])
        if not d.is_dir():
            continue
        xlsx = sorted(d.glob("*.xlsx"))
        init = [p for p in xlsx if p.stem.endswith(("_init", "initial")) or p.stem == "initial"]
        golden = [p for p in xlsx if p.stem.endswith("golden")]
        if not init or not golden:
            continue
        out.append({**t, "init_path": str(init[0]), "golden_path": str(golden[0])})
    return out


def seed(_suite: str = "spreadsheetbench") -> dict:
    """Download the archive. Nothing is pushed to the file workspace here —
    a task's workbook belongs in the sandbox for that task, not in a corpus
    shared by all of them."""
    n = download()
    ts = tasks()
    return {"tasks": n, "usable": len(ts), "dir": str(ROOT),
            "note": "workbooks stay on disk; a runner copies one per task into the sandbox"}
