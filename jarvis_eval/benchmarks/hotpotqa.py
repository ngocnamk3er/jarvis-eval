"""HotpotQA (distractor dev) — the standard multi-hop RAG benchmark.

Every question ships a pool of 10 Wikipedia paragraphs: 2 that actually
contain the answer ("supporting facts") and 8 topically-related distractors.
Answering needs both supporting paragraphs (a 2-hop chain, e.g. "who played
X in film Y" -> "what job did that person have").

We take a fixed sample of HOTPOTQA_SAMPLE questions (seed=42) and merge ALL
their paragraph pools into one workspace, de-duped by article title. So one
question's distractors also distract every other question — harder and more
realistic than scoring each pool in isolation.

Then per question:
  * retrieval  — does search_files return both supporting titles in the top k?
                 -> support_recall@k, both@k     (scored on every sampled Q)
  * answer     — full agent run -> EM / F1 vs the gold answer
                 (scored on the first BENCH_AGENT_SAMPLE only — agent runs are slow)
  * support_read — did the agent actually open both supporting paragraphs?

Files are named `<md5(title)[:12]>.txt`; `titlemap.json` maps that id back to
the title so scoring can compare retrieved ids to the gold titles.
"""
import hashlib
import json
import random
from pathlib import PurePosixPath

from jarvis_eval.benchmarks import hf
from jarvis_eval.clients import chat, files
from jarvis_eval.config import settings
from jarvis_eval.metrics import answer_em, answer_f1, extract_final_answer

DS, CONFIG, SPLIT = "hotpotqa/hotpot_qa", "distractor", "validation"
FOLDER = "/bench/hotpotqa"
TITLEMAP = hf.CACHE / "hotpotqa_titlemap.json"      # {file_id: article_title}
KS = (2, 5, 10)
_INSTRUCTION = (
    "Answer with just the answer, on a line starting 'FINAL ANSWER:'. Use "
    "search_files / read_file over the workspace to find the two facts you "
    "need — the question requires combining information from two documents."
)


def user() -> str:
    return f"{settings.EVAL_USERNAME}-hotpotqa"


def _fid(title: str) -> str:
    """Stable 12-hex id for an article title (== filename stem)."""
    return hashlib.md5(title.encode()).hexdigest()[:12]


def _fname(title: str) -> str:
    return _fid(title) + ".txt"


def _sample() -> list[dict]:
    """The same HOTPOTQA_SAMPLE questions every call (fixed shuffle seed)."""
    rows = hf.load_rows(DS, CONFIG, SPLIT)          # 7405 validation questions
    random.Random(42).shuffle(rows)
    return rows[: settings.HOTPOTQA_SAMPLE]


def _para_text(title: str, sentences: list[str]) -> str:
    return (title + "\n\n" + " ".join(sentences)).strip()


def _support_titles(row: dict) -> list[str]:
    """The (usually 2) article titles that contain the answer."""
    sf = row["supporting_facts"]
    return list(dict.fromkeys(sf["title"] if isinstance(sf, dict) else [t for t, _ in sf]))


# =========================================================================
# seed  —  jeval seed hotpotqa
# =========================================================================
def seed(_suite: str = "hotpotqa") -> dict:
    u = user()
    rows = _sample()

    # union of every sampled question's 10 paragraphs, de-duped by title
    paragraphs: dict[str, str] = {}   # title -> body
    for row in rows:
        ctx = row["context"]                        # {title: [...], sentences: [[...], ...]}
        titles = ctx["title"] if isinstance(ctx, dict) else [t for t, _ in ctx]
        sents = ctx["sentences"] if isinstance(ctx, dict) else [s for _, s in ctx]
        for t, s in zip(titles, sents):
            paragraphs.setdefault(t, _para_text(t, list(s)))

    TITLEMAP.write_text(json.dumps({_fid(t): t for t in paragraphs}))

    files.wipe(user=u)
    files.ensure_folder(FOLDER, user=u)
    docs = [(_fname(t), body) for t, body in paragraphs.items()]
    if len(docs) > settings.BENCH_MAX_DOCS:
        docs = docs[: settings.BENCH_MAX_DOCS]
    n = files.bulk_upload(docs, parent_path=FOLDER, user=u)
    done = files.wait_for_indexing(timeout=max(300.0, n * 0.4), user=u)

    statuses: dict[str, int] = {}
    for e in done:
        statuses[e["indexing_status"]] = statuses.get(e["indexing_status"], 0) + 1
    return {"questions": len(rows), "paragraphs": n, "statuses": statuses}


# =========================================================================
# run  —  jeval run --suite hotpotqa
# =========================================================================
def run(_cases, repeats: int = 1, suite: str = "hotpotqa") -> list[dict]:
    u = user()
    rows = _sample()
    titlemap = json.loads(TITLEMAP.read_text()) if TITLEMAP.exists() else {}
    agent_ids = {r["id"] for r in rows[: settings.BENCH_AGENT_SAMPLE]}   # who gets an agent run

    results: list[dict] = []
    for row in rows:
        qid, question, gold = row["id"], row["question"], row["answer"]
        gold_titles = _support_titles(row)
        m: dict = {}
        meta: dict = {"question": question, "gold": gold, "gold_titles": gold_titles,
                      "level": row.get("level"), "type": row.get("type")}

        # --- retrieval: every sampled question ---
        try:
            hits = files.search_vector(question, top_k=40, user=u)
            seen: set[str] = set()
            # retrieved article titles, de-duped, top 10
            ranked_titles = [t for h in hits
                             if (t := titlemap.get(PurePosixPath(h["path"]).stem, "?")) not in seen
                             and not seen.add(t)][:10]
            for k in KS:
                got = set(ranked_titles[:k]) & set(gold_titles)
                m[f"support_recall@{k}"] = len(got) / len(gold_titles) if gold_titles else 0.0
                m[f"both@{k}"] = 1.0 if len(got) == len(gold_titles) else 0.0   # found ALL supporting titles
            meta["ranked_titles"] = ranked_titles[:10]
        except Exception as e:  # noqa: BLE001 — record, keep scoring the suite
            meta["retrieval_error"] = f"{type(e).__name__}: {e}"

        # --- answer: only the sub-sample ---
        if qid in agent_ids:
            tr = chat.run_agent(f"{question}\n\n({_INSTRUCTION})", qid,
                                web_search=False, user=u)
            pred = extract_final_answer(tr.final_text)
            m["answer_em"] = answer_em(pred, gold)
            m["answer_f1"] = round(answer_f1(pred, gold), 4)
            # did the agent actually see both supporting paragraphs?
            touched = {PurePosixPath(p).stem for p in tr.retrieval_paths}              # from search_files
            touched |= {PurePosixPath(str(tc.input.get("path", ""))).stem
                        for tc in tr.tool_calls if tc.name == "read_file"}             # from read_file
            touched_titles = {titlemap.get(s) for s in touched} - {None}
            m["support_read"] = 1.0 if set(gold_titles) <= touched_titles else 0.0
            m["latency_s"] = tr.wall_seconds
            m["usd"] = tr.usd
            meta.update(pred=pred[:300], tool_sequence=tr.tool_names,
                        touched_titles=sorted(touched_titles), stopped=tr.stopped, error=tr.error)

        results.append({"suite": "hotpotqa", "case_id": qid, "repeat": 0,
                        "metrics": m, "meta": meta})
    return results
