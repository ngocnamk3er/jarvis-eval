"""HotpotQA (distractor dev) — the standard multi-hop RAG benchmark.

Each question needs 2 Wikipedia paragraphs out of a 10-paragraph pool. We
merge every sampled question's pool into one workspace (so other questions'
paragraphs act as extra distractors — harder, more realistic), then score:

- retrieval: do the 2 supporting-fact paragraphs come back in the top-k?
- answer:    full agent run → EM / F1 vs the gold answer (official SQuAD
             normalisation), on a sub-sample (BENCH_AGENT_SAMPLE).
- support:   did the agent actually read both supporting paragraphs?
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
TITLEMAP = hf.CACHE / "hotpotqa_titlemap.json"
KS = (2, 5, 10)
_INSTRUCTION = (
    "Answer with just the answer, on a line starting 'FINAL ANSWER:'. Use "
    "search_files / read_file over the workspace to find the two facts you "
    "need — the question requires combining information from two documents."
)


def user() -> str:
    return f"{settings.EVAL_USERNAME}-hotpotqa"


def _fid(title: str) -> str:
    return hashlib.md5(title.encode()).hexdigest()[:12]


def _fname(title: str) -> str:
    return _fid(title) + ".txt"


def _sample() -> list[dict]:
    rows = hf.load_rows(DS, CONFIG, SPLIT)
    random.Random(42).shuffle(rows)
    return rows[: settings.HOTPOTQA_SAMPLE]


def _para_text(title: str, sentences: list[str]) -> str:
    return (title + "\n\n" + " ".join(sentences)).strip()


def _support_titles(row: dict) -> list[str]:
    sf = row["supporting_facts"]
    return list(dict.fromkeys(sf["title"] if isinstance(sf, dict) else [t for t, _ in sf]))


# ---- seed ---------------------------------------------------------------
def seed(_suite: str = "hotpotqa") -> dict:
    u = user()
    rows = _sample()
    paragraphs: dict[str, str] = {}   # title -> body
    for row in rows:
        ctx = row["context"]
        titles = ctx["title"] if isinstance(ctx, dict) else [t for t, _ in ctx]
        sents = ctx["sentences"] if isinstance(ctx, dict) else [s for _, s in ctx]
        for t, s in zip(titles, sents):
            paragraphs.setdefault(t, _para_text(t, list(s)))

    # keyed by the bare md5 (== PurePosixPath(path).stem at score time)
    titlemap = {_fid(t): t for t in paragraphs}
    TITLEMAP.write_text(json.dumps(titlemap))

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


# ---- run -------------------------------------------------------------------
def run(_cases, repeats: int = 1, suite: str = "hotpotqa") -> list[dict]:
    u = user()
    rows = _sample()
    titlemap = json.loads(TITLEMAP.read_text()) if TITLEMAP.exists() else {}
    agent_ids = {r["id"] for r in rows[: settings.BENCH_AGENT_SAMPLE]}

    results: list[dict] = []
    for row in rows:
        qid, question, gold = row["id"], row["question"], row["answer"]
        gold_titles = _support_titles(row)
        m: dict = {}
        meta: dict = {"question": question, "gold": gold, "gold_titles": gold_titles,
                      "level": row.get("level"), "type": row.get("type")}

        # --- retrieval (all questions) ---
        try:
            hits = files.search_vector(question, top_k=40, user=u)
            seen: set[str] = set()
            ranked_titles = [t for h in hits
                             if (t := titlemap.get(PurePosixPath(h["path"]).stem, "?")) not in seen
                             and not seen.add(t)][:10]
            for k in KS:
                got = set(ranked_titles[:k]) & set(gold_titles)
                m[f"support_recall@{k}"] = len(got) / len(gold_titles) if gold_titles else 0.0
                m[f"both@{k}"] = 1.0 if len(got) == len(gold_titles) else 0.0
            meta["ranked_titles"] = ranked_titles[:10]
        except Exception as e:  # noqa: BLE001
            meta["retrieval_error"] = f"{type(e).__name__}: {e}"

        # --- answer (sub-sample) ---
        if qid in agent_ids:
            tr = chat.run_agent(f"{question}\n\n({_INSTRUCTION})", qid,
                                web_search=False, user=u)
            pred = extract_final_answer(tr.final_text)
            m["answer_em"] = answer_em(pred, gold)
            m["answer_f1"] = round(answer_f1(pred, gold), 4)
            # titles the agent actually surfaced (search_files results + read_file targets)
            touched = {PurePosixPath(p).stem for p in tr.retrieval_paths}
            touched |= {PurePosixPath(str(tc.input.get("path", ""))).stem
                        for tc in tr.tool_calls if tc.name == "read_file"}
            touched_titles = {titlemap.get(s) for s in touched} - {None}
            m["support_read"] = 1.0 if set(gold_titles) <= touched_titles else 0.0
            m["latency_s"] = tr.wall_seconds
            m["usd"] = tr.usd
            meta.update(pred=pred[:300], tool_sequence=tr.tool_names,
                        touched_titles=sorted(touched_titles), stopped=tr.stopped, error=tr.error)

        results.append({"suite": "hotpotqa", "case_id": qid, "repeat": 0,
                        "metrics": m, "meta": meta})
    return results
