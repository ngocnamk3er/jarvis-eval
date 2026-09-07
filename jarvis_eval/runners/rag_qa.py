"""RAG QA suite — full agent run over the seeded workspace, LLM-judged.

Each case: a question answerable from the corpus. The agent is told to use
its file tools. We grade the final answer for correctness (vs a reference)
and groundedness (vs what it actually retrieved), plus citation recall and
the usual efficiency numbers.

Case shape (datasets/rag_qa.jsonl):
  {"id","query","gold_answer","must_cite_paths":[...],"acceptable_points":[...]}
"""
from jarvis_eval.clients import chat
from jarvis_eval import judge

_INSTRUCTION = (
    "Answer using ONLY the user's file workspace — use search_files / grep_files "
    "to find the relevant file(s) and read_file to read them. Cite the file "
    "path(s) you used. If the workspace doesn't contain the answer, say so."
)


def _context_from_trace(trace) -> str:
    parts = []
    for tc in trace.tool_calls:
        if tc.name in ("search_files", "grep_files", "read_file"):
            parts.append(f"[{tc.name} {tc.input}]\n{tc.output}")
    return "\n\n".join(parts)[:8000]


def run(cases: list[dict], repeats: int = 1) -> list[dict]:
    results: list[dict] = []
    for case in cases:
        for rep in range(repeats):
            prompt = f"{case['query']}\n\n({_INSTRUCTION})"
            trace = chat.run_agent(prompt, case["id"], web_search=False)

            answer = trace.final_text
            context = _context_from_trace(trace)
            cited = [p for p in case.get("must_cite_paths", [])
                     if p in answer or p.lstrip("/") in answer]
            citation_recall = (len(cited) / len(case["must_cite_paths"])
                               if case.get("must_cite_paths") else 1.0)

            correctness = judge.score_answer(
                case["query"], case["gold_answer"], case.get("acceptable_points", []), answer)
            grounded = judge.score_groundedness(answer, context)

            results.append({
                "suite": "rag_qa",
                "case_id": case["id"],
                "repeat": rep,
                "metrics": {
                    "answer_correctness": correctness.get("score"),
                    "groundedness": grounded.get("score"),
                    "citation_recall": round(citation_recall, 3),
                    "retrieved_gold": 1.0 if any(
                        p in trace.retrieval_paths for p in case.get("must_cite_paths", [])
                    ) else 0.0,
                    "latency_s": trace.wall_seconds,
                    "usd": trace.usd,
                    "turns": trace.turns,
                },
                "meta": {
                    "query": case["query"],
                    "answer": answer[:2000],
                    "retrieval_paths": trace.retrieval_paths,
                    "tool_sequence": trace.tool_names,
                    "correctness_reasoning": correctness.get("reasoning"),
                    "groundedness_reasoning": grounded.get("reasoning"),
                    "hitl_rounds": trace.hitl_rounds,
                    "stopped": trace.stopped,
                    "error": trace.error,
                },
            })
    return results
