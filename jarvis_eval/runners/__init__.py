"""Suite runners. Each `run(cases, repeats) -> list[CaseResult]`."""
from jarvis_eval.runners import agent_tasks, rag_qa, retrieval

RUNNERS = {
    "retrieval": retrieval.run,
    "rag_qa": rag_qa.run,
    "agent_tasks": agent_tasks.run,
}
