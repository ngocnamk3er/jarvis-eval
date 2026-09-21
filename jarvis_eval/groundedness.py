"""Groundedness metrics — is the answer actually supported by what was retrieved?

Every metric here until now scored one of two things: did retrieval find the
right documents (ndcg, recall, mrr), or did the final string match the gold
answer (answer_em, answer_f1). Neither catches the failure that matters most
in RAG — an answer that is *correct* but not *grounded*, produced from the
model's own memory while the retrieved context said nothing useful. That
answer scores full marks on f1 and will fall apart on any question the model
hasn't memorised.

Three metrics, from RAGAS rather than hand-rolled prompts:

  faithfulness       claims in the answer that the context actually supports
  answer_relevancy   whether the answer addresses the question asked
  context_precision  whether the retrieved passages were the useful ones

These are LLM-as-judge, so they cost a call per case and carry the judge's
own error. Treat them as a signal over a suite, not a verdict on one answer.

Pinning note: ragas needs `langchain-community<0.4`. At 0.4.x that package
dropped `langchain_community.chat_models.vertexai`, which ragas still imports
at module scope, so a fresh install of both latest versions fails on
`import ragas` — nothing to do with the metrics themselves.
"""

import asyncio
import logging

from jarvis_eval.config import settings

logger = logging.getLogger(__name__)

# Judge calls are independent, but firing all of a suite's cases at once
# invites provider rate limits, which surface as scores of 0 and look like a
# quality regression rather than throttling.
_CONCURRENCY = 4


def _judge_embeddings():
    """AnswerRelevancy is the odd one out: it works by generating questions the
    answer would fit and comparing them to the real one in vector space, so it
    needs an embedding model as well as a judge. Reuses the harness's existing
    embedding configuration — the same model the retrieval being scored was
    built with, which keeps the comparison in one space."""
    from openai import AsyncOpenAI
    from ragas.embeddings import embedding_factory

    client = AsyncOpenAI(
        api_key=settings.EMBEDDING_API_KEY or settings.OPENROUTER_API_KEY,
        base_url=settings.EMBEDDING_BASE_URL,
    )
    return embedding_factory(provider="openai", model=settings.EMBEDDING_MODEL, client=client)


def _judge_llm():
    """A ragas LLM backed by the same OpenRouter key the harness already uses.

    ragas talks to providers through `instructor`, which needs a real OpenAI
    client rather than a base URL string — hence building one here instead of
    passing settings through.
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    client = AsyncOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
    )
    return llm_factory(settings.GROUNDEDNESS_MODEL, provider="openai", client=client)


def available() -> bool:
    """False when ragas isn't installed or there's no key to judge with.

    Checked rather than assumed so a harness run still produces its retrieval
    numbers on a machine without the optional extra installed.
    """
    if not settings.OPENROUTER_API_KEY:
        return False
    try:
        import ragas  # noqa: F401
    except Exception:
        return False
    return True


async def _score_one(metrics, case: dict) -> dict:
    """Score a single case. Missing inputs skip a metric rather than zero it —
    a zero would drag the suite average down and read as a real regression."""
    faith, relevancy, precision = metrics
    question = case.get("question") or ""
    answer = case.get("answer") or ""
    contexts = [c for c in (case.get("contexts") or []) if c]
    reference = case.get("gold") or ""
    out: dict[str, float] = {}

    async def run(name, coro):
        try:
            res = await coro
            out[name] = float(res.value)
        except Exception:
            logger.warning("%s failed for case %s", name, case.get("id"), exc_info=True)

    jobs = []
    if question and answer and contexts:
        jobs.append(run("faithfulness", faith.ascore(
            user_input=question, response=answer, retrieved_contexts=contexts)))
    if question and answer:
        jobs.append(run("answer_relevancy", relevancy.ascore(
            user_input=question, response=answer)))
    if question and reference and contexts:
        jobs.append(run("context_precision", precision.ascore(
            user_input=question, reference=reference, retrieved_contexts=contexts)))
    if jobs:
        await asyncio.gather(*jobs)
    return out


async def score_cases(cases: list[dict]) -> list[dict]:
    """Score `cases`, each {id, question, answer, contexts, gold}.

    Returns one dict of metric -> value per case, aligned with the input, with
    metrics omitted where the inputs weren't there. Returns empty dicts rather
    than raising when ragas is unavailable, so callers don't branch.
    """
    if not cases or not available():
        return [{} for _ in cases]

    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        Faithfulness,
    )

    llm = _judge_llm()
    metrics = (
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=_judge_embeddings()),
        ContextPrecisionWithReference(llm=llm),
    )

    sem = asyncio.Semaphore(_CONCURRENCY)

    async def guarded(case):
        async with sem:
            return await _score_one(metrics, case)

    return await asyncio.gather(*(guarded(c) for c in cases))
