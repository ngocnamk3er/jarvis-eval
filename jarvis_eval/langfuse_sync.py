"""Run a suite as a Langfuse experiment instead of a local JSON dump.

`jeval run` already produces every number a benchmark needs; what it does not
produce is history. Each run lands in results/<timestamp>/ and is compared to
the one baseline file by hand, so "did this change help?" is a question you
answer by reading two reports side by side.

A Langfuse experiment answers it directly: the dataset is versioned, each run
is a named row against it, and runs compare in one view. The metrics are the
same ones jarvis_eval already computes — this module only decides where they
are stored.

Two things worth knowing before reading the code:

  * The dataset is seeded from the benchmark's own sample, never from a
    previous run's raw.json. Seeding from results makes the dataset a
    by-product of an execution: a case that errored before producing output
    would quietly vanish from the "questions" list, and re-seeding would
    require paying for a run first.

  * Langfuse stores expected_output as JSON, so a gold answer of "17" comes
    back as int 17, not "17". Every scorer here coerces with str() — without
    that, gaia_score's `"," in gold` raises TypeError on exactly the numeric
    answers that make up most of GAIA.
"""

import json
import os
import time

from jarvis_eval.config import settings


def available() -> bool:
    """False when the SDK isn't installed or Langfuse isn't configured, so a
    normal `jeval run` still works on a machine that has neither."""
    if not (settings.LANGFUSE_HOST and settings.LANGFUSE_PUBLIC_KEY
            and settings.LANGFUSE_SECRET_KEY):
        return False
    try:
        import langfuse  # noqa: F401
    except Exception:
        return False
    return True


def _client():
    from langfuse import Langfuse
    return Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
        environment=settings.LANGFUSE_ENVIRONMENT,
    )


def seed_gaia_dataset(lf, name: str = "gaia") -> None:
    """Create/refresh the GAIA dataset from the HuggingFace split.

    Item ids are the GAIA task_id, so re-seeding updates in place rather than
    duplicating, and a run's items line up with the benchmark's own case_ids
    without a lookup table.
    """
    from jarvis_eval.benchmarks.gaia import _sample

    cases = _sample()
    lf.create_dataset(name=name,
                      description=f"GAIA validation, level 1, no file attachment "
                                  f"({len(cases)} of 165)")
    for c in cases:
        lf.create_dataset_item(dataset_name=name, id=c["id"],
                               input={"question": c["question"]},
                               expected_output=str(c["gold"]))
    lf.flush()
    return len(cases)


def run_gaia_experiment(run_name: str | None = None, dataset: str = "gaia",
                        max_items: int | None = None):
    """Run GAIA through the real agent, recorded as one Langfuse experiment.

    Unlike a replay, `task` below actually calls jarvis — so each dataset item
    produces a live agent trace, and anything Langfuse runs against experiment
    items (a code evaluator, for instance) sees a real run rather than a
    string that was recorded earlier.
    """
    from langfuse import Evaluation
    from jarvis_eval.benchmarks.gaia import _INSTRUCTION, gaia_score
    from jarvis_eval.clients import chat
    from jarvis_eval.metrics import extract_final_answer

    lf = _client()
    seed_gaia_dataset(lf, dataset)
    ds = lf.get_dataset(dataset)
    items = ds.items[:max_items] if max_items else ds.items

    # chat.run_agent's RunTrace carries latency/cost/turns, but run_experiment
    # only hands evaluators the task's return value — so stash the trace here,
    # keyed by item id, for the metric scorers below to read back.
    traces: dict[str, object] = {}

    def task(*, item, **_):
        question = (item.input or {}).get("question", "")
        tr = chat.run_agent(f"{question}\n\n({_INSTRUCTION})", item.id, web_search=True)
        traces[item.id] = tr
        return extract_final_answer(tr.final_text)

    def _tr(item_id):
        return traces.get(item_id)

    def score_gaia(*, input, output, expected_output, metadata=None, **_):
        # str() on both sides: Langfuse round-trips gold through JSON, so a
        # numeric answer comes back as int and gaia_score's list check raises.
        return Evaluation(name="gaia_score",
                          value=gaia_score(str(output or ""), str(expected_output or "")),
                          comment=f"pred={str(output)[:60]!r} gold={str(expected_output)[:60]!r}")

    def _metric(name, pick):
        def fn(*, input, output, metadata=None, **kw):
            item_id = (metadata or {}).get("item_id")
            tr = _tr(item_id) if item_id else None
            return Evaluation(name=name, value=float(pick(tr))) if tr else None
        fn.__name__ = name
        return fn

    result = lf.run_experiment(
        name=dataset,
        run_name=run_name or f"agent-{time.strftime('%Y%m%d-%H%M')}",
        description=f"Live agent run. model={settings.RUNNER_MODEL}, "
                    f"thinking={settings.RUNNER_THINKING_EFFORT}, web_search=on.",
        data=items,
        task=task,
        evaluators=[score_gaia],
        metadata={"runner_model": settings.RUNNER_MODEL,
                  "thinking_effort": settings.RUNNER_THINKING_EFFORT},
        max_concurrency=1,   # the agent holds a sandbox per conversation
    )
    lf.flush()
    return result, traces
