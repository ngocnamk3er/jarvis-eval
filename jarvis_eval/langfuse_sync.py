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


def _text_of(inp) -> str:
    """The question text, whichever shape the dataset item's input is in.

    Items seeded by seed_gaia_dataset() store {"question": ...}; items
    uploaded as CSV through the UI store the bare string, because the upload
    maps one column to Input and does not wrap it. Both are legitimate, and a
    runner that only understands its own shape breaks the moment someone
    curates the dataset by hand — which is the whole point of having it in
    Langfuse rather than in code.
    """
    if isinstance(inp, dict):
        return str(inp.get("question") or inp.get("input") or "")
    return str(inp or "")


def _question_of(item) -> str:
    """The question text of a dataset item."""
    return _text_of(item.input)


def run_gaia_experiment(run_name: str | None = None, dataset: str = "gaia",
                        max_items: int | None = None, concurrency: int = 1):
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

    # Deliberately does not seed: seeding writes items keyed by GAIA task_id,
    # while items uploaded through the UI get their own ids, so re-seeding on
    # every run would add a second copy of every question rather than update
    # one. Seed explicitly with `jeval seed-dataset` when the questions change.
    lf = _client()
    ds = lf.get_dataset(dataset)
    items = ds.items[:max_items] if max_items else ds.items

    # chat.run_agent's RunTrace carries latency/cost/turns, but run_experiment
    # hands evaluators only the task's return value — so stash the trace here
    # for the metric scorers below. Keyed by the question, the one value both
    # the task and the evaluators are given.
    traces: dict[str, object] = {}

    def task(*, item, **_):
        question = _question_of(item)
        tr = chat.run_agent(f"{question}\n\n({_INSTRUCTION})", item.id, web_search=True)
        traces[question] = tr
        return extract_final_answer(tr.final_text, strict=True)

    def score_gaia(*, input, output, expected_output, metadata=None, **_):
        # str() on both sides: Langfuse round-trips gold through JSON, so a
        # numeric answer comes back as int and gaia_score's list check raises.
        return Evaluation(name="gaia_score",
                          value=gaia_score(str(output or ""), str(expected_output or "")),
                          comment=f"pred={str(output)[:60]!r} gold={str(expected_output)[:60]!r}")

    def answered(*, input, output, expected_output=None, **_):
        """Whether the agent committed to an answer at all.

        Separate from gaia_score because the two failures need opposite
        fixes: a wrong answer wants a better model or prompt, an absent one
        wants more budget. Both score 0, so without this they are one number.
        On 2026-09-23 six of 42 cases produced nothing — 14% of the suite lost
        to running out of turns rather than to being unable to reason.
        """
        return Evaluation(name="answered", value=1.0 if str(output or "").strip() else 0.0)

    def _from_trace(name: str, pick):
        """Lift a field off the RunTrace the task stashed for this item.

        Keyed by the question, because that is the only thing both sides see:
        the task gets the dataset item, the evaluator gets `input`, and
        Langfuse passes no item id between them. Questions are unique within
        a dataset, so the lookup is exact.
        """
        def fn(*, input, output, **_):
            tr = traces.get(_text_of(input))
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
        evaluators=[
            score_gaia,
            answered,
            _from_trace("latency_s", lambda t: t.wall_seconds),
            _from_trace("usd", lambda t: t.usd),
            _from_trace("turns", lambda t: len(t.tool_calls)),
            _from_trace("hitl_rounds", lambda t: t.hitl_rounds),
        ],
        metadata={"runner_model": settings.RUNNER_MODEL,
                  "thinking_effort": settings.RUNNER_THINKING_EFFORT},
        # Each case runs in its own conversation, and a conversation that
        # calls bash holds its own sandbox pod — so concurrency here is
        # really "how many sandboxes at once". The warm pool keeps one
        # standing by; the rest are created cold, which costs seconds at the
        # start of those cases but nothing after.
        max_concurrency=concurrency,
    )
    lf.flush()
    return result, traces


def replay_experiment(dataset: str, run_name: str | None = None,
                      field: str = "trace_output", max_items: int | None = None):
    """Re-score answers a previous run already produced, calling no model.

    The point is to isolate a change to the *scorer* from a change to the
    *agent*. Re-running the agent to test a scoring fix confounds the two:
    the model is not deterministic, so the score moves for reasons that have
    nothing to do with the fix. Replaying the recorded answers holds the
    agent fixed and measures the scorer alone.

    What it cannot do is detect a regression — the answers came from a build
    that no longer exists, so a replay run says nothing about the current
    agent, and naming one as if it did would mislead whoever reads the
    comparison later.

    Reads the stored answer from the dataset item's metadata (`trace_output`
    by default), which is where a dataset built from traces keeps it.
    """
    from langfuse import Evaluation
    from jarvis_eval.benchmarks.gaia import gaia_score

    lf = _client()
    ds = lf.get_dataset(dataset)
    items = ds.items[:max_items] if max_items else ds.items

    stored = {_text_of(i.input): str((i.metadata or {}).get(field) or "") for i in items}

    def task(*, item, **_):
        return str((item.metadata or {}).get(field) or "")

    def score_gaia(*, input, output, expected_output, **_):
        return Evaluation(name="gaia_score",
                          value=gaia_score(str(output or ""), str(expected_output or "")),
                          comment=f"pred={str(output)[:60]!r} gold={str(expected_output)[:60]!r}")

    def answered(*, input, output, **_):
        return Evaluation(name="answered", value=1.0 if str(output or "").strip() else 0.0)

    result = lf.run_experiment(
        name=dataset,
        run_name=run_name or f"replay-{time.strftime('%Y%m%d-%H%M')}",
        description=f"Replay of stored answers ({field}) against the dataset's "
                    f"expected output. No model was called — this measures the "
                    f"scorer, not the agent.",
        data=items,
        task=task,
        evaluators=[score_gaia, answered],
        metadata={"mode": "replay", "source_field": field},
        max_concurrency=10,   # no agent involved, nothing to serialise on
    )
    lf.flush()
    return result
