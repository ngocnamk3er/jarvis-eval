"""`RunTrace` — everything one agent run produced, in one object.

The hotpotqa / gaia suites drive the real agent through /chat/stream
(see clients/chat.py) and collect the result here: the final answer, the
tools it called, what it retrieved, how many tokens / dollars / seconds it
cost. The suites then score that. (BEIR never builds one of these — it only
does retrieval, no agent.)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# (input $, output $) per 1,000,000 tokens — copied from jarvis-backend's
# AVAILABLE_MODELS (app/schemas/chat.py). Only used for a rough cost estimate
# in the report; an unknown model just costs $0.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-v4-flash": (0.09, 0.18),
    "qwen/qwen3.5-9b": (0.10, 0.15),
    "qwen/qwen3.7-plus": (0.32, 1.28),
    "deepseek/deepseek-v4-pro": (0.435, 0.87),
    "anthropic/claude-opus-4.8": (1.70, 25.00),
}


def usd_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """input*price_in + output*price_out, in dollars."""
    pin, pout = MODEL_PRICING.get(model, (0.0, 0.0))
    return round(input_tokens / 1e6 * pin + output_tokens / 1e6 * pout, 6)


@dataclass
class ToolCall:
    """One tool the agent invoked during a run (bash, search_files, ...)."""
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    # set to the parent `task` run_id when the call happened inside a
    # sub-agent's own loop rather than the main agent's turn.
    task_run_id: str | None = None


@dataclass
class RunTrace:
    case_id: str          # the benchmark question id this run answered
    model: str            # which model the agent ran as (for cost)
    thread_id: str | None = None   # the conversation created for this run

    final_text: str = ""           # the agent's last assistant message (the answer)
    tool_calls: list[ToolCall] = field(default_factory=list)

    # File paths the agent actually surfaced via search_files,
    # in call order — parsed out of the tool output text in chat.py. This is
    # the agent's *effective* retrieval set (what it saw), used by hotpotqa
    # to check it found both supporting paragraphs.
    retrieval_paths: list[str] = field(default_factory=list)

    # `present_file` events the agent emitted: [{name, mime, size, path}].
    file_outputs: list[dict] = field(default_factory=list)

    hitl_rounds: int = 0           # how many bash prompts we auto-approved
    input_tokens: int = 0          # summed over every LLM call in the run
    output_tokens: int = 0
    wall_seconds: float = 0.0

    stopped: bool = False          # ran out of HITL rounds, or was cut short
    error: str | None = None       # transport / stream error, if any
    messages: list[dict] = field(default_factory=list)  # the serialized conversation

    @property
    def usd(self) -> float:
        return usd_cost(self.model, self.input_tokens, self.output_tokens)

    @property
    def turns(self) -> int:
        """A rough 'how many steps' proxy — tool calls the main agent made
        itself (excludes calls nested inside a sub-agent)."""
        return sum(1 for t in self.tool_calls if t.task_run_id is None)

    @property
    def tool_names(self) -> list[str]:
        """The tool-call sequence, e.g. ['search_files', 'read_file', 'read_file']."""
        return [t.name for t in self.tool_calls]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usd"] = self.usd        # asdict() drops @property values, add them back
        d["turns"] = self.turns
        return d
