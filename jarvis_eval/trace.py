"""What one agent run produced — the unit every runner scores."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# input/output $ per 1M tokens, mirrored from jarvis-backend AVAILABLE_MODELS
# (app/schemas/chat.py). Used for a rough per-run cost estimate.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-v4-flash": (0.09, 0.18),
    "qwen/qwen3.5-9b": (0.10, 0.15),
    "qwen/qwen3.7-plus": (0.32, 1.28),
    "deepseek/deepseek-v4-pro": (0.435, 0.87),
    "anthropic/claude-opus-4.8": (1.70, 25.00),
}


def usd_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = MODEL_PRICING.get(model, (0.0, 0.0))
    return round(input_tokens / 1e6 * pin + output_tokens / 1e6 * pout, 6)


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    task_run_id: str | None = None  # set when the call was inside a subagent


@dataclass
class RunTrace:
    case_id: str
    model: str
    thread_id: str | None = None
    final_text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # paths surfaced by search_files / grep_files, in call order (parsed from
    # the tool output text) — the agent's *effective* retrieval set
    retrieval_paths: list[str] = field(default_factory=list)
    # present_file events: [{name, mime, size, path}]
    file_outputs: list[dict] = field(default_factory=list)
    hitl_rounds: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0
    stopped: bool = False          # hit MAX_HITL_ROUNDS or a stop
    error: str | None = None
    messages: list[dict] = field(default_factory=list)  # serialized parts

    @property
    def usd(self) -> float:
        return usd_cost(self.model, self.input_tokens, self.output_tokens)

    @property
    def turns(self) -> int:
        """Number of tool calls the main agent made (a rough 'steps' proxy)."""
        return sum(1 for t in self.tool_calls if t.task_run_id is None)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tool_calls]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usd"] = self.usd
        d["turns"] = self.turns
        return d
