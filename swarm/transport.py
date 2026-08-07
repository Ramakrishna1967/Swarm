"""Transport adapter for the model tool loop (§2.3, §2.2 admission).

The rest of swarm talks to a minimal ``Transport`` interface. Live mode uses
the Anthropic SDK; the same interface is implemented by a scripted mock used
in tests and demos when no API key is present. This keeps the orchestrator,
admission control, and tool loop independent of any one vendor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    inputs: dict = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class TurnResult:
    content: list = field(default_factory=list)
    stop_reason: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def is_tool(self) -> bool:
        return bool(self.tool_calls)


class TransportError(RuntimeError):
    pass


class Transport:
    """Interface each model backend must implement."""

    def warmup(self, messages, tools) -> None:
        """Fire a cache-warm request against the shared prefix (§2.2)."""
        raise NotImplementedError

    def turn(self, messages, tools, **params) -> TurnResult:
        """One assistant turn: returns content, stop reason and tool calls."""
        raise NotImplementedError


def _make_tool_call(name: str, inputs: dict) -> ToolCall:
    return ToolCall(id=f"tc_{abs(hash((name, str(inputs)))) % 10_000_000}", name=name, inputs=inputs)


class AnthropicTransport(Transport):
    """Live transport backed by the anthropic SDK."""

    def __init__(
        self,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        max_tokens: int = 64000,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        try:
            import anthropic  # type: ignore
            self.client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
        except Exception as exc:  # pragma: no cover
            raise TransportError(f"could not initialise Anthropic client: {exc}")

    def warmup(self, messages, tools) -> None:
        self.client.beta.messages.create(
            model=self.model, messages=messages, tools=tools,
            max_tokens=1, stream=False,
        )

    def _parse(self, response) -> TurnResult:
        tool_calls = []
        content = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, inputs=dict(block.input)))
            else:
                content.append(block)
        usage = getattr(response, "usage", None)
        return TurnResult(
            content=content, stop_reason=response.stop_reason, tool_calls=tool_calls,
            usage=Usage(
                input_tokens=getattr(usage, "input_tokens", 0),
                output_tokens=getattr(usage, "output_tokens", 0),
                cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0),
            ),
        )

    def turn(self, messages, tools, **extra) -> TurnResult:
        response = self.client.beta.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=messages, tools=tools, stream=False,
        )
        return self._parse(response)


class MockTransport(Transport):
    """Scripted transport for tests/demos (no API key required)."""

    def __init__(self, script: list[dict] | None = None, root: str | None = None) -> None:
        self.script = list(script or [])
        self.index = 0
        self.root = root

    def warmup(self, messages, tools) -> None:
        return None

    def turn(self, messages, tools, **extra) -> TurnResult:
        if not self.script:
            return TurnResult(stop_reason="stop")
        step = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        calls = []
        if step.get("tool") and not step.get("finish"):
            calls.append(_make_tool_call(step["tool"], step.get("inputs", {})))
        return TurnResult(
            content=[], stop_reason="stop" if step.get("finish") else "tool_use",
            tool_calls=calls, usage=Usage(input_tokens=1, output_tokens=1),
        )