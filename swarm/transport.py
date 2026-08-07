"""Transport adapters for the model tool loop (§2.3, §2.2 admission).

The rest of swarm talks to a minimal ``Transport`` interface and a
*vendor-neutral* wire format:

* messages: OpenAI chat-completions shape (``system``/``user``/``assistant``
  with ``tool_calls``/``tool`` result messages),
* tools: OpenAI function shape
  (``{"type": "function", "function": {name, description, parameters}}``).

Each backend translates to its own API and back. Two real backends ship:

* ``OpenAICompatTransport`` -- any OpenAI-compatible chat-completions
  endpoint (OpenAI, Groq, OpenRouter, Ollama, vLLM, LM Studio, ...), built on
  stdlib ``urllib`` so the package stays dependency-free. Set an API key via
  the provider's usual env var (e.g. ``OPENAI_API_KEY``, ``GROQ_API_KEY``).
* ``AnthropicTransport`` -- the Anthropic Messages API (``ANTHROPIC_API_KEY``).

``MockTransport`` is a scripted no-op used in tests and demos; it needs no key.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any


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
    rate_rpm: float | None = None

    @property
    def is_tool(self) -> bool:
        return bool(self.tool_calls)


class TransportError(RuntimeError):
    """Non-retryable transport failure (auth, bad request, unsupported)."""


class RateLimitError(TransportError):
    """HTTP 429: the request was well-formed but the API is throttling us.

    Carries the server's ``Retry-After`` (seconds) when the provider sends it.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RetryableTransportError(TransportError):
    """Connection errors, timeouts, and 5xx: safe to retry with backoff."""


class Transport:
    """Interface each model backend must implement."""

    def warmup(self, messages, tools) -> None:
        """Fire a cache-warm request against the shared prefix (§2.2)."""
        raise NotImplementedError

    def turn(self, messages, tools, **params) -> TurnResult:
        """One assistant turn: returns content, stop reason and tool calls."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Canonical <-> vendor conversions
# ---------------------------------------------------------------------------

def _function_tools(tools: list[dict]) -> list[dict]:
    """Normalize any supported tool schema to the OpenAI function shape.

    Accepts either the canonical ``{"type": "function", "function": ...}``
    form or the legacy ``{"name", "description", "input_schema"}`` form used
    by older swarm code and the Anthropic API.
    """
    out: list[dict] = []
    for t in tools or []:
        if "function" in t and "name" in t["function"]:
            out.append(t)
        elif "name" in t:
            out.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
    return out


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """OpenAI function shape -> Anthropic ``name/description/input_schema``."""
    out: list[dict] = []
    for t in _function_tools(tools):
        fn = t["function"]
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


def _to_anthropic(messages: list[dict]) -> tuple[str | list, list[dict]]:
    """Canonical chat messages -> Anthropic (system, messages).

    ``system`` is a separate parameter in the Messages API; tool results are
    ``role: "user"`` blocks with ``tool_result``; tool uses are ``content``
    blocks on the assistant message.
    """
    system: list[str] = []
    out: list[dict] = []
    for m in messages or []:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            if content:
                system.append(content)
            continue
        if role == "assistant":
            calls = m.get("tool_calls") or []
            blocks = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in calls:
                fn = tc.get("function", {})
                try:
                    payload = json.loads(fn.get("arguments") or "{}")
                except (TypeError, json.JSONDecodeError):
                    payload = {}
                blocks.append({"type": "tool_use", "id": tc["id"],
                               "name": fn.get("name", ""), "input": payload})
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""),
                 "content": content or ""},
            ]})
        else:  # user and anything else
            out.append({"role": "user", "content": content or ""})
    return (system if len(system) > 1 else (system[0] if system else "")), out


def _from_openai_usage(usage: dict | None) -> Usage:
    usage = usage or {}
    cache = usage.get("prompt_cache_hit_tokens", 0)
    if not cache and isinstance(usage.get("prompt_tokens_details"), dict):
        cache = usage["prompt_tokens_details"].get("cached_tokens", 0)
    return Usage(
        input_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=int(usage.get("completion_tokens", 0)),
        cache_read_input_tokens=int(cache),
    )


def _rate_limit_from(headers) -> float | None:
    """Requests-per-minute from a provider's ``*ratelimit*`` headers.

    OpenAI/Groq/OpenRouter send ``x-ratelimit-limit-requests``; the Anthropic
    style is ``anthropic-ratelimit-requests-limit``. Missing/unknown shapes
    return None -- the admission gate then keeps its existing rate.
    """
    if headers is None:
        return None
    for key in ("x-ratelimit-limit-requests", "x-ratelimit-requests-limit",
                "anthropic-ratelimit-requests-limit", "ratelimit-limit"):
        try:
            raw = headers.get(key)
        except AttributeError:
            continue
        if raw:
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
    return None


def _retry_after(headers) -> float | None:
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After") or headers.get("retry-after")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None

# ---------------------------------------------------------------------------
# OpenAI-compatible backend (OpenAI, Groq, OpenRouter, Ollama, vLLM, ...)
# ---------------------------------------------------------------------------

class OpenAICompatTransport(Transport):
    """Any OpenAI-compatible chat-completions endpoint, stdlib only.

    ``base_url`` is the full API root, e.g. ``https://api.openai.com/v1``,
    ``https://api.groq.com/openai/v1`` or ``http://localhost:11434/v1`` for
    Ollama. Endpoints that need no key (Ollama, LM Studio) can leave
    ``api_key=None``.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_tokens: int = 16384,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _request(self, messages, tools, **extra) -> tuple[dict, object | None]:
        body: dict[str, Any] = {"model": self.model, "messages": messages, **extra}
        fn_tools = _function_tools(tools)
        if fn_tools:
            body["tools"] = fn_tools
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self._endpoint(), data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8")), resp.headers
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            if exc.code == 401:
                raise TransportError(
                    f"authentication failed (HTTP 401) for {self.model}: {detail}"
                    " -- set the provider API key (e.g. OPENAI_API_KEY) or pass --api-key"
                ) from exc
            if exc.code == 429:
                # not a client bug; retry after the server's backoff window
                raise RateLimitError(
                    f"HTTP 429 from {self._endpoint()}"
                    + (f"; retry-after {hdr}s" if (hdr := _retry_after(exc.headers)) else "")
                    + f": {detail}",
                    retry_after=_retry_after(exc.headers),
                ) from exc
            if exc.code >= 500 or exc.code in (408, 425):
                raise RetryableTransportError(
                    f"HTTP {exc.code} from {self._endpoint()}: {detail}"
                ) from exc
            raise TransportError(
                f"HTTP {exc.code} (not retryable) from {self._endpoint()}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RetryableTransportError(
                f"could not reach {self._endpoint()}: {exc}"
            ) from exc

    def warmup(self, messages, tools) -> None:
        self._request(messages, tools, max_tokens=1)

    def turn(self, messages, tools, **params) -> TurnResult:
        max_tokens = params.pop("max_tokens", self.max_tokens)
        resp, headers = self._request(messages, tools, max_tokens=max_tokens, **params)
        try:
            choice = resp["choices"][0]
            msg = choice.get("message", {})
            tool_calls = [
                ToolCall(
                    id=tc.get("id", "tc_" + uuid.uuid4().hex[:8]),
                    name=(tc.get("function") or {}).get("name", ""),
                    inputs=_json_args((tc.get("function") or {}).get("arguments")),
                )
                for tc in (msg.get("tool_calls") or [])
            ]
            content = msg.get("content")
            stop_reason = choice.get("finish_reason")
            return TurnResult(
                content=[content] if content else [],
                stop_reason=stop_reason,
                tool_calls=tool_calls,
                usage=_from_openai_usage(resp.get("usage")),
                rate_rpm=_rate_limit_from(headers),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise TransportError(f"unexpected response from {self._endpoint()}: {resp}") from exc


def _json_args(raw: str | None) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _anthropic_error(exc: Exception) -> TransportError:
    """Translate an anthropic SDK exception into swarm's error hierarchy.

    The anthropic package is an optional dependency, so classification is by
    type name / status code rather than a hard import.
    """
    name = type(exc).__name__
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if "RateLimit" in name or status == 429:
        retry_after = getattr(exc, "response", None)
        ra = _retry_after(getattr(retry_after, "headers", None)) if retry_after else None
        return RateLimitError(f"anthropic rate limited: {exc}", retry_after=ra)
    if ("Connection" in name or "Timeout" in name or status in (None,)) \
            and not status:
        return RetryableTransportError(f"anthropic: {exc}")
    if isinstance(status, int) and status >= 500:
        return RetryableTransportError(f"anthropic HTTP {status}: {exc}")
    return TransportError(f"anthropic {name}: {exc}")


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

class AnthropicTransport(Transport):
    """Anthropic Messages API backend (``ANTHROPIC_API_KEY``)."""

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
            raise TransportError(f"could not initialise Anthropic client: {exc}") from exc

    def warmup(self, messages, tools) -> None:
        try:
            system, msgs = _to_anthropic(messages)
            self.client.beta.messages.create(
                model=self.model, messages=msgs, system=system,
                tools=_to_anthropic_tools(tools), max_tokens=1, stream=False,
            )
        except TransportError:
            raise
        except Exception as exc:
            raise _anthropic_error(exc) from exc

    def _parse(self, response) -> TurnResult:
        tool_calls: list[ToolCall] = []
        content: list[Any] = []
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
        system, msgs = _to_anthropic(messages)
        try:
            response = self.client.beta.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                messages=msgs, system=system,
                tools=_to_anthropic_tools(tools), stream=False,
            )
        except TransportError:
            raise
        except Exception as exc:
            raise _anthropic_error(exc) from exc
        return self._parse(response)


# ---------------------------------------------------------------------------
# Scripted mock (tests / demos, no key)
# ---------------------------------------------------------------------------

def _make_tool_call(name: str, inputs: dict) -> ToolCall:
    return ToolCall(id="tc_" + uuid.uuid4().hex[:8], name=name, inputs=inputs)


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
