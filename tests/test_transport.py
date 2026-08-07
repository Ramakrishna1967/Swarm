"""Transport adapters: canonical conversions and response parsing.

Covers §2.3 wire-format plumbing that the rest of the suite exercises only
through MockTransport: canonical<->Anthropic conversion and the
OpenAI-compatible response parser.
"""

from __future__ import annotations

import io
import json
import urllib.error
from unittest import mock

from swarm.transport import (
    OpenAICompatTransport,
    RateLimitError,
    RetryableTransportError,
    TransportError,
    _from_openai_usage,
    _function_tools,
    _rate_limit_from,
    _to_anthropic,
    _to_anthropic_tools,
)


def test_function_tools_passthrough_and_legacy():
    legacy = [{"name": "read_file", "description": "d",
               "input_schema": {"type": "object", "properties": {}}}]
    fn = _function_tools(legacy)
    assert fn[0]["type"] == "function"
    assert fn[0]["function"]["name"] == "read_file"
    assert fn[0]["function"]["parameters"] == {"type": "object", "properties": {}}
    canonical = [{"type": "function", "function": {"name": "grep", "description": "d",
                                                   "parameters": {}}}]
    assert _function_tools(canonical) == canonical


def test_to_anthropic_splits_system_and_maps_tool_calls():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "tc1", "type": "function",
                         "function": {"name": "write_file", "arguments": '{"path":"a.py"}'}}]},
        {"role": "tool", "tool_call_id": "tc1", "content": "wrote a.py"},
    ]
    system, msgs = _to_anthropic(messages)
    assert system == "sys"
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    block = msgs[1]["content"][0]
    assert block["type"] == "tool_use"
    assert block["id"] == "tc1" and block["name"] == "write_file"
    assert block["input"] == {"path": "a.py"}
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert msgs[2]["content"][0]["tool_use_id"] == "tc1"


def test_to_anthropic_tools_maps_schema():
    tools = [{"type": "function", "function": {"name": "f", "description": "d",
                                               "parameters": {"type": "object"}}}]
    out = _to_anthropic_tools(tools)
    assert out == [{"name": "f", "description": "d", "input_schema": {"type": "object"}}]


def test_to_anthropic_multiple_system_joins():
    system, _ = _to_anthropic([
        {"role": "system", "content": "a"},
        {"role": "system", "content": "b"},
        {"role": "user", "content": "u"},
    ])
    assert system == ["a", "b"]


def test_openai_usage_mapping():
    u = _from_openai_usage({"prompt_tokens": 10, "completion_tokens": 5,
                            "prompt_tokens_details": {"cached_tokens": 4}})
    assert (u.input_tokens, u.output_tokens, u.cache_read_input_tokens) == (10, 5, 4)
    u2 = _from_openai_usage({"prompt_cache_hit_tokens": 7})
    assert u2.cache_read_input_tokens == 7
    assert _from_openai_usage(None).input_tokens == 0


class _FakeResponse:
    """Minimal stand-in for urllib's HTTPResponse (context manager + read)."""

    def __init__(self, body: dict, status: int = 200, headers: dict | None = None) -> None:
        self._bytes = json.dumps(body).encode("utf-8")
        self.status = status
        self.code = status
        self.headers = headers or {}
        self._off = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def read(self, amt: int | None = None) -> bytes:
        out = self._bytes[self._off:]
        self._off = len(self._bytes)
        return out


def test_openai_compat_turn_parses_tool_calls():
    body = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "write_file", "arguments": '{"path":"a.py"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }
    with mock.patch("urllib.request.urlopen", return_value=_FakeResponse(body)) as urlopen:
        t = OpenAICompatTransport(model="m", base_url="https://example.com/v1", api_key="k")
        result = t.turn(
            [{"role": "user", "content": "hi"}],
            _function_tools([{"name": "f", "description": "d",
                              "input_schema": {"type": "object"}}]),
        )
    req = urlopen.call_args.args[0]
    sent = json.loads(req.data)
    assert sent["model"] == "m"
    assert sent["tools"][0]["function"]["name"] == "f"
    assert req.headers["Authorization"] == "Bearer k"
    assert result.is_tool
    assert result.tool_calls[0].name == "write_file"
    assert result.tool_calls[0].inputs == {"path": "a.py"}
    assert result.stop_reason == "tool_calls"


def test_openai_compat_turn_handles_401():
    err = urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)
    err.fp = io.BytesIO(b'{"error": "bad key"}')
    err.headers = {}
    t = OpenAICompatTransport(model="m", base_url="https://example.com/v1", api_key="k")
    with mock.patch("urllib.request.urlopen", side_effect=err):
        try:
            t.turn([{"role": "user", "content": "hi"}], [])
        except TransportError as exc:
            assert "authentication" in str(exc)
        else:
            raise AssertionError("expected TransportError")


def _http_error(code: int, headers=None):
    err = urllib.error.HTTPError("url", code, "error", headers or {}, None)
    err.fp = io.BytesIO(b'{"error": "x"}')
    err.headers = headers or {}
    return err


def test_openai_compat_error_classification():
    t = OpenAICompatTransport(model="m", base_url="https://example.com/v1", api_key="k")
    with mock.patch("urllib.request.urlopen", side_effect=_http_error(400)):
        try:
            t.turn([{"role": "user", "content": "hi"}], [])
        except RateLimitError:
            raise AssertionError("400 is not a rate limit")
        except RetryableTransportError:
            raise AssertionError("400 is not retryable")
        except TransportError as exc:
            assert "400" in str(exc)
        else:
            raise AssertionError("expected TransportError")
    # a 429 for an over-budget max_tokens must be a retryable RateLimitError
    with mock.patch("urllib.request.urlopen",
                    side_effect=_http_error(429, '{"error": "rate"}')) as urlopen:
        try:
            t.turn([{"role": "user", "content": "hi"}], [])
        except RateLimitError as exc:
            assert exc.retry_after is None
        else:
            raise AssertionError("expected RateLimitError")
    with mock.patch("urllib.request.urlopen", side_effect=_http_error(503)):
        try:
            t.turn([{"role": "user", "content": "hi"}], [])
        except RetryableTransportError:
            pass
        else:
            raise AssertionError("expected RetryableTransportError for 5xx")
    assert len(urlopen.call_args_list) > 0


def test_max_tokens_default_is_provider_safe():
    t = OpenAICompatTransport(model="gpt-4o-mini")
    assert t.max_tokens == 16384


def test_max_tokens_is_sent_in_request_and_rate_headers_returned():
    body = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    }
    headers = {"x-ratelimit-limit-requests": "120",
               "x-ratelimit-requests-remaining": "118"}
    with mock.patch("urllib.request.urlopen",
                    return_value=_FakeResponse(body, headers=headers)) as urlopen:
        t = OpenAICompatTransport(model="m", base_url="https://example.com/v1",
                                  api_key="k", max_tokens=2048)
        result = t.turn([{"role": "user", "content": "hi"}], [])
    sent = json.loads(urlopen.call_args.args[0].data)
    assert sent["max_tokens"] == 2048
    assert result.rate_rpm == 120.0


def test_rate_limit_header_parsing():
    class _Headers:
        """Case-insensitive header map, like urllib's HTTPMessage."""
        def __init__(self, data):
            self._d = {k.lower(): v for k, v in data.items()}
        def get(self, key, default=None):
            return self._d.get(key.lower(), default)
    assert _rate_limit_from(_Headers({"X-Ratelimit-Limit-Requests": "60"})) == 60.0
    assert _rate_limit_from({"x-ratelimit-requests-limit": "30"}) == 30.0
    assert _rate_limit_from(None) is None
    assert _rate_limit_from({"x-ratelimit-requests-limit": "bogus"}) is None