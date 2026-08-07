"""TUI render is a pure function of the folded status (§2.7)."""

from __future__ import annotations

import sys

from swarm import tui


def test_render_empty():
    out = tui.render({"agents": {}, "conflicts": [], "summary": {}})
    assert "no agents" in out
    assert "none detected" in out


def test_render_agents_and_conflicts():
    st = {
        "agents": {"A": {"state": "RUNNING", "turn": 4, "files": 2},
                   "B": {"state": "DONE", "turn": 9}},
        "conflicts": [{"conflict_kind": "broken_caller", "symbol": "m.f",
                       "definer": "A", "caller": "B", "severity": "high"}],
        "summary": {"tokens_in": 10, "tokens_out": 2, "cache_read": 1},
    }
    out = tui.render(st)
    assert "A" in out and "B" in out
    assert "broken_caller" in out and "m.f" in out
    assert "tokens" in out


def test_render_glyph_ascii_fallback():
    st = {"agents": {"A": {"state": "DONE"}}, "conflicts": [], "summary": {}}
    out = tui.render(st, encoding="cp1252")
    assert "+" in out  # ASCII done glyph