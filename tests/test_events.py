"""EventBus JSONL + fold_status projection (§2.7)."""

from __future__ import annotations

import json

from swarm.events import EventBus, fold_status


def test_emit_and_read_roundtrip(tmp_path):
    bus = EventBus(tmp_path / "e.jsonl")
    bus.emit("agent_state", agent="A", state="RUNNING", turn=1)
    bus.emit("conflict", conflict_kind="broken_caller", symbol="m.f")
    evs = bus.read()
    assert len(evs) == 2
    assert evs[0]["agent"] == "A"
    assert evs[1]["symbol"] == "m.f"


def test_render_events_dropped_under_pressure(tmp_path):
    bus = EventBus(tmp_path / "e.jsonl", max_render_buffer=2)
    for i in range(10):
        bus.emit("render_foo", i=i)
    assert bus.read() == []
    assert bus.dropped == 8
    assert len(bus.render_buffer) == 2


def test_fold_status_projection(tmp_path):
    bus = EventBus(tmp_path / "e.jsonl")
    bus.emit("run_started", run_id="r1", base="abc")
    bus.emit("agent_state", agent="A", state="RUNNING", turn=3)
    bus.emit("model_usage", agent="A", input_tokens=10, output_tokens=4,
             cache_read_input_tokens=2)
    bus.emit("conflict", conflict_kind="broken_caller", symbol="m.f", severity="high")
    st = fold_status(bus.read())
    assert st["agents"]["A"]["state"] == "RUNNING"
    assert st["summary"]["tokens_in"] == 10
    assert st["summary"]["cache_read"] == 2
    assert len(st["conflicts"]) == 1
    assert st["summary"]["base"] == "abc"


def test_jsonl_lines_are_parseable(tmp_path):
    bus = EventBus(tmp_path / "e.jsonl")
    for i in range(5):
        bus.emit("agent_state", agent="A", state="RUNNING", turn=i)
    for line in (tmp_path / "e.jsonl").read_text(encoding="utf-8").splitlines():
        json.loads(line)
