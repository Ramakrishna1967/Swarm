"""Append-only event log and status projection (§2.7).

The orchestrator writes fsync'd JSONL; the TUI and `swarm status` are both
*consumers* that fold the same events into their own view. There is no IPC
and no shared memory. A run is fully replayable from its log.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


class EventBus:
    def __init__(self, path: Path, max_render_buffer: int = 1000) -> None:
        self.path = path
        # Render-only events can be dropped under pressure; never contract/conflict.
        self.dropped = 0
        self.render_buffer: list[str] = []
        self.max_render_buffer = max_render_buffer
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Serializes appends: agents emit from their own worker threads, and a
        # torn line is worse than a lost telemetry event.
        self._lock = threading.Lock()

    def emit(self, kind: str, **data: object) -> None:
        record = {"time": time.time(), "kind": kind, **data}
        line = json.dumps(record, sort_keys=True) + "\n"
        if kind.startswith("render_"):
            self.render_buffer.append(line)
            if len(self.render_buffer) > self.max_render_buffer:
                self.render_buffer.pop(0)
                self.dropped += 1
            return
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())

    def read(self, offset: int = 0) -> list[dict]:
        events: list[dict] = []
        if not self.path.exists():
            return events
        with self.path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i < offset:
                    continue
                if line.strip():
                    events.append(json.loads(line))
        return events


def fold_status(events: list[dict]) -> dict:
    """Project the event list into a run status dict."""
    agents: dict[str, dict] = {}
    conflicts: list[dict] = []
    summary = {"tokens_in": 0, "tokens_out": 0, "cache_read": 0}
    for ev in events:
        kind = ev["kind"]
        if kind == "agent_state":
            agents[ev["agent"]] = {
                "state": ev["state"], "turn": ev.get("turn", 0),
                "tool": ev.get("tool"), "files": ev.get("files", 0),
            }
        elif kind == "conflict":
            conflicts.append(ev)
        elif kind in ("usage", "model_usage"):
            summary["tokens_in"] += ev.get("input_tokens", 0)
            summary["tokens_out"] += ev.get("output_tokens", 0)
            summary["cache_read"] += ev.get("cache_read_input_tokens", 0)
        elif kind == "run_started":
            summary.setdefault("base", ev.get("base"))
            summary.setdefault("run_id", ev.get("run_id"))
    return {
        "agents": agents,
        "conflicts": conflicts,
        "summary": summary,
        "event_count": len(events),
    }