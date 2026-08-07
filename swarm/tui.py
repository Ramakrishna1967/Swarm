"""Live dashboard (§2.7), a *consumer* of the event log.

There is no IPC and no coupling to the orchestrator: the TUI reads the
fsync'd append-only log and folds events into its own view, exactly like
``swarm status`` does in a second terminal.

``render`` is a pure function of a folded status dict, so it is unit-testable
with no terminal. The dashboard glyphs carry the meaning so the output also
works piped and on monochrome terminals. When stdout is not a TTY (or
``--no-tui``) we fall back to line-oriented output.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from .events import EventBus, fold_status

GLYPH = {
    "PENDING": "○", "PROVISIONING": "◐", "RUNNING": "●", "SETTLING": "◐",
    "THROTTLED": "⏸", "EXHAUSTED": "⚠", "FAILED": "✖", "CANCELLING": "■",
    "ABORTED": "■", "DONE": "✔",
}
GLYPH_ASCII = {
    "PENDING": "o", "PROVISIONING": ">", "RUNNING": "*", "SETTLING": ">",
    "THROTTLED": "~", "EXHAUSTED": "!", "FAILED": "x", "CANCELLING": "#",
    "ABORTED": "#", "DONE": "+",
}


def _glyph_set(encoding: str | None = None) -> dict:
    # Glyphs are redundant with the state word, so degrade to ASCII when the
    # terminal cannot represent the box-drawing set (e.g. Windows cp1252).
    enc = encoding or getattr(sys.stdout, "encoding", "") or ""
    if enc.lower() in {"utf-8", "utf8"}:
        return GLYPH
    return GLYPH_ASCII


def render(status: dict, *, encoding: str | None = None) -> str:
    """Render a folded status dict to a plain-text dashboard."""
    glyphs = _glyph_set(encoding)
    agents: dict = status.get("agents", {})
    conflicts: list = status.get("conflicts", [])
    summary: dict = status.get("summary", {})
    out: list[str] = []
    out.append("agents")
    if not agents:
        out.append("  (no agents yet)")
    for name in sorted(agents):
        a = agents[name]
        st = a.get("state", "") or "PENDING"
        glyph = glyphs.get(st, ".")
        extras = []
        if a.get("tool"):
            extras.append(a["tool"])
        if a.get("files"):
            extras.append(f"{a['files']} files")
        detail = ("  " + " ".join(extras)) if extras else ""
        out.append(f"  {glyph} {name:<12} {st:<12} turn {a.get('turn', 0)}{detail}")
    out.append("conflicts")
    for c in conflicts:
        kind = c.get("conflict_kind", c.get("kind", "?"))
        sev = c.get("severity", "medium")
        sym = c.get("symbol", "")
        party = f"  by {c.get('definer','-')} -> {c.get('caller','-')}"
        out.append(f"  [{sev}] {kind}  {sym}{party}")
    if not conflicts:
        out.append("  (none detected)")
    out.append(
        f"tokens  in {summary.get('tokens_in', 0)}  out {summary.get('tokens_out', 0)}"
        f"  cache_read {summary.get('cache_read', 0)}"
    )
    return "\n".join(out)


def watch(
    event_log: Path,
    *,
    interval: float = 0.25,
    max_seconds: float | None = None,
    stop_when: dict[str, Any] | None = None,
) -> dict:
    """Poll the event log, (re)printing the status on change.

    Returns the final folded status. ``stop_when`` lets tests end early, e.g.
    ``{"DONE": True}`` keyed on a folded field.
    """
    bus = EventBus(event_log)
    start = time.monotonic()
    last = ""
    while True:
        folded = fold_status(bus.read())
        text = render(folded)
        if text != last:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
            last = text
        if max_seconds is not None and time.monotonic() - start > max_seconds:
            break
        if stop_when and _satisfies(folded, stop_when):
            break
        time.sleep(interval)
    return fold_status(bus.read())


def _satisfies(folded: dict, want: dict) -> bool:
    for key, val in want.items():
        if folded.get(key) != val:
            return False
    return True