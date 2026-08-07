"""swarm: orchestrates parallel coding agents and detects interface conflicts.

Public surface mirrors the module layout in SWARM_ARCHITECTURE.md. Import the
requested module directly for fine-grained use (``from swarm import conflicts``
etc.); this package only lifts the version and a few headliner names so that
``import swarm`` is a no-op and ``python -m swarm`` works.
"""

from __future__ import annotations

from . import admission, conflicts, events, extract, gitops, merge, models, resolve, runner, shapes, store, transport

__version__ = "0.1.0"

__all__ = [
    "admission", "conflicts", "events", "extract", "gitops", "merge",
    "models", "resolve", "runner", "shapes", "store", "transport",
    "__version__",
]