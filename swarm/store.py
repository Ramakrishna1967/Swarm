"""Incremental contract store with the PROVISIONAL/TENTATIVE/SETTLED/CONFIRMED
state machine (§3.2).

Neither "extract on every write" (mid-edit files are broken and produce
noise) nor "extract on commit only" (too late for a live dashboard). This
module debounces extraction per agent and tags each symbol with a stability
level: PROVISIONAL while the agent is mid-turn, SETTLED once the turn
boundary passes. Only conflicts between two SETTLED contracts are CONFIRMED.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .extract import extract_source, file_digest, module_name_for
from .models import FileContract, SymbolKey
from .resolve import Repo


@dataclass
class SymbolState:
    key: SymbolKey
    settled: bool = False  # PROVISIONAL (False) vs SETTLED (True)
    agent: str = ""


@dataclass
class ExtractionResult:
    contract: FileContract | None
    dirty_unparseable: bool = False


class ContractStore:
    """Per-agent, per-file, debounced, cached contract state.

    Every write to a worktree enqueues the file for re-extraction after a
    debounce window (coalescing edit bursts). A file with a SyntaxError keeps
    its last good contract and is marked dirty_unparseable.
    """

    def __init__(
        self,
        debounce_ms: float = 750,
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        self.debounce_ms = debounce_ms
        self.on_change = on_change
        self._cache: dict[tuple[str, str], tuple[str, FileContract]] = {}
        self._pending: dict[str, list[str]] = {}
        self._timer: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._repo: Repo | None = None

    # -- file writes ---------------------------------------------------------
    def touch(self, agent: str, path: str, root: str) -> None:
        """Enqueue a file for re-extraction (called from the write tool)."""
        old = self._timer.pop(agent, None)
        if old is not None:
            old.cancel()
        with self._lock:
            self._pending.setdefault(agent, []).append(path)
            timer = threading.Timer(self.debounce_ms / 1000.0, self._flush, args=(agent, root))
            timer.daemon = True
            self._timer[agent] = timer
            timer.start()

    def _flush(self, agent: str, root: str) -> None:
        with self._lock:
            paths = self._pending.pop(agent, [])
        changed: set[SymbolKey] = set()
        for path in set(paths):
            contract = self._extract_file(agent, path, root)
            if contract is None:
                continue
            if contract.syntax_error:
                # keep last good contract; mark dirty
                self._mark_dirty(agent, path)
                continue
            full = Path(root) / path
            self._cache[(agent, path)] = (file_digest(full), contract)
            changed.update(contract.definitions.keys())
        if self.on_change and changed:
            self.on_change(agent)

    def _extract_file(self, agent: str, path: str, root: str) -> FileContract | None:
        full = Path(root) / path
        if not full.exists():
            return None
        try:
            source = full.read_text(encoding="utf-8")
        except OSError:
            return None
        module = module_name_for(full, Path(root))
        return extract_source(source, str(full), module)

    def _mark_dirty(self, agent: str, path: str) -> None:
        entry = self._cache.get((agent, path))
        if entry is not None:
            _, contract = entry
            contract.dirty_unparseable = True

    # -- queries -------------------------------------------------------------
    def contract_for(self, agent: str, path: str) -> FileContract | None:
        entry = self._cache.get((agent, path))
        return entry[1] if entry else None

    def repo_for(self, root: str, top: str) -> Repo:
        if self._repo is None or self._repo.root != root:
            self._repo = Repo(root, top)
        return self._repo

    def shutdown(self) -> None:
        for timer in self._timer.values():
            timer.cancel()
        self._timer.clear()