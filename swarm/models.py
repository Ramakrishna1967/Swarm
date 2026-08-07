"""Core data models for swarm's contract-aware parallel runner.

These mirror the dataclasses in SWARM_ARCHITECTURE.md (SymbolKey, Shape,
CallSite, FileContract) plus the agent lifecycle state machine used by the
orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, NewType


class Verdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    UNKNOWN = "unknown"


class Relation(str, Enum):
    IDENTICAL = "identical"
    WIDENED = "widened"
    NARROWED = "narrowed"
    RENAMED_COMPATIBLE = "renamed_compatible"
    DIVERGED = "diverged"


class ConflictKind(str, Enum):
    CONVERGENT = "convergent"
    SUBSUMABLE = "subsumable"
    BROKEN_CALLER = "broken_caller"
    BROKEN_BASE_CALLER = "broken_base_caller"
    DIVERGENT_DEF = "divergent_def"
    OVERRIDE_DIVERGENCE = "override_divergence"
    REMOVED_SYMBOL = "removed_symbol"
    RENAME_DISAGREEMENT = "rename_disagreement"


AgentState = Literal[
    "PENDING", "PROVISIONING", "RUNNING", "SETTLING", "THROTTLED",
    "EXHAUSTED", "FAILED", "CANCELLING", "ABORTED", "DONE",
]

# Terminal states (the lifecycle table in §3.1).
TERMINAL = {"DONE", "EXHAUSTED", "FAILED", "ABORTED"}
# States whose work is still mergeable.
MERGEABLE = {"DONE", "EXHAUSTED", "ABORTED"}


@dataclass(frozen=True)
class SymbolKey:
    module: str
    qualname: str

    def __str__(self) -> str:
        return f"{self.module}:{self.qualname}"


@dataclass(frozen=True)
class Shape:
    positional: tuple[str, ...]
    posonly_count: int
    required_positional: int
    kwonly_required: frozenset[str]
    kwonly_optional: frozenset[str]
    has_varargs: bool
    has_kwargs: bool
    is_method: bool
    implicit_self: bool
    decorators: tuple[str, ...] = ()
    arity_opaque: bool = False
    mro_incomplete: bool = False


@dataclass(frozen=True)
class CallSite:
    key: SymbolKey | None
    n_positional: int
    keywords: frozenset[str]
    has_star_args: bool = False
    has_star_kwargs: bool = False
    file: str = ""
    line: int = 0
    in_test: bool = False
    confidence: Literal["exact", "heuristic", "unresolved"] = "exact"
    reason: str | None = None
    attr: str | None = None


@dataclass
class FileContract:
    path: str
    definitions: dict[SymbolKey, Shape] = field(default_factory=dict)
    calls: list[CallSite] = field(default_factory=list)
    reexports: dict[str, SymbolKey] = field(default_factory=dict)
    syntax_error: str | None = None
    dirty_unparseable: bool = False


@dataclass(frozen=True)
class AbstractSymbol:
    """A symbol known to exist but whose shape we cannot extract."""
    key: SymbolKey
    reason: str


@dataclass(frozen=True)
class Conflict:
    kind: ConflictKind
    symbol: str
    definer: str | None
    caller: str | None
    detail: str
    severity: str = "high"
    file: str | None = None
    line: int | None = None
    a: str | None = None
    b: str | None = None


@dataclass(frozen=True)
class Repair:
    """A cross-agent contract injection sent as a peer system message."""
    target_agent: str
    symbol: str
    was: str
    now: str
    call_sites: list[tuple[str, int, str]]
    round: int


@dataclass
class RunResult:
    """Outcome of one Agent.run: the state the loop ended in plus a tally."""
    state: str = "PENDING"
    files: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0


@dataclass(frozen=True)
class Task:
    """One unit of work handed to one agent (§2.1)."""
    id: str
    title: str
    instructions: str
    expected_paths: tuple[str, ...] = ()


AgentId = NewType("AgentId", str)
RunId = NewType("RunId", str)
Sha = NewType("Sha", str)


def severity_for(kind: ConflictKind) -> str:
    return {
        ConflictKind.CONVERGENT: "low",
        ConflictKind.SUBSUMABLE: "low",
        ConflictKind.RENAME_DISAGREEMENT: "medium",
        ConflictKind.BROKEN_BASE_CALLER: "medium",
        ConflictKind.BROKEN_CALLER: "high",
        ConflictKind.DIVERGENT_DEF: "high",
        ConflictKind.OVERRIDE_DIVERGENCE: "high",
        ConflictKind.REMOVED_SYMBOL: "high",
    }.get(kind, "medium")