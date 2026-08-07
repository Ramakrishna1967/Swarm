"""Conflict detection: a three-way merge against the common ancestor.

Implements §2.6 and §4.2. Every agent branches from the same frozen base, so
for any pre-existing symbol a base shape is always available. Detection is
therefore *not* a pairwise diff of agent contracts (which could not tell
"one agent evolved the interface" from "two agents disagree") but a lookup
of each symbol's base shape against the agents that touch it.

Index shape:
    SymbolKey -> {definers: {agent: Shape}, callers: {agent: [CallSite]}}
"""

from __future__ import annotations

from .models import CallSite, Conflict, ConflictKind, FileContract, Shape, SymbolKey
from .shapes import Relation, Verdict, accepts, relation, summarizes


def _shape_str(shape: Shape) -> str:
    """A short human-readable signature for a Shape, §6.2-style
    (``authenticate`` -> ``(user, scope)``). Self/cls is elided; positional-
    only, *args / **kwargs and keyword-only params are marked."""
    slots = list(shape.positional)
    self_offset = 1 if (shape.implicit_self and slots and slots[0] in ("self", "cls")) else 0
    slots = slots[self_offset:]
    posonly = max(shape.posonly_count - self_offset, 0)

    params: list[str] = []
    for i, name in enumerate(slots):
        params.append(name)
        if posonly and i == posonly - 1:
            params.append("/")
    if shape.has_varargs:
        params.append("*args")
    kw = sorted(set(shape.kwonly_required) | set(shape.kwonly_optional))
    if kw:
        if not shape.has_varargs and params and params[-1] not in ("/", "*args"):
            params.append("*")
        params.extend(kw)
    if shape.has_kwargs:
        params.append("**kwargs")
    return f"({', '.join(params)})"


def build_index(
    agents: dict[str, FileContract],
    base: dict[SymbolKey, Shape],
) -> tuple[dict[SymbolKey, dict], set[SymbolKey]]:
    touched: set[SymbolKey] = set()
    for contract in agents.values():
        touched.update(contract.definitions.keys())
        for call in contract.calls:
            if call.key is not None:
                touched.add(call.key)

    index: dict[SymbolKey, dict] = {}
    for key in touched:
        index[key] = {"definers": {}, "callers": {}}
    for agent, contract in agents.items():
        for key, shape in contract.definitions.items():
            index[key]["definers"][agent] = shape
        for call in contract.calls:
            if call.key is not None:
                index[call.key]["callers"].setdefault(agent, []).append(call)
    return index, touched


def _was_now(old: Shape | None, new: Shape | None) -> tuple[str | None, str | None]:
    return (_shape_str(old) if old is not None else None,
            _shape_str(new) if new is not None else None)


class ConflictDetector:
    def __init__(
        self,
        base: dict[SymbolKey, Shape],
        base_callers: dict[SymbolKey, list[CallSite]] | None = None,
    ) -> None:
        self.base = base
        self.base_callers = base_callers or {}

    def detect(
        self,
        agents: dict[str, FileContract],
    ) -> tuple[list[Conflict], dict[SymbolKey, dict]]:
        index, touched = build_index(agents, self.base)
        conflicts: list[Conflict] = []
        for key in touched:
            entry = index[key]
            definers = entry["definers"]
            old = self.base.get(key)

            # --- nothing retained the symbol but callers remain ---------------
            if not definers and old is not None:
                was, _ = _was_now(old, None)
                for call in (self.base_callers.get(key, []) +
                             [c for g in entry["callers"].values() for c in g]):
                    conflicts.append(Conflict(
                        kind=ConflictKind.REMOVED_SYMBOL,
                        symbol=str(key), definer=None, caller=None,
                        detail="definition removed while callers remain",
                        severity="medium" if call.in_test else "high",
                        file=call.file, line=call.line,
                        a=was, b=None,
                    ))
                continue

            # --- symbol no longer defined anywhere: nothing to compare ---------
            if not definers:
                continue

            # --- Case 1: exactly one definer => evolution ---------------------
            if len(definers) == 1:
                owner = next(iter(definers))
                new = definers[owner]
                if old is not None and relation(old, new) in {
                    Relation.IDENTICAL, Relation.WIDENED,
                }:
                    continue
                # foreign callers that the new shape rejects
                for agent, sites in entry["callers"].items():
                    if agent == owner:
                        continue
                    for call in sites:
                        if accepts(new, call) is Verdict.REJECT:
                            was, now = _was_now(old, new)
                            conflicts.append(Conflict(
                                kind=ConflictKind.BROKEN_CALLER,
                                symbol=str(key), definer=owner, caller=agent,
                                detail="new definition rejects the call",
                                severity="medium" if call.in_test else "high",
                                file=call.file, line=call.line,
                                a=was, b=now,
                            ))
                # pre-existing untouched base callers that now break
                if old is not None:
                    for call in self.base_callers.get(key, []):
                        if accepts(new, call) is Verdict.REJECT:
                            was, now = _was_now(old, new)
                            conflicts.append(Conflict(
                                kind=ConflictKind.BROKEN_BASE_CALLER,
                                symbol=str(key), definer=owner, caller=None,
                                detail="new definition rejects a base caller",
                                severity="medium",
                                file=call.file, line=call.line,
                                a=was, b=now,
                            ))
                continue

            # --- Case 2: two or more definers => concurrent decision ----------
            shapes = list(definers.values())
            uniq: list[Shape] = []
            for s in shapes:
                if s not in uniq:
                    uniq.append(s)
            if len(uniq) == 1:
                conflicts.append(Conflict(
                    kind=ConflictKind.CONVERGENT, symbol=str(key),
                    definer=", ".join(definers), caller=None,
                    detail="identical independent change",
                    severity="low",
                ))
                continue
            rels = {relation(a, b) for a in shapes for b in shapes if a is not b}
            was, _ = _was_now(old, None)
            if len(uniq) >= 2:
                a, b = _shape_str(uniq[0]), _shape_str(uniq[1])
            else:
                a, b = was, _shape_str(uniq[0])
            if summarizes(shapes[0], shapes[1]):
                conflicts.append(Conflict(
                    kind=ConflictKind.SUBSUMABLE, symbol=str(key),
                    definer=", ".join(definers), caller=None,
                    detail="a total order exists; pick the most permissive",
                    severity="low",
                    a=a, b=b,
                ))
            elif rels <= {Relation.IDENTICAL, Relation.WIDENED, Relation.NARROWED}:
                conflicts.append(Conflict(
                    kind=ConflictKind.SUBSUMABLE, symbol=str(key),
                    definer=", ".join(definers), caller=None,
                    detail="compatible direction; a total order exists",
                    severity="low",
                    a=a, b=b,
                ))
            else:
                conflicts.append(Conflict(
                    kind=ConflictKind.DIVERGENT_DEF, symbol=str(key),
                    definer=", ".join(definers), caller=None,
                    detail="independent definitions diverge",
                    severity="high",
                    a=a, b=b,
                ))
        return conflicts, index


def detect(
    base: dict[SymbolKey, Shape],
    agents: dict[str, FileContract],
    base_calls: list[CallSite] | None = None,
) -> list[Conflict]:
    base_callers: dict[SymbolKey, list[CallSite]] = {}
    for call in base_calls or []:
        if call.key is not None:
            base_callers.setdefault(call.key, []).append(call)
    detector = ConflictDetector(base, base_callers)
    conflicts, _ = detector.detect(agents)
    return conflicts