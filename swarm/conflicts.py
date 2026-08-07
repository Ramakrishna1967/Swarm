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

from collections import defaultdict
from dataclasses import replace

from .models import CallSite, Conflict, ConflictKind, FileContract, Shape, SymbolKey, severity_for
from .shapes import Relation, Verdict, accepts, relation, summarizes


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


def _contributors(
    index: dict[SymbolKey, dict], key: SymbolKey,
) -> dict[str, list[CallSite]]:
    """Map every definer's own name/agent to the call sites it owns for key,
    plus an empty list for other agents that never call it."""
    sites = index[key]["callers"]
    result: dict[str, list[CallSite]] = {}
    for agent, calls in sites.items():
        result[agent] = calls
    for agent in index[key]["definers"]:
        result.setdefault(agent, [])
    return result


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
                for call in (self.base_callers.get(key, []) +
                             [c for g in entry["callers"].values() for c in g]):
                    conflicts.append(Conflict(
                        kind=ConflictKind.REMOVED_SYMBOL,
                        symbol=str(key), definer=None, caller=None,
                        detail="definition removed while callers remain",
                        severity="medium" if call.in_test else "high",
                        file=call.file, line=call.line,
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
                            conflicts.append(Conflict(
                                kind=ConflictKind.BROKEN_CALLER,
                                symbol=str(key), definer=owner, caller=agent,
                                detail="new definition rejects the call",
                                severity="medium" if call.in_test else "high",
                                file=call.file, line=call.line,
                            ))
                # pre-existing untouched base callers that now break
                if old is not None:
                    for call in self.base_callers.get(key, []):
                        if accepts(new, call) is Verdict.REJECT:
                            conflicts.append(Conflict(
                                kind=ConflictKind.BROKEN_BASE_CALLER,
                                symbol=str(key), definer=owner, caller=None,
                                detail="new definition rejects a base caller",
                                severity="medium",
                                file=call.file, line=call.line,
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
            if summarizes(shapes[0], shapes[1]):
                conflicts.append(Conflict(
                    kind=ConflictKind.SUBSUMABLE, symbol=str(key),
                    definer=", ".join(definers), caller=None,
                    detail="a total order exists; pick the most permissive",
                    severity="low",
                ))
            elif rels <= {Relation.IDENTICAL, Relation.WIDENED, Relation.NARROWED}:
                conflicts.append(Conflict(
                    kind=ConflictKind.SUBSUMABLE, symbol=str(key),
                    definer=", ".join(definers), caller=None,
                    detail="compatible direction; a total order exists",
                    severity="low",
                ))
            else:
                conflicts.append(Conflict(
                    kind=ConflictKind.DIVERGENT_DEF, symbol=str(key),
                    definer=", ".join(definers), caller=None,
                    detail="independent definitions diverge",
                    severity="high",
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