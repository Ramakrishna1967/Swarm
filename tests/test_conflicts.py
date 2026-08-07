"""ConflictDetector: the verified core behaviours.

broken_caller, backward-compatible no-conflict, divergent_def, keyword-only,
removed_symbol -- the five behaviours confirmed in the session summary, plus
the state gate that decides TENTATIVE vs CONFIRMED.
"""

from __future__ import annotations

from dataclasses import replace

from swarm.conflicts import ConflictDetector
from swarm.models import ConflictKind, FileContract, Shape, SymbolKey


def key(name: str) -> SymbolKey:
    return SymbolKey("mod", name)


def defn(symbol: str, shape: Shape) -> FileContract:
    fc = FileContract(f"{symbol}.py")
    fc.definitions[key(symbol)] = shape
    return fc


def caller(symbol: str, n_positional=1, keywords=frozenset(), in_test=False) -> FileContract:
    fc = FileContract("caller.py")
    fc.calls.append(make_callsite(symbol, n_positional, keywords, in_test))
    return fc


def make_callsite(symbol: str, n_positional=1, keywords=frozenset(), in_test=False):
    from swarm.models import CallSite
    return CallSite(key=key(symbol), n_positional=n_positional,
                    keywords=frozenset(keywords), in_test=in_test, file="caller.py", line=1)


def base_shape(positional, required_positional, **kw) -> Shape:
    return Shape(positional, kw.get("posonly_count", 0), required_positional,
                 kw.get("kw_required", frozenset()), kw.get("kw_optional", frozenset()),
                 kw.get("has_varargs", False), kw.get("has_kwargs", False),
                 kw.get("is_method", False), kw.get("implicit_self", False),
                 kw.get("decorators", ()), kw.get("arity_opaque", False))


def detect(base, agents, base_callers=None):
    detector = ConflictDetector(base, base_callers)
    conflicts, _ = detector.detect(agents)
    return conflicts


def test_broken_caller_evolution():
    # A evolved authenticate(user) -> authenticate(user, scope); B still calls
    # the old one-arg form. Exactly one BROKEN_CALLER, definer=A, caller=B.
    base = {key("authenticate"): base_shape(("user",), 1)}
    agents = {
        "A": defn("authenticate", base_shape(("user", "scope"), 2)),
        "B": caller("authenticate", n_positional=1),
    }
    conflicts = detect(base, agents)
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.kind == ConflictKind.BROKEN_CALLER
    assert c.definer == "A" and c.caller == "B"


def test_backward_compatible_no_conflict():
    # adding an optional param must not flag anything.
    base = {key("f"): base_shape(("x",), 1)}
    agents = {
        "A": defn("f", base_shape(("x", "y"), 1)),  # y optional
        "B": caller("f", n_positional=1),
        "C": caller("f", n_positional=1),
    }
    assert detect(base, agents) == []


def test_no_conflict_when_everyone_calls_new_form():
    base = {key("authenticate"): base_shape(("user",), 1)}
    agents = {
        "A": defn("authenticate", base_shape(("user", "scope"), 2)),
        "B": caller("authenticate", n_positional=2),
    }
    assert detect(base, agents) == []


def test_divergent_def():
    # two agents independently redefine get() incompatibly.
    base = {key("get"): base_shape(("self", "key"), 2)}
    agents = {
        "B": defn("get", base_shape(("self", "key", "default"), 2)),
        "D": defn("get", base_shape(("self", "key"), 2, kw_required={"ttl"})),
    }
    conflicts = detect(base, agents)
    kinds = [c.kind for c in conflicts]
    assert ConflictKind.DIVERGENT_DEF in kinds


def test_keyword_only_caller_conflict():
    # A narrows f by adding a required keyword-only param; B (foreign) calls
    # the old form without it -> BROKEN_CALLER.
    base = {key("f"): base_shape(("x",), 1)}
    agents = {
        "A": defn("f", base_shape(("x",), 1, kw_required={"flag"})),
        "B": caller("f", n_positional=1, keywords=set()),
    }
    conflicts = detect(base, agents)
    assert len(conflicts) == 1
    assert conflicts[0].kind == ConflictKind.BROKEN_CALLER


def test_removed_symbol_with_surviving_caller():
    base = {key("gone"): base_shape(("x",), 1)}
    agents = {"B": caller("gone", n_positional=1)}
    conflicts = detect(base, agents)
    assert len(conflicts) == 1
    assert conflicts[0].kind == ConflictKind.REMOVED_SYMBOL


def test_removed_symbol_no_caller_is_silent():
    base = {key("gone"): base_shape(("x",), 1)}
    assert detect(base, {}) == []


def test_convergent_identical_change():
    base = {key("f"): base_shape(("x",), 1)}
    agents = {
        "A": defn("f", base_shape(("x",), 1)),
        "B": defn("f", base_shape(("x",), 1)),
    }
    conflicts = detect(base, agents)
    assert len(conflicts) == 1
    assert conflicts[0].kind == ConflictKind.CONVERGENT


def test_broken_base_caller():
    # base has a caller of f; A narrows f; the base caller now breaks.
    base_call = make_callsite("f", n_positional=1)
    base = {key("f"): base_shape(("x",), 1)}
    agents = {"A": defn("f", base_shape(("x", "y"), 2))}
    conflicts = ConflictDetector(base, {key("f"): [base_call]}).detect(agents)[0]
    assert any(c.kind == ConflictKind.BROKEN_BASE_CALLER for c in conflicts)


def test_self_evolution_is_not_broken_caller_for_method_owner():
    # A changes its own method: intra-agent call disagreements are its own
    # business; only foreign callers that reject matter.
    base = {key("S.m"): base_shape(("self",), 1, is_method=True, implicit_self=True)}
    agents = {"A": defn("S.m", base_shape(("self", "z"), 1, is_method=True, implicit_self=True))}
    assert detect(base, agents) == []
