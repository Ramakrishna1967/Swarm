"""Shape compatibility: the directional predicate and the relation lattice.

Implements §4.1 of SWARM_ARCHITECTURE.md. The primitive is not arity
matching but "does this shape accept this call?" (accepts) and, over
definitions, "is this a widening, narrowing, renamed-compatible, or
diverged change?" (relation).
"""

from __future__ import annotations

from .models import CallSite, Relation, Shape, Verdict


# Decorators that are arity-transparent: they do not change how calls bind,
# so call-site checks against the decorated function stay valid.
ARITY_TRANSPARENT = {
    "staticmethod", "classmethod", "property", "abstractmethod",
    "override", "overload", "functools.wraps", "functools.lru_cache",
    "typing.overload",
}


def accepts(shape: Shape, call: CallSite) -> Verdict:
    """Does this shape accept this call? Bias toward silence (UNKNOWN)."""
    if call.has_star_args or call.has_star_kwargs or shape.arity_opaque:
        return Verdict.UNKNOWN
    slots = list(shape.positional)
    if shape.implicit_self:
        slots = slots[1:]
    if call.n_positional > len(slots) and not shape.has_varargs:
        return Verdict.REJECT
    consumed = set(slots[:call.n_positional])
    # Nameable params: positional-or-keyword (beyond the posonly boundary)
    # plus every keyword-only param.
    nameable = set(slots[shape.posonly_count:]) | set(shape.kwonly_required) | set(shape.kwonly_optional)
    for keyword in call.keywords:
        if keyword in consumed or (keyword not in nameable and not shape.has_kwargs):
            return Verdict.REJECT
    if set(slots[:shape.required_positional]) - consumed - set(call.keywords):
        return Verdict.REJECT
    if shape.kwonly_required - set(call.keywords):
        return Verdict.REJECT
    return Verdict.ACCEPT


def _compatible(old: Shape, new: Shape) -> bool:
    """Does `new` accept every call that `old` accepts? (structural, §4.1)

    This is a *sound* superset check: WIDENED is only reported when the new
    shape really accepts every call the old shape did. A call is a
    (n_positional, keywords) pair, so `new` must not shrink the positional
    surface, must not reject any keyword `old` callers could pass, and must
    not add a requirement `old` callers were not forced to satisfy.

    Renames that keep the position are handled separately by `relation`
    (RENAMED_COMPATIBLE); here a renamed required parameter is *not* a
    compatible change because old callers may pass it by keyword.
    """
    old_slots = list(old.positional)
    if old.implicit_self:
        old_slots = old_slots[1:]
    new_slots = list(new.positional)
    if new.implicit_self:
        new_slots = new_slots[1:]

    old_nameable = set(old_slots[old.posonly_count:]) | set(old.kwonly_required) | set(old.kwonly_optional)
    new_nameable = set(new_slots[new.posonly_count:]) | set(new.kwonly_required) | set(new.kwonly_optional)

    # 1. Positional surface must not shrink: new cannot drop params old had
    #    (unless new got varargs, which only widens).
    if old.has_varargs and not new.has_varargs:
        return False
    if not new.has_varargs and len(new_slots) < len(old_slots):
        return False

    # 2. Every keyword an old caller may pass must be passable to new.
    if old.has_kwargs and not new.has_kwargs:
        return False
    if not new.has_kwargs:
        for k in old_nameable:
            if k not in new_nameable:
                return False

    # 3. A name old callers may pass as a keyword must not become a
    #    positional-only param of new (new would reject the keyword).
    for k in old_nameable:
        if k in new_slots[:new.posonly_count]:
            return False

    # 4. Requirements must be non-increasing, and new's required positional
    #    params must be a subset of old's required set (old callers are only
    #    forced to fill old's required params).
    if new.required_positional > old.required_positional:
        return False
    if not new.has_varargs:
        new_req = set(new_slots[:new.required_positional])
        old_req = set(old_slots[:old.required_positional])
        if not new_req <= old_req:
            return False

    # 5. New's required keyword-only params must be ones old callers are
    #    already forced to pass.
    if not new.kwonly_required <= old.kwonly_required:
        return False

    return True


def _accepts_old_calls(old: Shape, new: Shape) -> bool:
    return _compatible(old, new)


def _rename_compatible(old: Shape, new: Shape) -> bool:
    """Position-preserving rename: same arity structure, names may differ.

    The design treats a rename that keeps every position as compatible even
    though a caller using the old *name* as a keyword would break -- the
    detector still checks callers via ``accepts``, so this leniency can never
    suppress a real caller conflict. The leniency is only claimed when the
    arity structure (counts, requirements, kw surface) is identical.
    """
    old_slots = list(old.positional)
    if old.implicit_self:
        old_slots = old_slots[1:]
    new_slots = list(new.positional)
    if new.implicit_self:
        new_slots = new_slots[1:]
    if len(old_slots) != len(new_slots):
        return False
    if old.required_positional != new.required_positional:
        return False
    if old.posonly_count != new.posonly_count:
        return False
    if old.has_varargs != new.has_varargs or old.has_kwargs != new.has_kwargs:
        return False
    if set(old.kwonly_required) != set(new.kwonly_required):
        return False
    if set(old.kwonly_optional) != set(new.kwonly_optional):
        return False
    return True


def relation(old: Shape, new: Shape) -> Relation:
    """Structural relation of two shapes, computed in both directions.

    IDENTICAL / WIDENED / NARROWED / RENAMED_COMPATIBLE / DIVERGED.
    """
    if old == new:
        return Relation.IDENTICAL
    forward = _compatible(old, new)
    backward = _compatible(new, old)
    if forward and backward:
        if old.positional != new.positional:
            return Relation.RENAMED_COMPATIBLE
        return Relation.IDENTICAL
    if forward:
        return Relation.WIDENED
    if backward:
        return Relation.NARROWED
    if _rename_compatible(old, new):
        return Relation.RENAMED_COMPATIBLE
    return Relation.DIVERGED


def summarizes(left: Shape, right: Shape) -> bool:
    """Is there a total order such that one shape accepts every call the
    other does? (used by the SUBSUMABLE check in §4.2)."""
    return _accepts_old_calls(left, right) or _accepts_old_calls(right, left)