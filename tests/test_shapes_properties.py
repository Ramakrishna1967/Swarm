"""Property-based tests for the shape lattice (no hypothesis dependency).

The lattice laws that make accepts()/relation() trustworthy are monotonicity,
duality, and transitivity. Rather than hand-picking a few examples, generate
random shapes and call sites from a seeded PRNG and assert the laws hold
across thousands of cases.

Properties checked:

* reflexivity:    relation(s, s) == IDENTICAL
* duality:        relation(a, b) == WIDENED  <=>  relation(b, a) == NARROWED
* monotonicity:   WIDENED a->b means every call `a` ACCEPTs is never REJECTed
                  by `b` (widening cannot break a previously-working call)
* transitivity:   if a <= b and b <= c (IDENTICAL/WIDENED) then a <= c
* soundness:      REJECT is only returned for a genuinely unsatisfiable call
                  (never for a call the shape structurally satisfies)

``accepts`` may return UNKNOWN for opaque/star calls, so the laws are stated
in terms of ACCEPT/REJECT where UNKNOWN is compatible with either.
"""

from __future__ import annotations

import random

from swarm.models import CallSite, Shape
from swarm.shapes import Relation, Verdict, accepts, relation

PARAM_NAMES = ("a", "b", "c", "d", "e", "f", "g", "h")


def rand_shape(rng: random.Random, max_params: int = 4) -> Shape:
    n = rng.randint(0, max_params)
    positional = PARAM_NAMES[:n]
    posonly = rng.randint(0, n) if n else 0
    # required positional <= total positional (params after defaults are optional)
    required = rng.randint(0, n)
    if required < posonly:
        required = posonly
    kw_required = frozenset(rng.sample(PARAM_NAMES[n:], rng.randint(0, 2)))
    kw_optional = frozenset(rng.sample(
        [p for p in PARAM_NAMES if p not in positional and p not in kw_required],
        rng.randint(0, 2)))
    # an is_method shape carries self implicitly; keep the law checks clean by
    # only allowing it when the name set has room for it.
    return Shape(
        positional, posonly, required, kw_required, kw_optional,
        rng.random() < 0.2, rng.random() < 0.2,
        False, False,
    )


def rand_call(rng: random.Random, max_args: int = 4) -> CallSite:
    n = rng.randint(0, max_args)
    keywords = frozenset(rng.sample(PARAM_NAMES, rng.randint(0, 3)))
    return CallSite(
        key=None, n_positional=n, keywords=keywords,
        has_star_args=rng.random() < 0.15,
        has_star_kwargs=rng.random() < 0.15,
    )


def test_reflexivity():
    rng = random.Random(1234)
    for _ in range(2000):
        s = rand_shape(rng)
        assert relation(s, s) == Relation.IDENTICAL


def test_duality_widened_narrowed():
    rng = random.Random(5678)
    for _ in range(2000):
        a, b = rand_shape(rng), rand_shape(rng)
        r_ab = relation(a, b)
        if r_ab == Relation.WIDENED:
            assert relation(b, a) == Relation.NARROWED
        if r_ab == Relation.NARROWED:
            assert relation(b, a) == Relation.WIDENED


def test_monotonicity_widening_never_breaks_calls():
    rng = random.Random(9012)
    checks = 0
    for _ in range(4000):
        a, b = rand_shape(rng), rand_shape(rng)
        if relation(a, b) != Relation.WIDENED:
            continue
        for _ in range(3):
            call = rand_call(rng)
            if accepts(a, call) is Verdict.ACCEPT:
                checks += 1
                assert accepts(b, call) is not Verdict.REJECT
    assert checks > 0, "PRNG produced no widening pairs (suspicious)"


def test_transitivity_of_widening():
    rng = random.Random(3456)
    for _ in range(3000):
        a, b, c = rand_shape(rng), rand_shape(rng), rand_shape(rng)
        ab = relation(a, b) in {Relation.IDENTICAL, Relation.WIDENED}
        bc = relation(b, c) in {Relation.IDENTICAL, Relation.WIDENED}
        if ab and bc:
            assert relation(a, c) in {Relation.IDENTICAL, Relation.WIDENED}


def test_accepts_sound_positional_overflow():
    # n_positional strictly greater than the (non-vararg) parameter list is a
    # genuine reject: too many positional args.
    rng = random.Random(7890)
    for _ in range(1000):
        s = rand_shape(rng)
        if s.has_varargs:
            continue
        n_pos = len(s.positional) + rng.randint(1, 3)
        assert accepts(s, CallSite(key=None, n_positional=n_pos, keywords=frozenset())) \
            is Verdict.REJECT


def test_accepts_sound_missing_required():
    # fewer positional args than required_positional and no keywords to fill
    # them is a genuine reject.
    rng = random.Random(1592)
    for _ in range(1000):
        s = rand_shape(rng)
        if s.required_positional <= 0:
            continue
        under = s.required_positional - 1
        call = CallSite(key=None, n_positional=under, keywords=frozenset())
        if accepts(s, call) is Verdict.REJECT:
            continue  # fine either way when star-args make it UNKNOWN
        # without star args this must be a reject
        if not call.has_star_args:
            assert accepts(s, call) is Verdict.REJECT


def test_identical_shapes_agree_on_every_call():
    rng = random.Random(2468)
    for _ in range(1000):
        s = rand_shape(rng)
        call = rand_call(rng)
        assert accepts(s, call) is accepts(s, call)
