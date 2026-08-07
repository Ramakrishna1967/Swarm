"""Lattice laws for accepts() and relation() (§8 verification #2).

The two functions in shapes.py are the whole detection algorithm, so they get
the most test weight. Property tests are plain loops here (no hypothesis).
"""

from __future__ import annotations

from swarm.models import CallSite, Shape
from swarm.shapes import Relation, Verdict, accepts, relation


def s(
    positional=("x", "y"), posonly_count=0, required_positional=1,
    kw_required=frozenset(), kw_optional=frozenset(),
    has_varargs=False, has_kwargs=False, is_method=False, implicit_self=False,
    decorators=(), arity_opaque=False,
) -> Shape:
    return Shape(positional, posonly_count, required_positional,
                 kw_required, kw_optional, has_varargs, has_kwargs,
                 is_method, implicit_self, decorators, arity_opaque)


def c(n_positional=1, keywords=frozenset(), has_star_args=False,
      has_star_kwargs=False) -> CallSite:
    return CallSite(key=None, n_positional=n_positional, keywords=frozenset(keywords),
                    has_star_args=has_star_args, has_star_kwargs=has_star_kwargs)


def test_relation_reflexive():
    for shape in [s(), s(required_positional=0), s(has_varargs=True),
                  s(("a",), kw_optional={"b"}), s(("self", "x"), is_method=True,
                                                  implicit_self=True, required_positional=2)]:
        assert relation(shape, shape) == Relation.IDENTICAL


def test_widened_accepts_superset():
    # old requires one param, new makes it optional (adds a default): every
    # call the old accepted must still be accepted by the new.
    old = s(("x",), required_positional=1)
    new = s(("x",), required_positional=0)
    assert relation(old, new) == Relation.WIDENED
    assert relation(new, old) == Relation.NARROWED
    calls = [c(1)]
    for call in calls:
        assert accepts(old, call) is Verdict.ACCEPT
        assert accepts(new, call) is not Verdict.REJECT


def test_narrowed_rejects_old_calls():
    old = s(("x",), required_positional=0)
    new = s(("x",), required_positional=1)
    assert relation(old, new) == Relation.NARROWED
    assert accepts(old, c(0)) is Verdict.ACCEPT
    assert accepts(new, c(0)) is Verdict.REJECT


def test_add_optional_param_is_not_breaking():
    old = s(("x",), required_positional=1)
    new = s(("x", "y"), required_positional=1)  # y has default
    assert relation(old, new) == Relation.WIDENED
    assert accepts(new, c(1)) is Verdict.ACCEPT


def test_renamed_compatible():
    old = s(("x",), required_positional=1)
    new = s(("y",), required_positional=1)
    assert relation(old, new) == Relation.RENAMED_COMPATIBLE


def test_keyword_only_required():
    shp = s(("x",), required_positional=1, kw_required={"k"})
    assert accepts(shp, c(1, keywords={"k"})) is Verdict.ACCEPT
    assert accepts(shp, c(1, keywords=set())) is Verdict.REJECT
    assert accepts(shp, c(2, keywords={"k"})) is Verdict.REJECT  # too many positional


def test_positional_only_bind():
    shp = s(("x", "y"), posonly_count=1, required_positional=2)
    # x is posonly: cannot be passed as keyword
    assert accepts(shp, c(1, keywords={"x"})) is Verdict.REJECT
    assert accepts(shp, c(2)) is Verdict.ACCEPT


def test_self_binding():
    shp = s(("self", "value"), is_method=True, implicit_self=True, required_positional=2)
    # a bound call omits self: one argument fills `value`
    assert accepts(shp, c(1)) is Verdict.ACCEPT
    assert accepts(shp, c(1, keywords={"value"})) is Verdict.REJECT  # duplicate
    assert accepts(shp, c(0)) is Verdict.REJECT  # value still required


def test_varargs_are_lattice_tops():
    shp = s(("x",), has_varargs=True, required_positional=1)
    assert accepts(shp, c(50)) is Verdict.ACCEPT
    assert accepts(shp, c(1)) is Verdict.ACCEPT
    assert accepts(shp, c(0)) is Verdict.REJECT  # x still required


def test_kwargs_accept_any_keyword():
    shp = s(("x",), has_kwargs=True, required_positional=1)
    assert accepts(shp, c(1, keywords={"zzz": True, "qq": True})) is Verdict.ACCEPT


def test_star_args_and_kwargs_unknown_on_call_side():
    assert accepts(s(("x",), required_positional=1), c(0, has_star_args=True)) is Verdict.UNKNOWN
    assert accepts(s(("x",), required_positional=1), c(0, has_star_kwargs=True)) is Verdict.UNKNOWN


def test_arity_opaque_silences():
    shp = s(("x",), required_positional=1, arity_opaque=True)
    assert accepts(shp, c(50, keywords={"a": True})) is Verdict.UNKNOWN


def test_double_positional_keyword_rejected():
    shp = s(("x",), required_positional=1)
    assert accepts(shp, c(1, keywords={"x": True})) is Verdict.REJECT
