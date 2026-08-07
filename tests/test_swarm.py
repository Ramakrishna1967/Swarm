"""Smoke tests preserved from the original root-level suite.

They use the canonical module paths (``swarm.shapes``, ``swarm.conflicts``,
``swarm.extract``) like the rest of the test suite; the old file imported
names from the package top level, which `swarm/__init__.py` never re-exported
and so never even collected.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from swarm.conflicts import detect
from swarm.extract import extract_source
from swarm.models import Relation, Verdict
from swarm.shapes import accepts, relation


class SwarmTests(unittest.TestCase):
    def test_optional_parameter_widens(self):
        old = extract_source("def f(user): pass").definitions.popitem()[1]
        new = extract_source("def f(user, scope=None): pass").definitions.popitem()[1]
        self.assertEqual(relation(old, new), Relation.WIDENED)

    def test_required_parameter_breaks_foreign_caller(self):
        base = extract_source("def authenticate(user): pass", "auth.py", "auth").definitions
        changed = extract_source("def authenticate(user, scope): pass", "auth.py", "auth")
        caller = extract_source("from auth import authenticate\nauthenticate(user)", "views.py", "views")
        self.assertEqual(accepts(next(iter(changed.definitions.values())), caller.calls[0]), Verdict.REJECT)
        findings = detect(base, {"A": changed, "B": caller})
        self.assertEqual([f.kind for f in findings], ["broken_caller"])

    def test_keyword_only_call_accepted(self):
        definition = extract_source("def get(key, *, ttl): pass", "cache.py", "cache")
        shape = next(iter(definition.definitions.values()))
        self.assertEqual(accepts(shape, extract_source("get(k, ttl=60)", "use.py", "use").calls[0]), Verdict.ACCEPT)
        self.assertNotEqual(accepts(shape, extract_source("get(k)", "u.py", "u").calls[0]), Verdict.ACCEPT)

    def test_unresolved_call_is_not_reported(self):
        base = extract_source("def f(x): pass", "x.py", "x").definitions
        changed = extract_source("def f(x, y): pass", "x.py", "x")
        caller = extract_source("obj.method(x)", "use.py", "use")
        self.assertEqual(caller.calls[0].confidence, "unresolved")
        self.assertEqual(detect(base, {"A": changed, "B": caller}), [])

    def test_divergent_definitions_are_reported(self):
        base = extract_source("def f(x): pass", "x.py", "x").definitions
        a = extract_source("def f(x, default=None): pass", "x.py", "x")
        b = extract_source("def f(x, *, ttl): pass", "x.py", "x")
        self.assertEqual(detect(base, {"A": a, "B": b})[0].kind, "divergent_def")

    def test_backward_compatible_evolution_is_not_conflict(self):
        base = extract_source("def f(x): pass", "x.py", "x").definitions
        agent = extract_source("def f(x, scale=1): pass", "x.py", "x")
        self.assertEqual(detect(base, {"A": agent}), [])

    def test_removed_symbol_with_surviving_caller(self):
        base = extract_source("def f(x): pass\nf(1)", "x.py", "x")
        removed = extract_source("", "x.py", "x")
        survivor = extract_source("from x import f\nf(2)", "user.py", "user")
        findings = detect(base.definitions, {"A": removed, "B": survivor})
        self.assertEqual([f.kind for f in findings], ["removed_symbol"])


if __name__ == "__main__":
    unittest.main()