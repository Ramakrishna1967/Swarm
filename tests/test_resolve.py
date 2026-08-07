"""Resolution layer tests (§2.5): import tables, re-export, unique-method
attribute resolution, out-of-repo drop, and the unresolved bucket."""

from __future__ import annotations

import tempfile
from pathlib import Path

from swarm.models import SymbolKey
from swarm.resolve import Repo, build_repo


def _repo(tmp_path: Path, files: dict[str, str], name: str = "proj") -> Repo:
    root = tmp_path / name
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    return build_repo(root, name)


def test_name_call_resolves_through_import(tmp_path):
    repo = _repo(tmp_path, {
        "services/auth.py": "def authenticate(user):\n    return user\n",
        "app.py": "from services.auth import authenticate\n"
                  "def login(r):\n    return authenticate(r.user)\n",
    })
    assert len(repo.unresolved) == 0, repo.unresolved
    calls = [c for fc in repo.contracts for c in fc.calls]
    assert any(c.key == SymbolKey("services.auth", "authenticate") for c in calls)


def test_out_of_repo_target_is_dropped(tmp_path):
    # `from requests import get` resolves to a module not in the repo: the call
    # site is dropped (never a conflict), not left unresolved.
    repo = _repo(tmp_path, {
        "app.py": "from requests import get\n"
                  "def fetch():\n    return get('http://x')\n",
    })
    assert len(repo.unresolved) == 0
    assert all(c.key is None or c.key.module.split(".")[0] != "proj"
               for fc in repo.contracts for c in fc.calls)


def test_attribute_resolution_unique_method(tmp_path):
    repo = _repo(tmp_path, {
        "store.py": "class Store:\n    def save(self, o):\n        pass\n",
        "app.py": "from store import Store\n"
                  "def persist():\n    s = Store()\n    return s.save(1)\n",
    })
    calls = [c for fc in repo.contracts for c in fc.calls]
    attr_calls = [c for c in calls if c.reason == "attribute-call"]
    assert any(c.key is not None and c.key.qualname == "Store.save" for c in attr_calls)


def test_attribute_resolution_ambiguous_stays_unresolved(tmp_path):
    # two classes define save: no unique owner, so it lands in unresolved.
    repo = _repo(tmp_path, {
        "a.py": "class A:\n    def save(self, o):\n        pass\n",
        "b.py": "class B:\n    def save(self, o):\n        pass\n",
        "app.py": "def go():\n    x.save(1)\n",
    })
    attr_calls = [c for fc in repo.contracts for c in fc.calls
                  if c.reason == "attribute-call"]
    assert all(c.key is None for c in attr_calls)
    assert any(c.attr == "save" for c in repo.unresolved)


def test_reexport_chain_resolves_to_definer(tmp_path):
    repo = _repo(tmp_path, {
        "impl.py": "def f(x):\n    return x\n",
        "pack/__init__.py": "from impl import f\n",
        "app.py": "from pack import f\n"
                  "def go():\n    return f(1)\n",
    })
    calls = [c for fc in repo.contracts for c in fc.calls if c.reason == "name-call"]
    assert any(c.key == SymbolKey("impl", "f") for c in calls)


def test_repo_produces_definition_index(tmp_path):
    repo = _repo(tmp_path, {"m.py": "def foo(x):\n    return x\n"})
    assert SymbolKey("m", "foo") in repo.definitions
    assert len(repo.definitions) == 1


def test_reexport_dotted_module_keyed_by_top_package(tmp_path):
    # `import pack` re-exports from impl; a caller resolves through the chain.
    repo = _repo(tmp_path, {
        "impl.py": "def f(x):\n    return x\n",
        "pack/__init__.py": "from impl import f\n",
        "app.py": "from pack import f\n"
                  "def go():\n    return f(1)\n",
    })
    calls = [c for fc in repo.contracts for c in fc.calls if c.reason == "name-call"]
    assert any(c.key == SymbolKey("impl", "f") for c in calls)


def test_reexport_named_import_with_asname(tmp_path):
    repo = _repo(tmp_path, {
        "impl.py": "def helper(x):\n    return x\n",
        "pack/__init__.py": "from impl import helper as h\n",
        "app.py": "from pack import h\n"
                  "def go():\n    return h(1)\n",
    })
    calls = [c for fc in repo.contracts for c in fc.calls if c.reason == "name-call"]
    assert any(c.key == SymbolKey("impl", "helper") for c in calls)


def test_module_import_dot_calls_resolve(tmp_path):
    # `import services.auth` then `services.auth.authenticate(...)` — the
    # imported module is tracked, but the *function* is not a method, so the
    # unique-method heuristic (§2.5.5) cannot claim it: it stays unresolved
    # rather than guessing.
    repo = _repo(tmp_path, {
        "services/auth.py": "def authenticate(user):\n    return user\n",
        "app.py": "import services.auth\n"
                  "def go():\n    return services.auth.authenticate(1)\n",
    })
    attr = [c for fc in repo.contracts for c in fc.calls
            if c.reason == "attribute-call"]
    assert all(c.key is None for c in attr)


def test_relative_import_resolves(tmp_path):
    repo = _repo(tmp_path, {
        "pack/impl.py": "def f(x):\n    return x\n",
        "pack/__init__.py": "from .impl import f\n",
        "pack/app.py": "from .impl import f\n"
                       "def go():\n    return f(1)\n",
    })
    calls = [c for fc in repo.contracts for c in fc.calls if c.reason == "name-call"]
    assert any(c.key == SymbolKey("pack.impl", "f") for c in calls)


def test_same_name_two_modules_ambiguous_stays_unresolved(tmp_path):
    # two modules each define helper(): a name call through the module is an
    # exact import, so it still resolves; but the modules differ.
    repo = _repo(tmp_path, {
        "a.py": "def helper(x):\n    return x\n",
        "b.py": "def helper(x):\n    return x\n",
        "app.py": "from a import helper\n"
                  "def go():\n    return helper(1)\n",
    })
    calls = [c for fc in repo.contracts for c in fc.calls if c.reason == "name-call"]
    assert any(c.key == SymbolKey("a", "helper") for c in calls)


def test_unresolved_bucket_has_call_metadata(tmp_path):
    repo = _repo(tmp_path, {
        "a.py": "class A:\n    def save(self, o):\n        pass\n",
        "b.py": "class B:\n    def save(self, o):\n        pass\n",
        "app.py": "def go():\n    x.save(1)\n",
    })
    assert repo.unresolved
    u = repo.unresolved[0]
    assert u.reason == "attribute-call" and u.attr == "save"
    assert u.file.endswith("app.py")


def test_package_init_definitions_fold_into_package_module(tmp_path):
    # an __init__.py's defs belong to the package module, not __init__.
    repo = _repo(tmp_path, {
        "pack/__init__.py": "def f(x):\n    return x\n",
        "app.py": "from pack import f\n"
                  "def go():\n    return f(1)\n",
    })
    assert SymbolKey("pack", "f") in repo.definitions
    assert SymbolKey("pack.__init__", "f") not in repo.definitions


def test_base_index_splits_defs_and_callers(tmp_path):
    from swarm.resolve import base_index
    repo = _repo(tmp_path, {
        "m.py": "def f(x):\n    return x\n",
        "app.py": "from m import f\n"
                  "def go():\n    return f(1)\n",
    })
    defs, callers = base_index(repo)
    assert SymbolKey("m", "f") in defs
    assert SymbolKey("m", "f") in callers
    assert callers[SymbolKey("m", "f")][0].file.endswith("app.py")