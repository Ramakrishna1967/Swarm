"""Merge layer unit tests (§2.8): severity ordering, whole-file strategies,
and residual contract detection on the merged integration tree.

merge_into_integration is exercised end-to-end in test_e2e.py against a real
git worktree; here we unit-test the decision surface in isolation: the branch
ordering, the take-a/take-b/defer strategies, and the residual detector.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from swarm.gitops import GitExecutor
from swarm.merge import MergeReport, apply_strategy, _merged_rejects, _order_branches
from swarm.models import Conflict, ConflictKind, SymbolKey


def _git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "services").mkdir(parents=True)
    (root / "services" / "auth.py").write_text(
        "def authenticate(user):\n    return user\n", encoding="utf-8")
    git = subprocess.run(["git", "-C", str(root), "init", "-q"],
                         capture_output=True, text=True)
    assert git.returncode == 0, git.stderr
    subprocess.run(["git", "-C", str(root), "config", "user.email", "t@t"],
                   capture_output=True, text=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "t"],
                   capture_output=True, text=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], capture_output=True)
    r = subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return root


def _git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), *args],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _conflict(agent: str, severity: str) -> Conflict:
    return Conflict(kind=ConflictKind.BROKEN_CALLER, symbol="s",
                    definer=agent, caller="other", detail="x", severity=severity)


def test_order_branches_cleanest_first():
    branches = {"a": "b/a", "b": "b/b", "c": "b/c"}
    conflicts = [_conflict("a", "high"), _conflict("b", "medium")]
    ordered = _order_branches(branches, conflicts)
    assert ordered[0] == "c"          # untouched by any conflict
    assert ordered[1] == "b"          # lower severity
    assert ordered[2] == "a"          # highest severity


def test_order_branches_no_conflicts_preserves_branch_order():
    assert _order_branches({"x": "x", "y": "y", "z": "z"}, None) == ["x", "y", "z"]


def test_order_branches_ignores_unknown_agents():
    branches = {"a": "b/a"}
    conflicts = [_conflict("nope", "high")]  # agent not in branches
    assert _order_branches(branches, conflicts) == ["a"]


def _branch_commit(root: Path, branch: str, path: str, content: str) -> None:
    _git(root, "checkout", "-qb", branch)
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", f"{branch} change")
    _git(root, "checkout", "-q", "master")


def test_apply_strategy_take_b_overwrites_integration(tmp_path):
    root = _git_repo(tmp_path)
    _branch_commit(root, "wt_a", "services/auth.py",
                   "def authenticate(user, scope):\n    return user\n")
    _branch_commit(root, "wt_b", "services/auth.py",
                   "def authenticate(user, scope, ttl):\n    return user\n")
    _git(root, "checkout", "-qb", "swarm/r/integration")

    applied = apply_strategy(GitExecutor(root), "swarm/r/integration",
                             {"a": "wt_a", "b": "wt_b"}, "take-b", "b",
                             ["services/auth.py"])
    assert applied == ["services/auth.py"]
    assert "ttl" in (root / "services/auth.py").read_text(encoding="utf-8")


def test_apply_strategy_take_a_picks_other_side(tmp_path):
    root = _git_repo(tmp_path)
    _branch_commit(root, "wt_a", "services/auth.py",
                   "def authenticate(user, scope):\n    return user\n")
    _branch_commit(root, "wt_b", "services/auth.py",
                   "def authenticate(user, scope, ttl):\n    return user\n")
    _git(root, "checkout", "-qb", "swarm/r/integration")

    apply_strategy(GitExecutor(root), "swarm/r/integration",
                   {"a": "wt_a", "b": "wt_b"}, "take-a", "a", ["services/auth.py"])
    text = (root / "services/auth.py").read_text(encoding="utf-8")
    assert "scope" in text and "ttl" not in text


def test_apply_strategy_defer_does_nothing(tmp_path):
    root = _git_repo(tmp_path)
    applied = apply_strategy(GitExecutor(root), "swarm/r/integration",
                             {"a": "wt_a"}, "defer", "a", ["services/auth.py"])
    assert applied == []


def test_apply_strategy_unknown_strategy_is_noop(tmp_path):
    root = _git_repo(tmp_path)
    assert apply_strategy(GitExecutor(root), "b", {}, "merge-everything",
                          "a", ["services/auth.py"]) == []


def test_merged_rejects_flags_broken_call(tmp_path):
    root = _git_repo(tmp_path)
    # merged tree: definition grew a required param but the caller didn't.
    (root / "services/auth.py").write_text(
        "def authenticate(user, scope):\n    return user\n", encoding="utf-8")
    (root / "admin").mkdir(exist_ok=True)
    (root / "admin" / "views.py").write_text(
        "from services.auth import authenticate\n"
        "def login(request):\n    return authenticate(request.user)\n", encoding="utf-8")
    from swarm.resolve import build_repo
    rejects = _merged_rejects(build_repo(root, "proj"))
    assert len(rejects) == 1
    assert "services.auth:authenticate" in rejects[0]


def test_merged_rejects_silent_when_repaired(tmp_path):
    root = _git_repo(tmp_path)
    (root / "services/auth.py").write_text(
        "def authenticate(user, scope):\n    return user\n", encoding="utf-8")
    (root / "admin").mkdir(exist_ok=True)
    (root / "admin" / "views.py").write_text(
        "from services.auth import authenticate\n"
        "def login(request):\n    return authenticate(request.user, 'read')\n",
        encoding="utf-8")
    from swarm.resolve import build_repo
    assert _merged_rejects(build_repo(root, "proj")) == []


def test_merge_into_integration_reports_residual(tmp_path):
    from swarm.merge import merge_into_integration
    root = _git_repo(tmp_path)
    base = _git(root, "rev-parse", "HEAD")
    _branch_commit(root, "wt_a", "services/auth.py",
                   "def authenticate(user, scope):\n    return user\n")
    _branch_commit(root, "wt_b", "admin/views.py",
                   "from services.auth import authenticate\n"
                   "def login(request):\n    return authenticate(request.user)\n")
    rep = merge_into_integration(GitExecutor(root), base, "run1",
                                 {"a": "wt_a", "b": "wt_b"}, root,
                                 work_dir=root / ".swarm" / "run1")
    assert isinstance(rep, MergeReport)
    assert set(rep.merged) == {"a", "b"}
    assert rep.errors == [], rep.errors
    assert len(rep.residual) == 1, rep.residual


def test_merge_into_integration_never_touches_user_branch(tmp_path):
    from swarm.merge import merge_into_integration
    root = _git_repo(tmp_path)
    base = _git(root, "rev-parse", "HEAD")
    _branch_commit(root, "wt_a", "services/auth.py",
                   "def authenticate(user, scope):\n    return user\n")
    _branch_commit(root, "wt_b", "admin/views.py",
                   "def login():\n    return 1\n")
    current = _git(root, "rev-parse", "HEAD")
    merge_into_integration(GitExecutor(root), base, "run1",
                           {"a": "wt_a", "b": "wt_b"}, root,
                           work_dir=root / ".swarm" / "run1")
    # user's checked-out branch is untouched: same commit, still master.
    assert _git(root, "rev-parse", "HEAD") == current
    assert _git(root, "branch", "--show-current") == "master"


def test_merge_into_integration_textual_conflict_is_surfaced(tmp_path):
    from swarm.merge import merge_into_integration
    root = _git_repo(tmp_path)
    base = _git(root, "rev-parse", "HEAD")
    # both agents edit the same region of the same file differently.
    _branch_commit(root, "wt_a", "services/auth.py",
                   "def authenticate(user):\n    return user + 1\n")
    _branch_commit(root, "wt_b", "services/auth.py",
                   "def authenticate(user):\n    return user * 2\n")
    rep = merge_into_integration(GitExecutor(root), base, "run1",
                                 {"a": "wt_a", "b": "wt_b"}, root,
                                 work_dir=root / ".swarm" / "run1")
    assert rep.merged, rep.merged
    assert rep.errors, rep.errors  # second merge hits the conflict
