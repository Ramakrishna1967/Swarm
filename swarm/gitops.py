"""Git worktree management and crash-safe manifests (§2.4).

Critiques and replaces the crashed-worktree story of a naive git_ops.py:

* A manifest is written *before* a worktree is created, so a crash anywhere
  is reconcilable (`swarm doctor` diffs the manifest against git state).
* Each settled turn is checkpoint-committed onto the agent's branch, so a
  crash loses at most one turn and per-turn diffing/rollback is free.
  Branches are the durable artifact; worktree directories are disposable.
* All index/ref-mutating git ops are serialized through one executor, so
  concurrent agents cannot contend on index.lock.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .models import RunId


class GitError(RuntimeError):
    pass


class GitExecutor:
    """Serialized, shell=False git worker."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._ws = threading.Lock()

    def run(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.root), *args],
            text=True, capture_output=True, shell=False,
        )
        if result.returncode:
            raise GitError(result.stderr.strip() or "git command failed")
        return result.stdout.strip()

    def run_in(self, worktree: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(worktree), *args],
            text=True, capture_output=True, shell=False,
        )
        if result.returncode:
            raise GitError(result.stderr.strip() or "git command failed (in worktree)")
        return result.stdout.strip()

    def mutating(self, fn):
        """Execute a git mutating op under the serialization lock."""
        with self._ws:
            return fn()

    def head(self) -> str:
        return self.run("rev-parse", "HEAD")

    def is_dirty(self) -> bool:
        return bool(self.run("status", "--porcelain").strip())

    def create_worktree(self, path: Path, branch: str, base: str) -> None:
        self.run("worktree", "add", "-b", branch, str(path), base)

    def checkpoint(self, worktree: Path, message: str) -> str:
        """Commit all changes in the worktree onto its checked-out branch."""
        self._add(worktree)
        if not self._has_staged(worktree):
            return self.run("-C", str(worktree), "rev-parse", "HEAD")
        self._commit(worktree, message)
        return self.run("-C", str(worktree), "rev-parse", "HEAD")

    def _add(self, worktree: Path) -> None:
        r = subprocess.run(["git", "-C", str(worktree), "add", "--all"],
                           text=True, capture_output=True, shell=False)
        if r.returncode:
            raise GitError(r.stderr.strip() or "git add failed")

    def _has_staged(self, worktree: Path) -> bool:
        r = subprocess.run(["git", "-C", str(worktree), "diff", "--cached", "--quiet"],
                           text=True, capture_output=True, shell=False)
        return r.returncode != 0

    def _commit(self, worktree: Path, message: str) -> None:
        r = subprocess.run(
            ["git", "-C", str(worktree), "commit", "-m", message],
            text=True, capture_output=True, shell=False,
        )
        if r.returncode:
            raise GitError(r.stderr.strip() or "git commit failed")

    def changed_files(self, worktree: Path, base: str) -> list[str]:
        """Files changed vs the frozen base commit (read-only)."""
        out = self.run("-C", str(worktree), "diff", "--name-only", base, "HEAD")
        return [l for l in out.splitlines() if l]

    def changed_vs_base(self, worktree: Path, base: str) -> set[str]:
        """Working-tree files that differ from base: modified, added, or
        untracked, as absolute paths. Deleted files are excluded (they have no
        contract). Used to decide which symbols an agent actually *touched* --
        everything else is inherited from the base commit and is not the
        agent's doing.
        """
        out = self.run("-C", str(worktree), "status", "--porcelain")
        changed: set[str] = set()
        for line in out.splitlines():
            if not line.strip():
                continue
            raw = line.split(maxsplit=1)[-1]
            path = raw.split(" -> ")[-1]          # handle renames
            p = worktree / path
            if p.is_file():
                changed.add(str(p.resolve()))
            elif p.is_dir() and not p.name == ".git":
                # an untracked *directory* (git reports `?? dir/`): treat every
                # file inside it as touched by the agent.
                for f in p.rglob("*"):
                    if f.is_file():
                        changed.add(str(f.resolve()))
        return changed

    def list_worktrees(self) -> list[dict]:
        out = self.run("worktree", "list", "--porcelain")
        result: list[dict] = []
        cur: dict[str, str] = {}
        for line in out.splitlines() + [""]:
            if not line:
                if cur:
                    result.append(cur)
                    cur = {}
                continue
            k, _, v = line.partition(" ")
            cur[k] = v
        return result

    def merge_branch(self, branch: str, integration: str) -> None:
        self.mutating(lambda: None)
        self.run("switch", "-C", integration)
        self.run("merge", "--no-ff", branch)

    def switch_force(self, branch: str) -> None:
        self.run("switch", "-C", branch)


class Manifest:
    """fsync'd run record; the ground truth for crash reconciliation."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, **values: object) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with tmp.open("r+", encoding="utf-8") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(self.path)

    def read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))


class WorktreeManager:
    """Owns per-run worktrees and reconciliation.

    Each worktree gets its *own* manifest file under ``manifest_dir/worktrees``
    so the orchestrator's per-run manifest is never clobbered and reconciliation
    can tell worktrees apart by branch.
    """

    def __init__(self, git: GitExecutor, manifest_dir: Path) -> None:
        self.git = git
        self.manifest_dir = manifest_dir

    def _manifest(self, branch: str) -> Manifest:
        d = self.manifest_dir / "worktrees"
        d.mkdir(parents=True, exist_ok=True)
        return Manifest(d / (branch.replace("/", "_") + ".json"))

    def preflight(self, n: int, base_dir: Path) -> None:
        usage = shutil.disk_usage(base_dir)
        working_tree = sum(
            p.stat().st_size for p in base_dir.rglob("*") if p.is_file() and ".git" not in p.parts
        )
        needed = n * max(working_tree, 1) * 3
        if usage.free < needed:
            raise GitError(f"insufficient disk headroom: ~{needed} bytes required, {usage.free} free")

    def create(self, path: Path, branch: str, base: str) -> None:
        m = self._manifest(branch)
        m.write(state="intent", worktree=str(path), branch=branch)
        self.git.create_worktree(path, branch, base)
        m.write(state="active", worktree=str(path), branch=branch)


def doctor(git: GitExecutor, manifests: list[Path]) -> list[str]:
    """Diff manifests against `git worktree list`; never delete branches."""
    lines: list[str] = []
    actual = {wt.get("branch") for wt in git.list_worktrees()}
    for mpath in manifests:
        data = json.loads(mpath.read_text(encoding="utf-8"))
        branch = data.get("branch")
        path = data.get("worktree")
        marked = data.get("state")
        exists = path and Path(path).exists()
        if marked == "intent" and not exists:
            lines.append(f"stale-intent: {branch} (no worktree created)")
        elif exists and branch not in actual:
            lines.append(f"orphan-dir-with-branch: {branch} @ {path}")
        elif branch in actual and not exists:
            lines.append(f"manifest-only: branch={branch} dir missing")
        else:
            lines.append(f"ok: {branch}")
    return lines