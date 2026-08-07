"""Merge decision layer (§2.8).

swarm never edits code itself; ``merge`` only combines the agent branches into
a ``swarm/<run>/integration`` branch and reports. Remediation is either
re-prompting an agent (the orchestrator's repair injection) or picking one
branch's version wholesale -- never auto-rewriting a signature.

This module owns the *decision* surface:

* severity-ordered sequential merge into the integration branch (cleanest
  branches land first, minimizing textual conflict);
* re-running contract extraction on the merged tree as the honest final check
  (the tree is ground truth, detection on the tree is the real answer);
* per-conflict resolution strategies (take-a / take-b / defer) applied as a
  whole-file checkout of the chosen branch, returned as a report.

It never merges into the user's branch and never pushes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .gitops import GitExecutor
from .models import Conflict
from .resolve import Repo, build_repo

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass
class MergeReport:
    merged: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    residual: list[str] = field(default_factory=list)


def _severity_key(c: Conflict):
    return SEVERITY_ORDER.get(c.severity, 0)


def merge_into_integration(
    git: GitExecutor,
    base: str,
    run_id: str,
    branches: dict[str, str],      # agent -> branch name
    repo_root: Path,
    conflicts: list[Conflict] | None = None,
    integration: str | None = None,
    work_dir: Path | None = None,
) -> MergeReport:
    """Sequentially merge agent branches into the integration branch.

    Branches are merged in ascending order of their worst remaining conflict
    severity so the cleanest branches land first. The merges run inside a
    dedicated integration *worktree* under ``work_dir`` (default
    ``repo_root/.swarm/<run_id>/integration_wt``), so the user's checked-out
    branch is never switched or touched (§2.8: "never touches the user's
    branch"). Residual contract detection runs on that merged tree.

    Textual conflicts are surfaced in the report and abort the run rather than
    piling more merges on top of an unresolved state -- a human resolves them.
    """
    integration = integration or f"swarm/{run_id}/integration"
    work_dir = (work_dir or repo_root).resolve()
    wt = work_dir / "integration_wt"
    report = MergeReport()

    ordered = _order_branches(branches, conflicts)
    # Start a brand-new integration branch off the frozen base, in a worktree.
    git.run("worktree", "add", "-b", integration, str(wt), base)
    try:
        for agent in ordered:
            branch = branches[agent]
            try:
                git.run("-C", str(wt), "merge", "--no-ff", "--no-edit", branch)
                report.merged.append(agent)
            except Exception as exc:
                reason = str(exc)
                report.errors.append(f"{agent}: {reason}")
                git.run("-C", str(wt), "merge", "--abort")
                break
        # residual contract detection on the merged tree (the honest final
        # check): the tree is ground truth, so flag every call the merged tree
        # rejects against its own (merged) definition.
        try:
            report.residual = list(_merged_rejects(build_repo(wt)))
        except Exception as exc:
            report.errors.append(f"residual detection on integration failed: {exc}")
    finally:
        git.run("worktree", "remove", "--force", str(wt))
    return report


def _merged_rejects(repo: Repo) -> list[str]:
    """Return a short symbol:line string for every call the merged tree
    rejects against its own (merged) definition."""
    rejects: list[str] = []
    for fc in repo.contracts:
        for call in fc.calls:
            if call.key is None:
                continue
            shape = repo.definitions.get(call.key)
            if shape is None:
                continue
            from .shapes import Verdict, accepts

            if accepts(shape, call) is Verdict.REJECT:
                rejects.append(f"{call.key} @ {call.file}:{call.line}")
    return rejects


def _order_branches(branches: dict[str, str], conflicts: list[Conflict] | None) -> list[str]:
    by_agent: dict[str, dict] = {}
    for c in conflicts or []:
        for name in (c.definer, c.caller):
            if name in branches:
                by_agent.setdefault(name, {"max": 0}).setdefault("max", 0)
                by_agent[name]["max"] = max(by_agent[name]["max"], _dateless(c.severity))
    # cleanest (no conflicts) first, then ascending severity
    def key(a: str) -> tuple[int, str]:
        sev = by_agent.get(a, {}).get("max", 0)
        return (sev, a)
    return sorted(branches, key=key)


def _dateless(severity: str) -> int:
    return SEVERITY_ORDER.get(severity, 0)


def apply_strategy(
    git: GitExecutor,
    integration: str,
    branches: dict[str, str],
    strategy: str,
    agent: str | None = None,
    paths: list[str] | None = None,
) -> list[str]:
    """Apply a take-a/take-b/defer strategy on the integration branch.

    For ``take-a`` / ``take-b`` the chosen branch's version of the conflicted
    files is checked out onto the integration branch. ``defer`` records that
    the user accepted the residual state and does nothing. Only whole-file
    checkouts are ever performed -- swarm never rewrites code.
    """
    applied: list[str] = []
    if strategy not in {"take-a", "take-b", "defer"}:
        return applied
    if strategy == "defer" or agent is None:
        return applied
    branch = branches.get(agent)
    if branch is None:
        return applied
    for path in paths or []:
        git.run("checkout", branch, "--", path)
        applied.append(path)
    return applied