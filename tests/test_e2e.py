"""End-to-end: Orchestrator on a real git repo with MockTransport.

Covers §2.2/§2.8/§6.2: worktrees + branches, concurrent runs, checkpointing,
broken_caller detection, repair injection, and the merge layer producing an
integration branch with zero residual conflicts.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from swarm.events import EventBus
from swarm.gitops import GitExecutor
from swarm.merge import merge_into_integration
from swarm.models import Conflict, ConflictKind, Task
from swarm.orchestrator import Orchestrator
from swarm.transport import MockTransport, ToolCall, Transport, TurnResult, Usage


def git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), *args],
                       capture_output=True, text=True, shell=False)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "services").mkdir(parents=True)
    (root / "services" / "auth.py").write_text(
        "def authenticate(user):\n    return user\n", encoding="utf-8")
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@t")
    git(root, "config", "user.name", "t")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "base")
    return root


def _scripts() -> dict[str, list[dict]]:
    return {
        "A": [{"tool": "write_file",
               "inputs": {"path": "services/auth.py",
                          "content": "def authenticate(user, scope):\n    return user\n"}},
              {"tool": "report_done",
               "inputs": {"summary": "scope", "files_changed": ["services/auth.py"]}}],
        "B": [{"tool": "write_file",
               "inputs": {"path": "admin/views.py",
                          "content": "from services.auth import authenticate\n"
                                     "def login(request):\n"
                                     "    return authenticate(request.user)\n"}},
              {"tool": "report_done",
               "inputs": {"summary": "caller", "files_changed": ["admin/views.py"]}}],
    }


def test_orchestrator_detects_broken_caller(tmp_path):
    root = _git_repo(tmp_path)
    scripts = _scripts()
    orch = Orchestrator(root, run_dir=tmp_path / "runs",
                        transport_factory=lambda n: MockTransport(scripts[n]))
    out = orch.run([Task(id="A", title="evolve", instructions="add scope"),
                    Task(id="B", title="caller", instructions="call authenticate")])
    assert out["results"]["A"].state == "DONE"
    assert out["results"]["B"].state == "DONE"
    assert len(out["conflicts"]) == 1
    c = out["conflicts"][0]
    assert c.kind == ConflictKind.BROKEN_CALLER
    assert c.definer == "A" and c.caller == "B"
    assert c.symbol == "services.auth:authenticate"


def test_one_agent_crash_does_not_abort_run(tmp_path):
    """§3.1/§5: a crash in one agent's worker must not cancel the run -- the
    failed agent folds to FAILED, the rest finalize, checkpoint, and the
    manifest is written complete."""
    root = _git_repo(tmp_path)
    scripts = _scripts()
    orch = Orchestrator(root, run_dir=tmp_path / "runs",
                        transport_factory=lambda n: MockTransport(scripts.get(n)))
    orig = orch._drive

    def drive(task):
        if task.id == "B":
            raise RuntimeError("boom")
        return orig(task)

    orch._drive = drive

    out = orch.run([Task(id="A", title="evolve", instructions="add scope"),
                    Task(id="B", title="caller", instructions="call authenticate")])
    assert out["results"]["A"].state == "DONE"
    assert out["results"]["B"].state == "FAILED"
    # sanity: the run still finalized and wrote a complete manifest
    manifest = _read_manifest(orch, out["run_id"])
    assert manifest.get("state") == "complete"
    assert manifest.get("states", {}).get("B") == "FAILED"


def _read_manifest(orch, run_id):
    import json
    p = orch.manifest_dir / run_id / "run.json"
    return json.loads(p.read_text(encoding="utf-8"))


def test_checkpoints_then_merge_produces_integration(tmp_path):
    root = _git_repo(tmp_path)
    scripts = _scripts()
    orch = Orchestrator(root, run_dir=tmp_path / "runs",
                        transport_factory=lambda n: MockTransport(scripts[n]))
    out = orch.run([Task(id="A", title="evolve", instructions="add scope"),
                    Task(id="B", title="caller", instructions="call authenticate")])
    run_id = out["run_id"]

    # agents checkpointed their work onto swarm/<run>/<agent> branches
    branches = {name: f"swarm/{run_id}/{name}" for name in orch.agents}
    for b in branches.values():
        assert git(root, "rev-parse", "--verify", b)

    base = git(root, "rev-parse", "HEAD")
    git_exec = GitExecutor(root)
    rep = merge_into_integration(git_exec, base, run_id, branches, root,
                                 conflicts=out["conflicts"])
    assert set(rep.merged) >= {"A", "B"}, rep
    assert rep.errors == [], rep.errors
    # B's caller still uses the old one-arg form, so the merge honestly reports
    # the residual contract conflict for a human to resolve -- swarm never edits.
    assert len(rep.residual) == 1, rep.residual
    assert "services.auth:authenticate" in rep.residual[0]


def test_repair_injection_bounded(tmp_path):
    root = _git_repo(tmp_path)
    orch = Orchestrator(root, run_dir=tmp_path / "runs", auto_repair_rounds=2,
                        transport_factory=lambda n: MockTransport([]))
    # set up a live caller agent so the injection has a target
    from swarm.runner import Agent, RunnerConfig
    from swarm.admission import Admission
    from swarm.store import ContractStore
    caller = Agent(name="B", worktree=tmp_path / "wtB", transport=None,
                   admission=Admission(), store=ContractStore(),
                   task_prompt="x", config=RunnerConfig(auto_repair_rounds=2),
                   emit=lambda **k: None)
    orch.agents["B"] = caller
    orch.agent_states["B"] = "RUNNING"
    orch.settled["A"] = _agent_contract_shape(tmp_path / "runs")

    conflict = Conflict(kind=ConflictKind.BROKEN_CALLER, symbol="m.authenticate",
                        definer="A", caller="B", detail="x",
                        severity="high", file="v.py", line=1)
    for _ in range(5):  # repeated calls must not re-inject past the bound
        orch._inject_repairs([conflict])
    injections = [m for m in caller.pending_injections if m["role"] == "system"]
    assert len(injections) == 1, injections
    assert "A peer agent" in injections[0]["content"]


def _agent_contract_shape(run_dir: Path):
    from swarm.models import FileContract, Shape, SymbolKey
    fc = FileContract("x")
    fc.definitions[SymbolKey("m", "authenticate")] = Shape(
        ("user", "scope"), 0, 2, frozenset(), frozenset(), False, False,
        False, False)
    return fc


# ---- live repair loop -------------------------------------------------------
def _turn(calls, stop="tool_use") -> TurnResult:
    return TurnResult(content=[], stop_reason=stop, tool_calls=calls,
                      usage=Usage(input_tokens=1, output_tokens=1))


def _call(name: str, inputs: dict) -> ToolCall:
    return ToolCall(id=f"tc_{name}", name=name, inputs=inputs)


class RepairScript(Transport):
    """Scripted transport whose turns are ordered so the caller (B) settles
    only after the definer (A) has folded its new shape -- the point at which
    detection fires and the peer repair message is injected into B. B's next
    turn must then carry that injection, which is what the test asserts."""

    def __init__(self, name: str, a_settled: threading.Event) -> None:
        self.name = name
        self.a_settled = a_settled
        self.n = 0

    def warmup(self, messages, tools) -> None:
        return None

    def turn(self, messages, tools, **extra) -> TurnResult:
        self.n += 1
        if self.name == "A":
            if self.n == 1:
                return _turn([_call("write_file", {"path": "services/auth.py",
                            "content": "def authenticate(user, scope):\n    return user\n"})])
            if self.n == 2:
                return _turn([], stop="stop")
            if self.n == 3:
                self.a_settled.set()
                return _turn([_call("report_done", {"summary": "scope",
                            "files_changed": ["services/auth.py"]})])
        else:  # B, the caller
            if self.n == 1:
                return _turn([_call("write_file", {"path": "admin/views.py",
                            "content": "from services.auth import authenticate\n"
                                       "def login(request):\n"
                                       "    return authenticate(request.user)\n"})])
            if self.n == 2:
                self.a_settled.wait(timeout=10)
                return _turn([], stop="stop")
            if self.n == 3:
                texts = " ".join(m.get("content", "")
                                 for m in messages if m.get("role") == "system")
                assert "A peer agent" in texts, messages
                return _turn([_call("edit_file", {"path": "admin/views.py",
                            "old": "authenticate(request.user)",
                            "new": "authenticate(request.user, 'read')"}),
                              _call("report_done", {"summary": "fixed caller",
                            "files_changed": ["admin/views.py"]})])
        return _turn([], stop="stop")


def test_live_repair_injection_loop(tmp_path):
    """The flagship path, live: caller settles, receives the peer injection,
    then edits its caller; the final report is clean (no residual broken
    caller) because the repair landed on disk."""
    root = _git_repo(tmp_path)
    a_settled = threading.Event()

    def factory(name: str):
        return RepairScript(name, a_settled)

    orch = Orchestrator(root, run_dir=tmp_path / "runs",
                        transport_factory=factory)
    out = orch.run([Task(id="A", title="evolve", instructions="add scope"),
                    Task(id="B", title="caller", instructions="call authenticate")])
    assert out["results"]["A"].state == "DONE"
    assert out["results"]["B"].state == "DONE"
    # B repaired its caller, so the final report is clean.
    assert out["conflicts"] == [], out["conflicts"]
    # B's worktree really does contain the two-arg call now.
    views = orch.run_dir / "B" / "admin" / "views.py"
    assert "authenticate(request.user, 'read')" in views.read_text(encoding="utf-8")
    # the injection was actually emitted to the event bus mid-run.
    events = EventBus(orch.run_dir / "events.jsonl").read()
    assert any(e.get("kind") == "injected" and e.get("agent") == "B" for e in events), events
    # merge the branches: nothing residual, the repair won end-to-end.
    base = git(root, "rev-parse", "HEAD")
    git_exec = GitExecutor(root)
    branches = {"A": f"swarm/{out['run_id']}/A", "B": f"swarm/{out['run_id']}/B"}
    rep = merge_into_integration(git_exec, base, out["run_id"], branches, root,
                                 conflicts=out["conflicts"])
    assert set(rep.merged) >= {"A", "B"}, rep
    assert rep.errors == [], rep.errors
    assert rep.residual == [], rep.residual