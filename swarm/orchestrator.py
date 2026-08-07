"""Orchestrator: agent lifecycle, worktrees, admission, settle->CONFIRMED
detection and cross-agent repair injection (§2.2, §2.8, §6.2).

The orchestrator owns everything the individual runners cannot:

* the frozen base commit and a detached worktree used to build the base
  contract index + base call-site index (the three-way-merge reference);
* one git worktree + branch per agent (manifest-before-git so a crash is
  reconcilable by ``swarm doctor``);
* running the agents concurrently (each ``Agent.run`` in its own worker) so a
  crash in one never cancels a run;
* a shared ``on_settle`` hook: when any agent hits a turn boundary the
  orchestrator folds that agent's *SETTLED* contract into a shared index,
  re-runs detection, emits confirmed conflicts, and -- for the
  ``BROKEN_CALLER`` case where the definer is settled and the caller is still
  live -- injects a peer system message into the caller's next turn, bounded
  by ``RunnerConfig.auto_repair_rounds``;
* cooperative cancellation (a shared ``threading.Event``).
"""

from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from .admission import Admission
from .conflicts import ConflictDetector
from .events import EventBus
from .gitops import GitExecutor, Manifest, WorktreeManager
from .models import Conflict, ConflictKind, FileContract, RunResult, Task
from .resolve import base_index, build_repo, merged_contract
from .runner import Agent, RunnerConfig
from .store import ContractStore
from .transport import Transport, TransportError

RUN_DIR_NAME = ".swarm"


class OrchestratorError(RuntimeError):
    pass


class Orchestrator:
    """Turns a task list + base commit into worktrees, agents, and a report."""

    def __init__(
        self,
        repo: Path,
        *,
        base: str | None = None,
        run_dir: Path | None = None,
        transport_factory: Callable[[str], Transport] | None = None,
        admission: Admission | None = None,
        auto_repair_rounds: int = 2,
        max_workers: int = 5,
        system_prompt: str = "",
    ) -> None:
        self.repo = Path(repo).resolve()
        self.git = GitExecutor(self.repo)
        self.base = base or self.git.head()
        self.top = self.repo.name.replace(".", "_")
        self.manifest_dir = (run_dir or self.repo / RUN_DIR_NAME).resolve()
        self.run_dir: Path | None = None
        self.event_bus = EventBus(self.manifest_dir / "events.jsonl")
        self.admission = admission or Admission()
        self.transport_factory = transport_factory
        self.auto_repair_rounds = auto_repair_rounds
        self.max_workers = max_workers
        self.system_prompt = system_prompt

        # shared cross-agent detection state (locked)
        self._lock = threading.Lock()
        self.agents: dict[str, Agent] = {}
        self.settled: dict[str, FileContract] = {}
        self.agent_states: dict[str, str] = {}
        self.sent_repairs: set[tuple[str, str, str]] = set()
        self.cancel = threading.Event()

    # -- base index ----------------------------------------------------------
    def build_base(self) -> None:
        base_dir = self.run_dir / "base"
        self.git.run("worktree", "add", "--detach", str(base_dir), self.base)
        try:
            repo = build_repo(base_dir, self.top)
            self.base_defs, self.base_callers = base_index(repo)
        finally:
            self.git.run("worktree", "remove", "--force", str(base_dir))

    # -- one run -------------------------------------------------------------
    def run(self, tasks: list[Task]) -> dict:
        if not tasks:
            raise OrchestratorError("no tasks to run")
        run_id = uuid.uuid4().hex[:4]
        self.run_dir = (self.manifest_dir / run_id).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.event_bus.path = self.run_dir / "events.jsonl"

        manifest = Manifest(self.run_dir / "run.json")
        manifest.write(state="intent", run_id=run_id, base=self.base,
                       tasks=[t.id for t in tasks])
        self.event_bus.emit("run_started", run_id=run_id, base=self.base)

        self.build_base()

        wm = WorktreeManager(self.git, self.run_dir)
        for task in tasks:
            try:
                branch = f"swarm/{run_id}/{task.id}"
                wt_dir = self.run_dir / task.id
                wm.create(wt_dir, branch, self.base)
            except Exception as exc:
                agent = None
                self.agent_states[task.id] = "FAILED"
                self.event_bus.emit("agent_state", agent=task.id, state="FAILED", turn=0)
                raise OrchestratorError(f"worktree for {task.id} failed: {exc}") from exc
            agent = self._create_agent(task, wt_dir)
            self.agents[task.id] = agent
            self.agent_states[task.id] = "PROVISIONING"
            self.event_bus.emit("agent_state", agent=task.id, state="PROVISIONING", turn=0)
        manifest.write(state="active", run_id=run_id, base=self.base,
                       tasks=[t.id for t in tasks])

        self._warmup()

        results: dict[str, RunResult] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._drive, task): task for task in tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    results[task.id] = fut.result()
                except Exception as exc:
                    # one agent's crash must never abort the run (§3.1/§5):
                    # fold the failure, keep going, and let the final
                    # contracts / checkpoint / manifest still run below.
                    with self._lock:
                        self.agent_states[task.id] = "FAILED"
                    self.event_bus.emit(
                        "agent_state", agent=task.id, state="FAILED", turn=0,
                        error=str(exc)[:300],
                    )
                    results[task.id] = RunResult(state="FAILED")

        self._final_contracts()
        conflicts = self._detect_and_emit()

        # Checkpoint every agent's work onto its branch *after* detection:
        # detection depends on reading which files each agent touched (a clean
        # working tree would hide them). Branches are the durable artifact.
        for task in tasks:
            agent = self.agents[task.id]
            try:
                self.git.checkpoint(agent.worktree, f"swarm checkpoint {run_id} {task.id}")
            except Exception:
                pass

        manifest.write(state="complete", run_id=run_id, base=self.base,
                       states=self.agent_states)
        return {
            "run_id": run_id,
            "results": results,
            "conflicts": conflicts,
            "summary": self._summary(),
        }

    def _create_agent(self, task: Task, wt: Path) -> Agent:
        return Agent(
            name=task.id,
            worktree=wt,
            transport=self.transport_factory(task.id) if self.transport_factory else None,
            admission=self.admission,
            store=ContractStore(),
            task_prompt=task.instructions or task.title,
            config=RunnerConfig(auto_repair_rounds=self.auto_repair_rounds),
            emit=self.event_bus.emit,
            system_prompt=self.system_prompt,
            on_settle=self._on_settle,
            cancel=self.cancel,
        )

    def _drive(self, task: Task) -> RunResult:
        with self._lock:
            self.agent_states[task.id] = "RUNNING"
        self.event_bus.emit("agent_state", agent=task.id, state="RUNNING", turn=0)
        result = self.agents[task.id].run()
        with self._lock:
            self.agent_states[task.id] = result.state
        return result

    def _warmup(self) -> None:
        """Cache-warm-then-fan-out (§2.2): fire one cache-warm request against
        the shared prefix (system prompt + tool list, byte-identical across
        agents) before the fleet launches, so the concurrent requests hit the
        cache instead of all missing. A warmup failure never blocks the run --
        it is an optimization, not a requirement."""
        by_name = list(self.agents.items())
        first = next((a for _, a in by_name if a.transport is not None), None)
        if first is None:
            return
        try:
            first.transport.warmup(
                [{"role": "system", "content": self.system_prompt}],
                first.tool_schema(),
            )
        except Exception:
            pass

    def _agent_contract(self, worktree: Path):
        repo = build_repo(worktree, self.top)
        touched = self.git.changed_vs_base(worktree, self.base)
        return merged_contract(repo, touched)

    # -- settle / detection --------------------------------------------------
    def _on_settle(self, agent: Agent) -> None:
        """Fired from the agent's own worker at a turn boundary."""
        with self._lock:
            self.agent_states[agent.name] = "SETTLED"
            self.settled[agent.name] = self._agent_contract(agent.worktree)
        self._detect_and_emit()

    def _final_contracts(self) -> None:
        for name, agent in self.agents.items():
            with self._lock:
                # Always re-fold from the agent's *final* worktree: an agent
                # may have settled early (its contract folded at that moment)
                # and then edited further -- e.g. fixing a caller after a peer
                # repair injection. The final report must reflect the final
                # tree, not the last settle boundary.
                self.settled[name] = self._agent_contract(agent.worktree)
                # A crash/abort is terminal and must not be reported DONE: the
                # manifest keeps the honest FAILED/ABORTED label even though
                # its (partial) contract is still folded for detection.
                if self.agent_states.get(name) in {
                    "PENDING", "PROVISIONING", "RUNNING", "SETTLING", "THROTTLED",
                }:
                    self.agent_states[name] = "DONE"

    def _detect_and_emit(self) -> list[Conflict]:
        with self._lock:
            settled = dict(self.settled)
        detector = ConflictDetector(self.base_defs, self.base_callers)
        conflicts, _ = detector.detect(settled)
        self._emit_conflicts(conflicts)
        self._inject_repairs(conflicts)
        return conflicts

    def _inject_repairs(self, conflicts: list[Conflict]) -> None:
        for c in conflicts:
            if c.kind not in (ConflictKind.BROKEN_CALLER, ConflictKind.BROKEN_BASE_CALLER):
                continue
            target, definer = c.caller, c.definer
            if target is None or definer is None:
                continue
            if not self._agent_live(target):
                continue
            rounds = self._repair_round(c.symbol, target)
            if rounds >= self.auto_repair_rounds:
                continue
            key = (c.symbol, definer, target)
            if key in self.sent_repairs:
                continue
            self.sent_repairs.add(key)
            message = _build_repair_message(c)
            agent = self.agents.get(target)
            if agent is not None:
                agent.pending_injections.append({"role": "system", "content": message})
            self.event_bus.emit("injected", agent=target, symbol=c.symbol,
                                round=rounds + 1)

    def _repair_round(self, symbol: str, caller: str) -> int:
        return sum(1 for k in self.sent_repairs if k[0] == symbol and k[2] == caller)

    def _agent_live(self, name: str) -> bool:
        return self.agent_states.get(name) not in {"DONE", "EXHAUSTED", "FAILED", "ABORTED"}

    # -- events --------------------------------------------------------------
    def emit_state(self, agent: str, state: str, turn: int) -> None:
        self.event_bus.emit("agent_state", agent=agent, state=state, turn=turn)

    def _emit_conflicts(self, conflicts: list[Conflict]) -> None:
        for c in conflicts:
            self.event_bus.emit(
                kind="conflict", conflict_kind=c.kind.value, symbol=c.symbol,
                definer=c.definer, caller=c.caller, severity=c.severity,
            )

    def _summary(self) -> dict:
        events = self.event_bus.read()
        return {"event_count": len(events), "states": dict(self.agent_states)}


def _build_repair_message(c: Conflict) -> str:
    lines = [
        "A peer agent changed an interface you call, in a worktree you cannot see.",
        f"  symbol: {c.symbol}",
    ]
    if c.a:
        lines.append(f"  was: {c.a}")
    if c.b:
        lines.append(f"  now: {c.b}")
    if c.file:
        lines.append(f"  breaking call site: {c.file}:{c.line}")
    lines.append("Update your call sites to match; the definition belongs to the")
    lines.append("other agent. If you believe the interface should not change,")
    lines.append("say so and stop rather than editing the definition.")
    return "\n".join(lines)


def default_transport_factory(_name: str) -> Transport | None:
    """Transport selected from env (SWARM_PROVIDER, *_API_KEY, SWARM_MODEL,
    SWARM_BASE_URL). Defaults to an OpenAI-compatible endpoint; Anthropic is
    selected with SWARM_PROVIDER=anthropic."""
    from .transport import AnthropicTransport, OpenAICompatTransport
    provider = os.environ.get("SWARM_PROVIDER", "openai").lower()
    max_tokens = int(os.environ.get("SWARM_MAX_TOKENS") or 16384)
    if provider == "anthropic":
        try:
            return AnthropicTransport(
                api_key=os.environ.get("ANTHROPIC_API_KEY"), max_tokens=max_tokens,
            )
        except TransportError:
            return None
    return OpenAICompatTransport(
        model=os.environ.get("SWARM_MODEL") or "gpt-4o-mini",
        base_url=os.environ.get("SWARM_BASE_URL") or "https://api.openai.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_tokens=max_tokens,
    )