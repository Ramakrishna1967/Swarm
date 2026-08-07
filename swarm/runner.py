"""Agent runner: the tool loop, sandbox, budgets, and the settle signal (§2.3).

The tools are exposed to the model as a fixed, arity-stable list. Every file
write goes through a typed tool so the contract store can be poked on write
(the debounced extraction trigger lives *inside* write_file/edit_file). The
turn boundary -- an assistant turn that ends with no tool use -- is the
settle signal: it is the point at which the agent has completed a coherent
unit of work by its own judgment.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .admission import Admission
from .models import RunResult
from .store import ContractStore
from .transport import (
    RateLimitError, RetryableTransportError, ToolCall, TransportError, TurnResult,
)

TOOL_NAMES = [
    "list_dir", "read_file", "write_file", "edit_file",
    "grep", "run_shell", "report_done",
]

MAX_ITERATIONS_DEFAULT = 40
WALL_CLOCK_DEFAULT = 30 * 60  # seconds

# A turn that ends with no tool use is a settle signal (a coherent-work
# boundary), not completion. `SETTLE_SILENCE` lists the stop reasons that make
# an empty turn a normal silent boundary so the loop keeps going; anything else
# (e.g. a refusal) is surfaced by NOT being in the set and still continues the
# loop rather than falsely completing. Completion is only `report_done`.
SETTLE_SILENCE = {"stop"}


class SandboxViolation(RuntimeError):
    pass


@dataclass
class RunnerConfig:
    max_iterations: int = MAX_ITERATIONS_DEFAULT
    wall_clock_s: float = WALL_CLOCK_DEFAULT
    task_budget_tokens: int = 0  # 0 == unlimited
    repair_rounds_remaining: int = 2
    auto_repair_rounds: int = 2


@dataclass
class ToolResult:
    content: str
    value: bool = True


class Sandbox:
    """Ensures every path stays under the agent's worktree root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if str(p) != str(self.root) and not str(p).startswith(str(self.root) + os.sep):
            raise SandboxViolation(f"path escapes the worktree: {path}")
        return p

    def validate(self, path: str) -> Path:
        return self.resolve(path)


class Agent:
    """One runner: owns the conversation, the tool loop, and its token budget."""

    def __init__(
        self,
        name: str,
        worktree: Path,
        transport,
        admission: Admission,
        store: ContractStore,
        task_prompt: str,
        config: RunnerConfig | None = None,
        emit=None,
        system_prompt: str = "",
        on_settle: Callable[["Agent"], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> None:
        self.name = name
        self.worktree = worktree
        self.sandbox = Sandbox(worktree)
        self.transport = transport
        self.admission = admission
        self.store = store
        self.config = config or RunnerConfig()
        self.emit = emit or (lambda *a, **k: None)
        self.system_prompt = system_prompt
        self.on_settle = on_settle
        self.cancel = cancel
        self.messages: list[dict] = []
        self.state: str = "PENDING"
        self.turn = 0
        self.files_touched: set[str] = set()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.tool_results: list[dict] = []
        # peer-contract injections ({role: system}) queued for next turn
        self.pending_injections: list[dict] = []
        self.task_prompt = f"{task_prompt}".strip()

    def _append_system(self, role: str, content: str) -> None:
        if content:
            self.messages.append({"role": "system", "content": content})

    def tool_schema(self) -> list[dict]:
        """The tool list in the canonical OpenAI function shape; each backend
        converts it to its own wire format."""
        def fn(name, desc, properties, required):
            return {
                "type": "function",
                "function": {
                    "name": name, "description": desc,
                    "parameters": {"type": "object", "properties": properties, "required": required},
                },
            }
        return [
            fn("read_file", "read a file from the worktree",
               {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}, ["path"]),
            fn("write_file", "write a whole file (overwrites)",
               {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
            fn("edit_file", "replace an old string with new in a file",
               {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, ["path", "old", "new"]),
            fn("list_dir", "list a directory", {"path": {"type": "string"}, "depth": {"type": "integer"}}, ["path"]),
            fn("grep", "search file contents", {"pattern": {"type": "string"}, "glob": {"type": "string"}}, ["pattern"]),
            fn("run_shell", "run a shell command (tests, lint, git)",
               {"command": {"type": "string"}, "timeout_s": {"type": "integer"}}, ["command"]),
            fn("report_done", "declare the task complete",
               {"summary": {"type": "string"}, "files_changed": {"type": "array", "items": {"type": "string"}}}, ["summary"]),
        ]

    # -- tool implementations -----------------------------------------------
    def _call_tool(self, call: ToolCall) -> ToolResult:
        name, inputs = call.name, call.inputs
        handler = getattr(self, "_tool_" + name, None)
        if handler is None:
            return ToolResult("unknown tool: " + name, value=False)
        try:
            content = handler(**inputs)
            return ToolResult(str(content))
        except SandboxViolation as exc:
            return ToolResult(f"error: {exc}")
        except Exception as exc:
            return ToolResult(f"error: {exc}")

    def _tool_read_file(self, path: str, start: int = 1, end: int | None = None) -> str:
        p = self.sandbox.resolve(path)
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        if end is None:
            end = len(lines)
        sel = lines[max(start, 1) - 1: end]
        return "\n".join(f"{i+1}: {l}" for i, l in enumerate(sel, start=start))

    def _tool_write_file(self, path: str, content: str) -> str:
        p = self.sandbox.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        self.files_touched.add(path)
        return f"wrote {path} ({len(content)} bytes)"

    def _tool_edit_file(self, path: str, old: str, new: str) -> str:
        p = self.sandbox.resolve(path)
        text = p.read_text(encoding="utf-8")
        if old not in text:
            raise ValueError(f"old string not found in {path}")
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        self.files_touched.add(path)
        return f"edited {path}"

    def _tool_list_dir(self, path: str = ".", depth: int = 2) -> str:
        p = self.sandbox.resolve(path)
        return "\n".join(sorted(p.name for p in p.iterdir() if not p.name.startswith(".git"))) if p.is_dir() else "not a dir"

    def _tool_run_shell(self, command: str, timeout_s: int = 120) -> str:
        result = subprocess.run(
            shlex.split(command), cwd=self.worktree, shell=False,
            text=True, capture_output=True, timeout=timeout_s,
        )
        out = (result.stdout or "") + (result.stderr or "")
        return out[-4000:] or "(no output)"

    def _tool_report_done(self, summary: str, files_changed: list) -> str:
        return f"accepted: {summary}"

    # -- loop ----------------------------------------------------------------
    def build_messages(self) -> list[dict]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        if self.task_prompt:
            msgs.append({"role": "user", "content": self.task_prompt})
        # peer-contract injections go in as system messages after cached history
        msgs.extend(self.pending_injections)
        msgs.extend(self.messages)
        return msgs

    def run(self) -> RunResult:
        self.state = "RUNNING"
        self.emit(kind="agent_state", agent=self.name, state="RUNNING", turn=0)
        start = time.monotonic()
        for iteration in range(self.config.max_iterations):
            self.turn = iteration + 1
            if self.cancel is not None and self.cancel.is_set():
                self.state = "ABORTED"
                self.emit(kind="agent_state", agent=self.name, state="ABORTED", turn=self.turn)
                break
            if time.monotonic() - start > self.config.wall_clock_s:
                self.state = "EXHAUSTED"
                self.emit(kind="agent_state", agent=self.name, state="EXHAUSTED", turn=self.turn)
                break
            # admission gate
            if not self.admission.acquire():
                self.state = "THROTTLED"
                self.emit(kind="agent_state", agent=self.name, state="THROTTLED", turn=self.turn)
                time.sleep(1.0)
                continue
            try:
                result: TurnResult = self.transport.turn(
                    self.build_messages(), self.tool_schema(),
                )
            except RateLimitError as exc:
                self.admission.release()
                self.state = "THROTTLED"
                self.emit(kind="agent_state", agent=self.name, state="THROTTLED",
                          turn=self.turn, retry_after=exc.retry_after)
                time.sleep(min(exc.retry_after or 1.0, 30.0))
                continue
            except RetryableTransportError:
                self.admission.release()
                self.state = "THROTTLED"
                self.emit(kind="agent_state", agent=self.name, state="THROTTLED", turn=self.turn)
                time.sleep(1.0)
                continue
            except TransportError as exc:
                # auth / bad request / anything 4xx is not going to recover on
                # a retry; failing fast beats burning the whole iteration budget.
                self.admission.release()
                self.state = "FAILED"
                self.emit(kind="agent_state", agent=self.name, state="FAILED",
                          turn=self.turn, error=str(exc)[:300])
                break
            except Exception as exc:
                self.admission.release()
                self.state = "FAILED"
                self.emit(kind="agent_state", agent=self.name, state="FAILED",
                          turn=self.turn, error=str(exc)[:300])
                break
            self.admission.release(result.rate_rpm)

            self.total_input_tokens += result.usage.input_tokens
            self.total_output_tokens += result.usage.output_tokens
            self.emit(kind="model_usage", agent=self.name,
                      input_tokens=result.usage.input_tokens,
                      output_tokens=result.usage.output_tokens,
                      cache_read=result.usage.cache_read_input_tokens)

            # Settle signal: end of turn with no tool use. This is a coherent-
            # work boundary, not completion. A `stop` boundary is "silence" --
            # clear the processed peer injections and keep the loop going.
            # Completion is only reached via the explicit `report_done` tool.
            if not result.is_tool:
                self.state = "SETTLING"
                self.emit(kind="agent_state", agent=self.name, state="SETTLING", turn=self.turn)
                if result.stop_reason in SETTLE_SILENCE:
                    self.pending_injections.clear()
                    if self.on_settle is not None:
                        self.on_settle(self)
                continue

            # execute tool calls
            done = False
            for call in result.tool_calls:
                self.messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": call.id, "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.inputs)},
                    }],
                })
                if call.name == "report_done":
                    res = self._call_tool(call)
                    self.messages.append({"role": "tool", "tool_call_id": call.id, "content": res.content})
                    done = True
                    break
                res = self._call_tool(call)
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": res.content})
            if done:
                self.state = "DONE"
                self.emit(kind="agent_state", agent=self.name, state="DONE", turn=self.turn, files=len(self.files_touched))
                break
        else:
            self.state = "EXHAUSTED"
            self.emit(kind="agent_state", agent=self.name, state="EXHAUSTED", turn=self.turn)
        return RunResult(state=self.state, files=list(self.files_touched),
                         input_tokens=self.total_input_tokens, output_tokens=self.total_output_tokens,
                         turns=self.turn)