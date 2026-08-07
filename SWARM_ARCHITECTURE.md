# swarm v1 — System Architecture

## Context

`swarm` orchestrates 2–6 parallel LLM coding agents on isolated git worktrees and detects **interface contract conflicts** between their in-flight changes before merge. The failure mode it targets: agent A changes `authenticate(user)` → `authenticate(user, scope)`; agent B, isolated, writes a call to the old one-arg form. Both worktrees are internally consistent, both pass their own tests, git sees no textual conflict, the merge succeeds, the bug is silent until runtime.

Existing tools solve parallel execution and task-level coordination. None diff function *shape* across concurrently-modified branches. That gap is the whole product.

Two notes on state before the design:

- **`D:\projects\Swarm` is empty.** The prototype described in the brief (`models.py`, `interfaces.py`, `conflicts.py`, `git_ops.py`) is not on disk and is not a git repo. Everything below is greenfield; the prototype is treated as a design sketch to critique, which is how it was offered.
- **Confirmed decisions:** sub-agents are driven by swarm's own deterministic tool-use loop through a swappable `Transport` backend (`--provider openai|groq|openrouter|ollama|lmstudio|anthropic`), not a vendor CLI subprocess; this turn's deliverable is the design only, no implementation.

Assumptions are tagged **[A]** throughout so they can be challenged.

---

## 1. System Overview

`swarm run` decomposes a goal into N tasks, creates one git worktree + branch per task off a frozen base commit, and drives each task with an independent model tool-use loop through the selected provider's `Transport`. Every file write triggers debounced AST extraction into a per-agent *interface contract*. A detector diffs those contracts **against the base commit** — not against each other — which is what makes concurrent unordered edits tractable. Confirmed conflicts are surfaced live in a TUI and, for the asymmetric case, injected into the other agent's next turn as a mid-conversation system message so it can self-correct. At the end the user gets a merge decision with per-conflict remediation options. swarm never edits code itself and never touches the user's branch.

```
 swarm run "goal" --agents 4
        │
        ▼
 ┌──────────────┐   tasks[]   ┌──────────────────────────────────────────┐
 │ CLI /        │────────────►│ Orchestrator                             │
 │ Decomposer   │             │  • lifecycle & admission control         │
 └──────────────┘             │  • serialized git executor               │
        │                     │  • event bus (append-only JSONL)         │
   base commit B              └───┬───────────────┬──────────────────┬───┘
        │                         │               │                  │
        ▼                         ▼               ▼                  ▼
 ┌──────────────┐          ┌───────────┐   ┌───────────┐      ┌───────────┐
 │ Worktree Mgr │─wt/br───►│ Runner A  │   │ Runner B  │ ...  │ Runner N  │
 │ + manifest   │          │ tool loop │   │ tool loop │      │ tool loop │
 └──────────────┘          └─────┬─────┘   └─────┬─────┘      └─────┬─────┘
                                 │ write_file    │                  │
                                 ▼               ▼                  ▼
                          ┌────────────────────────────────────────────┐
                          │ Contract Extractor (ast, per-file, cached) │
                          └────────────────────┬───────────────────────┘
                                               │ contracts + base contract
                                               ▼
                          ┌────────────────────────────────────────────┐
                          │ Conflict Detector                          │
                          │  three-way shape merge vs base B           │
                          │  symbol-keyed index, not pairwise agents   │
                          └───────┬──────────────────────────┬─────────┘
                                  │ events                   │ CONFIRMED + settled
                                  ▼                          ▼
                          ┌───────────────┐        ┌─────────────────────────┐
                          │ TUI (reader   │        │ Cross-pollination:      │
                          │ of event log) │        │ inject peer contract    │
                          └───────────────┘        │ into caller's next turn │
                                  │                └─────────────────────────┘
                                  ▼
                          ┌────────────────────────────────────────────┐
                          │ Merge Decision Layer                       │
                          │  auto-merge · re-prompt · manual · defer   │
                          │  → swarm/<run>/integration (never main)    │
                          └────────────────────────────────────────────┘
```

---

## 2. Component Breakdown

### 2.1 CLI / Task Decomposition

**Responsibility.** Parse args, resolve and freeze the base commit, produce a task list, own the run manifest lifecycle.

```python
swarm run GOAL [--agents N] [--task TASK ...] [--base REF] [--effort LEVEL]
               [--budget-tokens N] [--no-tui] [--auto-repair {off,once,twice}]
swarm status [RUN_ID]          # projection over the event log; safe concurrently
swarm conflicts [RUN_ID]       # detail view, exit 1 if any CONFIRMED
swarm resolve RUN_ID CONFLICT_ID --strategy {take-a,take-b,reprompt,defer}
swarm merge RUN_ID [--dry-run] # build integration branch
swarm doctor                   # reconcile manifest vs `git worktree list`
swarm abort RUN_ID [--keep-branches]
swarm log RUN_ID --agent A     # replay one agent's transcript
```

Tasks come from `--task` repeated (explicit, preferred), or from one decomposition call when only `GOAL` is given:

```python
def decompose(goal: str, n: int, repo_map: str) -> list[Task]
# provider model with json_schema-structured output, e.g.
# openai "gpt-4o-mini" / groq "llama-3.3-70b-versatile" / anthropic "claude-opus-5":
#   {tasks: [{id, title, instructions, expected_paths: [str]}]}
# `expected_paths` is advisory only — used for scheduling hints and to warn on
# heavy overlap. It is NOT enforced; enforcement would just re-invent
# task-level coordination, which is a non-goal.
```

**Failure modes.** Dirty working tree at start (refuse — a dirty tree makes the base ambiguous); `--base` not an ancestor of HEAD (warn, proceed); decomposition returns overlapping or degenerate tasks (show the split, require confirmation unless `--yes`); the method call refuses output (surface and fall back to `--task` prompting).

### 2.2 Orchestrator

**Responsibility.** Agent lifecycle, API admission control, the serialized git executor, the event bus, cancellation.

```python
class Orchestrator:
    async def run(self, tasks: list[Task]) -> RunResult
    def emit(self, ev: Event) -> None          # queue + fsync'd JSONL
    async def cancel(self, reason: str) -> None # cooperative, then hard
```

Structure: one `asyncio.Task` per agent, a single `GitExecutor` for index-mutating git ops, an `ApiAdmission` gate in front of every provider request, and an `EventBus` that is the *only* channel to the TUI.

**API admission control.** All N agents share one org rate-limit bucket, so uncoordinated fan-out means 429 storms.

- A semaphore caps in-flight requests (default `min(agents, 4)`).
- A token bucket sized from the provider's `*-ratelimit-*` response headers (e.g. `anthropic-ratelimit-*`), refreshed on every response; requests wait for capacity rather than discovering it via 429.
- SDK retries raised from the default 2 to 8, with jitter; `retry-after` honored verbatim on 429.
- **Cache-warm-then-fan-out.** All agents share a byte-identical prompt prefix (§2.3). Concurrent requests with the same prefix *all* miss, because an entry is only readable once the first response begins streaming. So: fire one warmup request (`max_tokens=0` against the shared prefix), await first byte, then release the fleet. One cache write instead of N.

**Cancellation.** `SIGINT` → `cancel()`: stop issuing new API requests, let in-flight tool calls finish (a half-written file is worse than a slow exit), commit each agent's settled work to its branch, mark the manifest `aborted`, leave worktrees for `swarm doctor`. Second `SIGINT` within 3s → hard kill, manifest keeps `intent` records so nothing orphans.

**Failure modes.** One agent's unhandled exception must not cancel the run (`asyncio.gather(..., return_exceptions=True)`, per-agent `FAILED` state); event queue backpressure (bounded queue, drop *render* events under pressure, never drop contract/conflict events); clock skew is irrelevant by construction, since nothing in the model depends on wall-clock ordering.

### 2.3 Agent Runner

One per agent. Owns the conversation, the tool loop, and the token budget.

**Loop shape.** The Transport interface (§2.2) turns the LLM into a typed contract for swarm: canonical OpenAI chat-completions messages and a canonical tool list. Each backend adapter (`OpenAICompatTransport` for OpenAI/Groq/OpenRouter/Ollama/vLLM via stdlib HTTP, `AnthropicTransport` for the Anthropic Messages API, `MockTransport` for tests) translates to its own wire format. The loop itself is a hand-written `while tool_calls present` loop — see `runner.py` — with the extraction trigger living *inside* the `write_file` tool function and the settle/inject decisions between iterations.

```python
@beta_tool
def read_file(path: str, start: int = 1, end: int | None = None) -> str: ...
@beta_tool
def write_file(path: str, content: str) -> str: ...      # → enqueue extraction
@beta_tool
def edit_file(path: str, old: str, new: str) -> str: ...  # → enqueue extraction
@beta_tool
def list_dir(path: str = ".", depth: int = 2) -> str: ...
@beta_tool
def grep(pattern: str, glob: str = "**/*.py") -> str: ...
@beta_tool
def run_shell(command: str, timeout_s: int = 120) -> str: ...
@beta_tool
def report_done(summary: str, files_changed: list[str]) -> str: ...
```

`run_shell` is the escape hatch (tests, linters, `git`); the dedicated file tools exist because swarm needs a typed hook on writes, which an opaque shell string cannot provide. Every `path` is resolved to its canonical form and rejected unless it stays under the agent's worktree root — the sandbox boundary, non-negotiable.

**"Done" is a tool call, not a heuristic.** `report_done` is explicit and unambiguous; a turn ending without tool use is treated as a *turn boundary* (the settle signal, §3), not as completion. Belt-and-braces stops: `max_iterations` (default 40), a token `task_budget`, and a wall-clock ceiling.

**Model config.** Provider- and model-dependent — selected via `--provider`/`--model` (or `SWARM_PROVIDER`/`SWARM_MODEL`). Reasonable defaults per provider:

```python
# OpenAI / OpenAI-compatible (OpenAICompatTransport)
provider="openai"; model="gpt-4o-mini"; max_tokens=64000
# Groq  provider="groq";  model="llama-3.3-70b-versatile"
# Ollama provider="ollama"; model="qwen2.5-coder:latest"   # keyless, localhost
# Anthropic (AnthropicTransport)
provider="anthropic"; model="claude-opus-5"; thinking_effort high; max_tokens=64000
```

Where a provider runs elevated safety guards and can refuse a turn, check the refusal before reading `content`, surface it, and do not retry blindly.

**Prompt caching design.** All agents share one prefix, so the cache pays N-fold.

| Position | Content | Stability |
|---|---|---|
| `tools` | fixed tool list, sorted by name | frozen for the run |
| `system[0]` | agent instructions, contract-awareness rules | frozen |
| `system[1]` | repo map + base commit digest, provider cache breakpoint here | frozen per run |
| `messages[0]` | this agent's task | per-agent |
| `messages[n]` | turns, tool results | per-turn |

No timestamps, no run IDs, no agent IDs, no `datetime.now()` anywhere in `system`. Tools rendered in sorted order. Peer-contract injections go in as `{"role": "system"}` **messages** — appending after the cached history preserves the prefix, where editing top-level system would re-bill every prior turn. Verify with the provider's cached-token usage (e.g. `usage.cache_read_input_tokens` on Anthropic, `prompt_cache_hit_tokens` on DeepSeek, `prompt_tokens_details.cached_tokens` on OpenAI-compatible); if it stays zero across turns, something in the prefix is varying.

**Failure modes.** Rate limit / timeout → admission gate + backend retry, agent state `THROTTLED`, no work lost. Tool exception → return an error so the model adapts, never crash the loop. `run_shell` hang → timeout, truncated output. Context growth → provider-managed or agent-side compaction of the transcript, appending the full response so nothing is lost. Model loops on one file → `max_iterations` trips, state `EXHAUSTED`, partial work preserved on the branch.

### 2.4 Worktree Manager

**Critique of the sketched `git_ops.py`.** Subprocess wrapping is right; the crash story is not.

1. **Orphans.** `git worktree add` then crash leaves both a directory and a `.git/worktrees/<name>` admin entry. `git worktree prune` will not clean a worktree with local commits or a dirty tree — precisely the state a crashed agent leaves. Result: silent disk growth and stale entries that make later `worktree add` fail.
2. **No intent log.** Nothing on disk records that swarm was *about to* create a worktree, so post-crash reconciliation has no ground truth.
3. **Unserialized concurrent git.** N agents plus the orchestrator invoking git against one repo contends on `index.lock` and on ref updates. Worktrees have separate indexes, which helps, but shared refs/objects do not.
4. **Destructive cleanup.** `worktree remove --force` on a crashed agent throws away the only copy of its work.

**Redesign.**

```python
class WorktreeManager:
    def preflight(self, n: int) -> None                    # disk headroom check
    def create(self, run: RunId, agent: AgentId) -> Worktree
    def checkpoint(self, wt: Worktree, msg: str) -> Sha     # per settled turn
    def changed_files(self, wt: Worktree) -> list[Path]     # vs base, read-only
    def release(self, wt: Worktree, keep_branch: bool = True) -> None
    def reconcile(self) -> ReconcileReport                  # `swarm doctor`
```

- **Manifest first, git second.** `.git/swarm/runs/<run_id>.json`, fsync'd. Write `{"state": "intent", ...}` → run `git worktree add` → rewrite `{"state": "active"}`. A crash anywhere is recoverable because the intent record exists before the side effect.
- **Branches are the durable artifact; worktree dirs are disposable.** Auto-checkpoint-commit each agent's work at every settled turn onto `swarm/<run>/<agent>`. A crash then loses at most one turn, and per-turn diffing and rollback come free.
- **Serialized git executor.** All index/ref-mutating ops go through one worker. Read-only ops (`diff --name-only`, `cat-file`, `rev-parse`) run in parallel. Every invocation is an argv list with `shell=False`.
- **`reconcile()` is a first-class command.** Diff manifest against `git worktree list --porcelain`: orphan dir with commits → keep the branch, offer to prune the dir; orphan dir clean → prune; manifest entry with no dir → clear the entry; unexpected `swarm/*` branch → report, never delete.
- **Disk preflight.** Worktrees share the object store, so the marginal cost is the working tree only. Estimate `n × working_tree_bytes`, refuse below 3× headroom, re-check between agent starts.
- **[A]** Internal checkpoint commits use `--no-verify`. User pre-commit hooks on a 40-turn agent run are slow and can fail on intentionally-transient states. This is a deliberate choice, not an oversight, and it is worth challenging — hooks *would* catch some breakage early. It applies only to swarm's own checkpoints; an agent invoking `git commit` via `run_shell` is unaffected.

### 2.5 Contract Extractor

**Critique of the sketched `interfaces.py`.** `ast.NodeVisitor` over `ast.parse` is the right foundation — it gives string and comment immunity for free, which is why AST beats any text-based approach here. The blind spots are all in resolution, and each one is a false-positive generator:

| Blind spot | Naive behavior | Consequence |
|---|---|---|
| Decorators | ignored | `@app.route`, `@click.command`, `@contextmanager` rewrite arity → phantom mismatches |
| `*args` / `**kwargs` | counted as one param each | every forwarding wrapper looks incompatible |
| Keyword-only / posonly (`*`, `/`) | flattened into positional | a legal call reads as illegal |
| Defaults | ignored | adding an optional param reads as a breaking change — the single most common FP |
| Class inheritance | absent | `Sub().m()` unresolvable; `Base.m` shape changes miss overrides |
| Attribute calls | name-only match | `a.save()` vs `b.save()` collapse into one symbol |
| Imports / re-exports | absent | `requests.get()` collides with a local `get()`; a re-export hides the real definition |
| Dynamic dispatch | absent | `getattr(o, name)()` invisible, `functools.wraps` misleading |

**Redesign.** Two-layer: syntactic extraction, then a resolution pass. Resolution is the layer that decides whether a finding is trustworthy, and its default is *silence*.

```python
@dataclass(frozen=True)
class Shape:
    positional: tuple[str, ...]     # posonly + pos-or-kw, in order
    posonly_count: int
    required_positional: int        # leading params without defaults
    kwonly_required: frozenset[str]
    kwonly_optional: frozenset[str]
    has_varargs: bool
    has_kwargs: bool
    is_method: bool
    implicit_self: bool             # False for @staticmethod
    decorators: tuple[str, ...]
    arity_opaque: bool              # unknown decorator ⇒ do not trust arity

@dataclass(frozen=True)
class SymbolKey:                    # the join key for the whole system
    module: str                     # repo-relative dotted path
    qualname: str                    # "Class.method" / "func" / "Outer.Inner.m"

@dataclass(frozen=True)
class CallSite:
    key: SymbolKey | None            # None ⇒ unresolved
    n_positional: int
    keywords: frozenset[str]
    has_star_args: bool              # f(*xs)  ⇒ arity unknown
    has_star_kwargs: bool            # f(**kw) ⇒ keywords unknown
    file: Path
    line: int
    in_test: bool
    confidence: Literal["exact", "heuristic", "unresolved"]
```

Resolution rules, ordered by how much false-positive risk they remove:

1. **Import table per file.** Build `{local_name → (module, qualname)}` from all `Import`/`ImportFrom`. A bare `Name` call resolves through it. If the target module is stdlib or site-packages (not under repo root) → **drop the call site entirely**. This alone kills the third-party-name-collision class.
2. **Arity-transparent decorator allowlist.** `functools.wraps`, `staticmethod`, `classmethod`, `property`, `abstractmethod`, `override`, `functools.lru_cache`, `typing.overload`. Anything else sets `arity_opaque=True`: definition-vs-definition comparison still runs (a decorator or name change is real signal) but **all call-site checks against that symbol are suppressed**.
3. **`*args`/`**kwargs` are lattice tops, not counts.** `has_varargs` means unbounded positional capacity; `has_kwargs` means any keyword accepted. On the call side, `f(*xs)` makes arity unknown → `confidence="unresolved"`, never a conflict.
4. **Intra-repo MRO.** Resolve `Sub.m` through base classes defined in the repo. A class with any out-of-repo base is `mro_incomplete` → override checks suppressed.
5. **Attribute calls — the hard one, and the biggest FP risk.** No type inference in v1. `obj.method()` resolves only if `method` is defined by **exactly one** class in the repo. Two or more candidates → `confidence="unresolved"`. **[A]** This heuristic is the largest single risk to product precision; §8 makes it the first spike.
6. **Unresolved is a visible bucket, never a conflict.** `swarm conflicts --unresolved` shows what could not be reasoned about, so the tool's blind spots are legible instead of silently reported as clean.

Performance: cache per `(path, mtime_ns, size)`, verified by content hash on hit. Only files git reports changed vs base are re-parsed. `SyntaxError` → retain the last good contract for that file, mark `dirty_unparseable`, emit nothing.

### 2.6 Conflict Detector

**The last-writer-wins heuristic in the sketch is wrong, and not marginally so.** Three reasons:

1. **Wall-clock finish order carries no semantic information.** Which agent's API request returned first is a function of token counts, retry luck, and rate-limit queuing. LWW picks a winner from noise.
2. **It silently discards the signal.** The interesting fact is *"two agents independently made incompatible decisions about the same interface"* — a thing a human must adjudicate. LWW converts that into "B won," suppressing exactly the output the product exists to produce.
3. **It has no base, so it cannot tell the two cases apart.** "A evolved an interface and B needs updating" and "A and B disagree" are structurally different problems with different remediations, and pairwise comparison of A against B cannot distinguish them.

**The fix is not CRDTs or vector clocks.** Those model causality, and here there is none by construction: agents never observe each other, so all edits are concurrent and every clock is identical. The right model is the one git already uses for text, lifted to contracts: **three-way merge against the common ancestor**. Every agent branches from the same frozen base commit `B`, so `base_shape` is always available for any pre-existing symbol — that is the piece the sketch was missing, and having it makes the whole problem well-posed.

```python
def build_index(base: Contract, agents: dict[AgentId, Contract]) -> Index
# Index: SymbolKey → {definers: {agent: Shape}, base: Shape | None,
#                     callers: {agent: [CallSite]}, base_callers: [CallSite]}

def detect(idx: Index) -> list[Conflict]
```

Only symbols with ≥2 definers, or with one definer plus a foreign caller, need any work.

### 2.7 Live Dashboard / TUI

**Responsibility.** Render run state. Zero coupling to orchestrator internals.

**Strategy.** The orchestrator emits to an append-only fsync'd JSONL event log; the TUI is a *consumer* that folds events into its own view model. `rich.Live` at 8 Hz — the terminal, not the agent, is the render bottleneck, so decoupling render rate from write rate is mandatory. Render functions read only the projection and compute nothing.

The log-as-interface choice pays twice: `swarm status` in a second terminal is the same projection over the same file, so no IPC, no shared memory, no server. And a run is fully replayable after the fact for debugging.

```
swarm · run 7f3a · base 9c1e4d2 · 4 agents · 6m12s ────────────── 3.2M tok · $18.40
┌ agents ─────────────────────────────────────────────────────────────────────┐
│ A auth-scopes    ● RUNNING  t14  edit_file services/auth.py      3 files     │
│ B rate-limiter   ● RUNNING  t9   run_shell pytest -q             5 files     │
│ C admin-ui       ◐ SETTLED  t22  report_done                     8 files     │
│ D metrics        ⏸ THROTTLED retry in 4s (429)                   2 files     │
└─────────────────────────────────────────────────────────────────────────────┘
┌ conflicts ──────────────────────────────────────────────────────────────────┐
│ ✖ CONFIRMED  broken_caller  auth.authenticate                  A def → C call│
│    base (user)  A (user, scope)  ·  C calls authenticate(u) @ admin/views:88 │
│    → repair sent to C (round 1/2)                                           │
│ ⚠ TENTATIVE  divergent_def   cache.Cache.get                   B def, D def  │
│    B (key, default=None)  D (key, *, ttl)  ·  provisional, B mid-turn        │
│ ○ 3 unresolved call sites (attribute dispatch) — swarm conflicts --unresolved│
└─────────────────────────────────────────────────────────────────────────────┘
```

Information density target: every agent on one line (state, turn count, current tool, files touched); every conflict in two lines (verdict + kind + symbol + parties, then the three shapes and the breaking call site); a persistent count of unresolved sites so the tool's blind spots stay visible. Cost and token totals in the header, because they are the thing a user actually watches.

**Failure modes.** Not a TTY → `--no-tui` auto-detected, line-oriented output. Terminal too narrow → drop columns before wrapping. Log grows unbounded → rotate per run; the projection is incremental so size does not affect render cost.

### 2.8 Merge Decision Layer

**Principle: swarm never edits code.** All remediation is either (a) re-prompt an agent, or (b) pick one branch's version wholesale. Auto-rewriting a signature that swarm did not author is how a conflict detector becomes a fourth agent writing code nobody reviewed, and it is where the blast radius stops being bounded.

Remediation by kind:

| Kind | Default | Mechanism |
|---|---|---|
| `convergent` | auto | no action; branches are compatible |
| `subsumable` | auto with notice | keep the more permissive definer's version of that symbol; re-prompt the other agent only if its call sites break |
| `broken_caller` | auto-repair, bounded | inject the definer's before/after contract + breaking call sites into the caller agent's next turn as a `{"role": "system"}` message; max 2 rounds (`--auto-repair`), then escalate |
| `divergent_def` | manual | present base/A/B shapes and the call sites each breaks; offer take-a / take-b / reprompt-both / defer |
| `override_divergence` | manual | show base-class change against the override |
| `removed_symbol` | manual | show the removal against surviving callers |

`swarm merge` builds `swarm/<run>/integration` with sequential `git merge --no-ff`, in ascending order of remaining conflict severity so the cleanest branches land first. Then it re-runs extraction **on the merged tree** and reports any residual conflict — the tree is the ground truth, and detection on the merge result is the honest final check. It never merges into the user's branch and never pushes; the user integrates.

`--dry-run` reports what would merge, what textual conflicts git would raise, and residual contract conflicts, without creating the branch.

---

## 3. Data Flow and State Machine

### 3.1 SubAgent lifecycle

| From | To | Trigger |
|---|---|---|
| `PENDING` | `PROVISIONING` | scheduler admits agent |
| `PROVISIONING` | `RUNNING` | worktree + branch created, manifest `active` |
| `PROVISIONING` | `FAILED` | worktree creation failed (disk, git, name collision) |
| `RUNNING` | `SETTLING` | assistant turn ends with no `tool_use` |
| `SETTLING` | `RUNNING` | next user/system message sent (repair injection or continuation) |
| `SETTLING` | `DONE` | `report_done` called |
| `RUNNING` | `THROTTLED` | 429 / timeout inside admission or backend retry |
| `THROTTLED` | `RUNNING` | retry succeeds |
| `THROTTLED` | `FAILED` | retries exhausted |
| `RUNNING` | `EXHAUSTED` | `max_iterations`, task budget, or wall-clock ceiling hit |
| `RUNNING` | `FAILED` | unhandled exception in the loop |
| `RUNNING`/`SETTLING` | `CANCELLING` | user abort |
| `CANCELLING` | `ABORTED` | in-flight tool finished, work checkpointed |
| `DONE`/`EXHAUSTED`/`FAILED`/`ABORTED` | — | terminal; branch retained |

`DONE`, `EXHAUSTED`, and `ABORTED` are all *mergeable* — partial work on a branch is still work. Only `FAILED` before the first checkpoint yields nothing.

### 3.2 When extraction runs

Neither "on every write" (mid-edit files are broken and produce noise) nor "on commit only" (too late for a live dashboard, which is a stated requirement). The answer is **debounced post-write extraction with a two-tier confidence gate**.

```
write_file / edit_file returns
        │
        ▼  enqueue(path), debounce 750ms per agent   ← coalesces edit bursts
   parse changed files (mtime+hash cached)
        │
        ├─ SyntaxError → keep last good contract, mark dirty_unparseable, emit nothing
        ▼
   per-agent contract updated, tagged PROVISIONAL
        │
        ▼
   detect() → conflicts tagged TENTATIVE
              · shown in TUI as ⚠
              · never injected into an agent
              · never escalated to merge decisions
        │
   assistant turn ends with no tool_use  (the settle signal)
        │
        ▼
   contract → SETTLED ; checkpoint commit ; re-run detect()
        │
        ▼
   conflicts between two SETTLED contracts → CONFIRMED
              · TUI ✖ · eligible for injection and merge decisions
```

**Why the turn boundary is the settle signal.** It is the point at which the agent has, by its own judgment, completed a coherent unit of work — the natural quiescence marker in a tool-use loop. Filesystem quiescence alone is not enough: an agent commonly writes a new caller in one tool call and the callee in the next, and a purely time-based gate reports that gap as a conflict. **[A]** This is the load-bearing assumption of the whole live-detection design and §8 makes it the second spike.

**Handling an agent's own not-yet-finished work.**

- **Intra-agent findings are never cross-agent conflicts.** If A's call site disagrees with A's own definition, that is A's business. It is surfaced as a hint in A's next turn, not as a conflict — cheap self-repair, zero user-facing noise.
- Only `PROVISIONAL × PROVISIONAL` and `PROVISIONAL × SETTLED` produce `TENTATIVE`; `SETTLED × SETTLED` produces `CONFIRMED`.
- A symbol an agent has deleted but not yet re-added reads as `removed_symbol` while provisional; on settle it either resolves or promotes.

---

## 4. The Conflict Detection Algorithm (Core IP)

### 4.1 Shape compatibility

Not arity matching. The primitive is a directional predicate: *does this shape accept this call?*

```python
def accepts(s: Shape, c: CallSite) -> Verdict:  # ACCEPT | REJECT | UNKNOWN
    if c.has_star_args or c.has_star_kwargs or s.arity_opaque:
        return UNKNOWN                      # bias to silence

    slots = list(s.positional)
    if s.implicit_self and c.key_is_bound_call:
        slots = slots[1:]                   # receiver supplies self

    # 1. positional capacity
    if c.n_positional > len(slots) and not s.has_varargs:
        return REJECT
    consumed = set(slots[:c.n_positional])

    # 2. keyword admissibility
    nameable = set(slots[s.posonly_count:]) | s.kwonly_required | s.kwonly_optional
    for k in c.keywords:
        if k in consumed:
            return REJECT                   # duplicate value for one param
        if k not in nameable and not s.has_kwargs:
            return REJECT                   # unexpected keyword

    # 3. required params all filled
    required_pos = set(slots[:s.required_positional])
    if required_pos - consumed - c.keywords:
        return REJECT
    if s.kwonly_required - c.keywords:
        return REJECT

    return ACCEPT
```

Defaults, `*args`, `**kwargs`, keyword-only, positional-only, and `self` binding all fall out of this rather than needing special cases. Adding a parameter *with* a default is `ACCEPT` for every old call — which is what makes the most common evolution not a conflict.

On top of `accepts`, the definition-vs-definition relation:

```python
def relation(old: Shape, new: Shape) -> Relation:
    # IDENTICAL | WIDENED | NARROWED | RENAMED_COMPATIBLE | DIVERGED
```

Computed structurally in both directions — *is every call `old` accepts also accepted by `new`?* — via:

1. **Positional prefix preserved.** `new.positional[:len(old.positional)] == old.positional`, or `new.has_varargs`. A rename inside the prefix breaks keyword calls even though positional calls survive; when positions and count match and only names differ, the relation is `RENAMED_COMPATIBLE` — a distinct, lower-severity outcome, because whether it breaks anything depends entirely on whether real call sites use keywords.
2. **Capacity non-decreasing.** `len(new.positional) ≥ len(old.positional)` or `new.has_varargs`.
3. **Requirements non-increasing.** `new.required_positional ≤ old.required_positional`; `new.kwonly_required ⊆ old.kwonly_required`. A *new required* parameter is the canonical breaking change.
4. **Keyword surface non-shrinking.** `old_nameable ⊆ new_nameable` or `new.has_kwargs`.
5. **Posonly boundary not advanced.** `new.posonly_count ≤ old.posonly_count`.

Both directions hold → `IDENTICAL`. Forward only → `WIDENED` (new is strictly more permissive). Reverse only → `NARROWED`. Neither → `DIVERGED`.

`return_hint` is recorded but **not** part of compatibility. Return-type checking is a type-checker's job and a non-goal; treating it as a conflict source would generate noise for zero coordination value.

### 4.2 Legitimate evolution vs incompatible independent decisions

This is the distinction LWW cannot make, and it is entirely determined by *how many agents touched the definition*.

```python
for key, e in index.items():
    definers = e.definers                       # {agent: Shape}
    base     = e.base                           # Shape | None

    # ── Case 1: exactly one definer ⇒ EVOLUTION ──────────────────────────
    if len(definers) == 1:
        (owner, new), = definers.items()
        if base is not None and relation(base, new) in (IDENTICAL, WIDENED):
            continue                            # backward compatible; nothing to do
        for agent, sites in e.callers.items():
            if agent == owner:
                continue                        # intra-agent; hint, not conflict
            for cs in sites:
                if accepts(new, cs) is REJECT:
                    yield Conflict(BROKEN_CALLER, key, definer=owner, caller=agent,
                                   base=base, new=new, site=cs,
                                   severity=HIGH if not cs.in_test else MEDIUM)
        for cs in e.base_callers:               # pre-existing untouched callers
            if accepts(new, cs) is REJECT:
                yield Conflict(BROKEN_BASE_CALLER, key, definer=owner, site=cs,
                               severity=MEDIUM) # owner's own tests likely catch it

    # ── Case 2: two or more definers ⇒ CONCURRENT DECISION ───────────────
    else:
        shapes = set(definers.values())
        if len(shapes) == 1:
            kind = CONVERGENT                   # identical independent change
        else:
            rels = {relation(a, b) for a in shapes for b in shapes if a is not b}
            if rels <= {IDENTICAL, WIDENED, NARROWED}:
                kind = SUBSUMABLE               # a total order exists; pick the top
            elif rels <= {IDENTICAL, WIDENED, NARROWED, RENAMED_COMPATIBLE}:
                kind = RENAME_DISAGREEMENT
            else:
                kind = DIVERGENT_DEFINITION     # genuine adjudication needed
        yield Conflict(kind, key, definers=definers, base=base,
                       severity=severity_for(kind, e))
```

The two cases carry different remediations because they are different problems:

- **One definer** — someone owns the interface. It evolved. Every other agent's disagreeing call site is *stale*, and the fix is mechanical: tell the caller what the contract is now. This is auto-repairable and is the flagship behavior.
- **Two definers** — nobody owns it. Two agents independently decided what the interface should be. `SUBSUMABLE` still has a mechanical answer (a total order exists — take the most permissive, provided every call site is accepted by it). `DIVERGENT_DEFINITION` does not: there is no information anywhere in the system that says which decision is correct. Escalate to the human with all three shapes and the breaking call sites, and *do not guess*. Guessing here is precisely LWW's mistake.

The same structure covers deletion: `base is not None` and no definer retains the symbol → `REMOVED_SYMBOL`, checked against every surviving caller.

Additionally, over the intra-repo MRO: if agent A changes `Base.m` and agent B defines or modifies `Sub.m`, compare the two. `relation(A_base_m, B_sub_m) == DIVERGED` → `OVERRIDE_DIVERGENCE`. Suppressed when `mro_incomplete`.

### 4.3 Minimizing false positives

Precision over recall, unconditionally. A tool that cries wolf gets disabled in a week; one that misses some conflicts is still strictly better than the current state of nothing.

| Source | Guard |
|---|---|
| Strings, comments, docstrings | free — `ast` never sees them. The reason not to use text matching or tree-sitter-as-text |
| Third-party name collision | import table resolution; any target outside the repo root is **dropped**, not compared |
| Re-exports (`from .impl import f` in `__init__`) | follow re-export chains within the repo to the defining module; `SymbolKey` is always canonical |
| `*args` / `**kwargs` forwarding | lattice tops, never counts; `f(*xs)` → `UNKNOWN` |
| Unknown decorators | `arity_opaque=True` → call-site checks suppressed for that symbol |
| Attribute calls | resolved only when the method name has exactly one repo-wide definition; otherwise `unresolved` **[A]** |
| Dynamic dispatch (`getattr`) | not modeled → `unresolved` |
| Out-of-repo base classes | `mro_incomplete` → override checks suppressed |
| Test fixtures | **not** excluded — a broken test caller is real breakage. Tagged `in_test`; severity `MEDIUM`; filterable. **[A]** Excluding them outright would hide genuine conflicts in the code most likely to exercise an interface |
| Agent's own in-progress work | `PROVISIONAL` gate; intra-agent findings are hints, never conflicts |
| Unparseable files | last good contract retained, nothing emitted |
| Backward-compatible evolution | `relation(base, new) ∈ {IDENTICAL, WIDENED}` short-circuits before any call-site scan — kills the "added an optional param" FP class |

Every suppression is *counted and reported*. `swarm conflicts --unresolved` lists what could not be reasoned about, with the reason. A tool whose blind spots are invisible is worse than one whose blind spots are documented.

### 4.4 Complexity and scalability

The pairwise framing is the wrong thing to worry about. With `n ≤ 6`, `n²/2 = 15` — noise.

| Stage | Cost | Where it hurts |
|---|---|---|
| Base call-site index | `O(F)` parse, **once** at run start, cached on disk by tree SHA | 5k-file repo cold start: single-digit seconds. Acceptable once, not per tick |
| Incremental extraction | `O(Δfiles × filesize)` per debounce tick | **This is the real wall.** Naive full-repo re-parse per write is `O(F)` at ~2 Hz × 6 agents |
| Index build | `O(S)` in symbols touched | negligible |
| Detection | `O(Σ candidate_symbols × callers)` | negligible; only symbols with ≥2 definers or a foreign caller are examined |

So the fix is to make extraction incremental, which the design already does: parse only `git diff --name-only` output against base, cache on `(path, mtime_ns, size)` + content hash, 750ms debounce to coalesce bursts. The symbol-keyed inverted index means detection never iterates agent pairs at all — it iterates *symbols with multiple writers*, which in a well-partitioned run is a handful.

Where the naive design genuinely breaks, stated plainly: a monorepo where the *base* call-site index is 50k+ files, because the one-time cost stops being one-time if the cache is keyed wrongly. Mitigation: key the base index on tree SHA so it is shared across runs on the same commit, and make it lazy — build the base call-site index for a module only when some agent actually modifies a definition in it. Most runs then never pay for most of the repo.

Growth beyond ~10 agents would need the extraction pool moved to a process pool (`ast.parse` is CPU-bound and holds the GIL). Out of scope for v1's 2–6 target, but the debounce-queue boundary is the natural seam for it.

---

## 5. Failure Modes and Recovery

| Scenario | Detection | Recovery |
|---|---|---|
| Agent process/task crashes mid-task | per-agent `asyncio.Task` exception; run continues | state `FAILED`; work up to the last settled checkpoint is on the branch; worktree retained; other agents unaffected; branch still mergeable |
| API rate limit (429) | admission token bucket, then backend retry with `retry-after` | state `THROTTLED`; exponential backoff + jitter on the backend; conversation state intact so no work is lost; TUI shows countdown |
| API timeout / connection error | `APITimeoutError` / `APIConnectionError` | same retry path; if the last turn's tool calls already ran, the assistant turn is re-requested — tools are idempotent by design (`write_file` is a full overwrite) |
| The output call prefers `stop_reason == "refusal"`-style refusals | checked **before** reading `content` | surface the refusal and the task, mark `FAILED`, do not retry blindly |
| Model loops / burns budget | `max_iterations`, `task_budget`, wall-clock ceiling | state `EXHAUSTED`; partial work checkpointed and mergeable; TUI flags it distinctly from `FAILED` |
| Git worktree corruption / stale admin entry | `swarm doctor` diffs manifest against `git worktree list --porcelain` | dir with commits → keep branch, offer dir prune; dir clean → prune; manifest-only entry → clear; unexpected `swarm/*` branch → report, never delete |
| `index.lock` contention | git exits non-zero with a lock message | serialized git executor makes it near-impossible by construction; residual case retries with backoff up to 5s, then surfaces |
| Two agents both crash after partial writes | two `FAILED` states; both worktrees retained | each branch holds its last settled checkpoint; detection still runs on the two settled contracts, so **conflicts between two crashed agents are still reported** — nothing about detection depends on agents finishing |
| Unparseable file at crash time | `SyntaxError` during extraction | last good contract retained; file marked `dirty_unparseable`; listed in `swarm conflicts --unresolved`; never silently treated as clean |
| Ctrl-C during the run | `SIGINT` handler → `CANCELLING` | stop new API requests; let in-flight tool calls finish; checkpoint every agent; manifest `aborted`; worktrees + branches retained; `swarm status` still works. Second `SIGINT` within 3s → hard kill; intent records prevent orphans |
| Ctrl-C during merge | `SIGINT` in `swarm merge` | merge targets `swarm/<run>/integration` only; `git merge --abort` on partial state; user branches never touched |
| Disk full from worktree proliferation | preflight `shutil.disk_usage` before each agent start; `OSError ENOSPC` during writes | refuse to start below 3× estimated headroom; mid-run `ENOSPC` → pause all agents, checkpoint, prompt user to free space or abort; `swarm doctor --prune` reclaims dirs from completed runs while keeping branches |
| Event log write fails | `OSError` on append | degrade to in-memory projection, warn once; never crash a run over telemetry |
| Base commit moves under the run | `rev-parse` at start, re-verified at merge | base SHA is frozen in the manifest; a mismatch at merge time is an error with a clear message, not a silent three-way merge against the wrong ancestor |

---

## 6. CLI and Terminal UX Spec

### 6.1 Clean run

```
$ swarm run "add scoped auth + rate limiting" --agents 3

swarm 0.1.0 · base 9c1e4d2 (main) · run 7f3a
  indexing base call sites … 1,284 files, 3,902 symbols (cached)
  decomposing goal … 3 tasks
    A  auth-scopes      services/auth.py, services/session.py
    B  rate-limiter     middleware/, config/
    C  admin-ui         admin/views.py, admin/templates/
  creating worktrees … .swarm/7f3a/{A,B,C}
  warming prompt cache … 1 write, 18,204 tok

[live dashboard]

── run 7f3a complete · 8m41s ─────────────────────────────────────────────────
  A  DONE  22 turns  4 files  412 lines   ✔ pytest 118 passed
  B  DONE  17 turns  6 files  308 lines   ✔ pytest 118 passed
  C  DONE  25 turns  9 files  671 lines   ✔ pytest 118 passed

  contracts   47 definitions changed · 213 call sites checked
  conflicts   0 confirmed · 0 tentative
  unresolved  4 call sites (attribute dispatch) — swarm conflicts 7f3a --unresolved
  tokens      3.9M in / 214k out · cache hit 91% · $21.30

  no contract conflicts detected.
  → swarm merge 7f3a          builds swarm/7f3a/integration
  → swarm merge 7f3a --dry-run
```

### 6.2 Conflict caught live

```
── conflict detected · 4m02s ─────────────────────────────────────────────────
✖ CONFIRMED  broken_caller        services.auth:authenticate          HIGH

  base   authenticate(user)
  A      authenticate(user, scope)          services/auth.py:41   [definer]
                     └─ new required positional parameter

  C calls the old form:
    admin/views.py:88    authenticate(request.user)
    admin/views.py:203   authenticate(u)
    tests/test_admin.py:31  authenticate(fake_user)          [test]

  A owns this interface (sole definer) → this is interface evolution.
  Repairing C automatically (round 1 of 2).
```

Then, injected into C's next turn as a `{"role": "system"}` message — not a rewrite of the top-level system prompt, so C's cached history survives:

```
Peer agent A changed an interface you call, in a worktree you cannot see.

  services.auth.authenticate
    was:  authenticate(user)
    now:  authenticate(user, scope)   # scope is required

Your call sites that no longer type-check:
  admin/views.py:88, admin/views.py:203, tests/test_admin.py:31

Update them. If you believe `scope` should be optional, say so and stop —
do not change services/auth.py; it belongs to another agent.
```

Follow-up, after C settles:

```
✔ RESOLVED  broken_caller  services.auth:authenticate   (C, round 1)
  admin/views.py:88   authenticate(request.user, scope="admin")
  admin/views.py:203  authenticate(u, scope="admin")
  tests/test_admin.py:31  authenticate(fake_user, scope="admin")
```

And the case with no mechanical answer:

```
✖ CONFIRMED  divergent_def       cache.Cache.get                     HIGH

  base   get(self, key)
  B      get(self, key, default=None)      services/cache.py:56
  D      get(self, key, *, ttl)            services/cache.py:56

  Neither shape accepts every call the other does. Two agents made
  independent, incompatible decisions; swarm will not choose for you.

  call sites affected
    B  services/queue.py:14   get(k, default=[])      ✔ B   ✖ D
    D  metrics/collect.py:77  get(k, ttl=60)          ✖ B   ✔ D

  swarm resolve 7f3a c2 --strategy take-a        keep B's shape, re-prompt D
  swarm resolve 7f3a c2 --strategy take-b        keep D's shape, re-prompt B
  swarm resolve 7f3a c2 --strategy reprompt      send both agents both shapes
  swarm resolve 7f3a c2 --strategy defer         merge anyway, record in report
```

**Density rules for the live view.** One line per agent, two per conflict, unresolved count always visible, cost always visible. Nothing scrolls off: the conflict pane is scrollable and the agent pane is fixed-height. Colour is redundant with glyphs (`● ◐ ⏸ ✖ ⚠ ○ ✔`) so it works piped and on monochrome terminals. `--json` on every read command for scripting.

---

## 7. What v1 Deliberately Excludes

| Excluded | Why deferring is safe |
|---|---|
| **Non-Python languages** | The `Shape` model and the three-way-merge algorithm are language-agnostic; only the extractor is not. Keep `ContractExtractor` behind a one-method protocol (`extract(path, source) -> FileContract`) and stop there. Building a tree-sitter abstraction layer before the second language exists is designing for a requirement that has not arrived |
| **Type-level checking (parameter types, return types, generics)** | That is a type-checker's job and an explicit non-goal. Shape catches the documented failure mode; types are a superset that `pyright` already delivers better. See §8 — this exclusion is also the design's biggest open question |
| **Cross-module call resolution via type inference** | The unique-method-name heuristic + import tables cover direct calls. Real inference means a points-to analysis, which is a research project, and its absence is *visible* in the unresolved bucket rather than silent |
| **Multi-machine / distributed execution** | Single dev machine is the stated target. The manifest and event log are already file-based, so a future coordinator has clean seams, but no network protocol, no service, no auth to design now |
| **Task-level coordination (shared to-do lists, ownership negotiation, A2A messaging)** | Explicit non-goal — prior art solves it. swarm's one channel between agents is a contract injection, and keeping it to one channel is what keeps swarm composable with the tools that do own task coordination |
| **Automatic code repair by swarm itself** | swarm proposes, agents edit. A detector that also writes code becomes an unreviewed fourth agent and its blast radius stops being bounded |
| **IDE / editor integration** | Terminal-native is a constraint. The JSONL event log is a trivially consumable interface if an extension ever wants one |
| **Merging into the user's branch, or pushing** | swarm builds an integration branch and stops. Integration is a human decision with a human's context |
| **Semantic (behavioural) conflict detection** | Two agents changing what a function *does* while keeping its shape is undetectable by any static method and is out of scope by construction. Say so in the docs rather than implying coverage |
| **>6 agents** | The extraction path is single-process. The debounce queue is the seam for a process pool; adding one before there is demand is speculative |
| **Persistent cross-run history / analytics** | Per-run JSONL is enough to debug a run. Aggregation across runs is a product question, not an architecture one |

---

## 8. Open Design Questions

Ordered by how much they could invalidate the design. The first three should be spikes *before* the TUI is written.

**1. Does attribute-call resolution have usable precision? (highest risk)**
The unique-method-name heuristic is the load-bearing assumption behind every `broken_caller` finding on method calls, and method calls are most of real Python. If `save`, `get`, `run`, and `handle` each have six definitions in a typical repo, the resolvable fraction collapses and swarm degrades into a function-only tool with a large unresolved bucket. **Spike:** run the extractor over 10 real open-source Python repos and measure (a) fraction of attribute call sites resolvable, (b) precision of the unique-name resolution against manual review of 100 samples. If resolvable is under ~50%, the honest options are to add a narrow local type inference (assignment-flow within a function body, which is cheap and would cover most cases) or to scope v1 to module-level functions.

**2. Is the turn boundary a good enough settle signal?**
The whole live-detection design rests on agents ending turns at coherent points. If agents habitually end a turn mid-refactor — caller written, callee not — `TENTATIVE` conflicts will be constant noise and `CONFIRMED` will fire on transient states. **Spike:** instrument 20 real agent runs and measure how often a turn boundary leaves the worktree in a state that would produce a self-inconsistent contract. If it is common, the fallback is a stricter gate: settle only when the agent's own tests pass, or only on `report_done` — which costs live-ness and is a real product regression.

**3. Is `pyright` on the merged tree a better product than all of this?**
Honest assessment: running a type-checker on `swarm/<run>/integration` would catch a strict superset of what shape analysis catches, for a fraction of the code. The genuine differentiator is not the AST work — it is **live, pre-merge, per-turn detection with automatic agent re-prompting**, which no type-checker can do because it needs the union of two worktrees before either finishes. That reframing should shape priorities: the extractor is a means, the injection loop is the product. Worth a spike comparing a `pyright`-on-merge baseline against the shape detector on the same conflict corpus, because if shape analysis adds little over the baseline, the right v1 is *live cross-pollination driven by a cheap detector* rather than a sophisticated one.

**4. How often do these conflicts actually occur?**
If 2–6 agents on reasonably-partitioned tasks conflict once per 50 runs, swarm is a vitamin. **Spike:** run 30 multi-agent tasks on 3 real repos with detection in report-only mode and count confirmed conflicts per run. This number should gate how much is invested in the TUI, and it is measurable in a day.

**5. Does automatic re-prompting help or thrash?**
Injecting a peer contract mid-run could resolve cleanly, or send the caller agent off refactoring things it should not touch. The prompt in §6.2 tries to fence this ("do not change services/auth.py"), but instruction adherence on a 20-turn agent under context pressure is an empirical question. Bounded rounds and a comparison against report-only mode are the way to answer it.

**6. Should `RENAMED_COMPATIBLE` be a conflict at all?**
A parameter rename that preserves position breaks only keyword callers. It could be reported as a distinct low-severity class, folded into `DIVERGED`, or suppressed unless a keyword call site actually breaks. The third is most principled and cheapest; I am not confident it is what users want to see, since a silent rename is also a code-review smell.

**7. Is severity-ordered sequential merge right?**
Merging cleanest-first minimizes textual conflict, but a different order might minimize *contract* conflict, and the two objectives can disagree. An optimal ordering is a small search problem. Unclear whether it matters in practice at n ≤ 6 — probably not, but it is unmeasured.

**8. Test-caller severity.**
§4.3 tags test call sites `MEDIUM` rather than excluding them. Reasonable people will disagree: tests are also where a broken contract shows up first, arguing for `HIGH`, and they are also full of intentional edge-case calls, arguing for exclusion. Needs a real corpus to settle.

**9. Where does the token budget live?**
`task_budget` is per-request; a 40-turn agent needs a cumulative ceiling, and a run needs a fleet-wide one. The orchestrator can enforce a run budget by refusing admission, but the *graceful degradation* story — agents wrapping up cleanly rather than being cut off — needs `task_budget` set per turn from remaining allowance. The interaction between per-turn budgets, `effort`, and thinking-on-by-default is not something I would commit to without measuring on a real run.

---

## Verification

No implementation this turn. When v1 is built, the end-to-end check is:

1. **Synthetic conflict fixture.** A small repo with `authenticate(user)` and two tasks engineered to collide. `swarm run` must report exactly one `broken_caller`, repair it, and merge clean. This is the product's canonical demo and its regression test.
2. **Unit tests on `accepts` and `relation`.** Property-based (`hypothesis`) over generated `Shape`/`CallSite` pairs, asserting the lattice laws: `relation(s, s) == IDENTICAL`; `WIDENED` implies every call `accepts`-ed by old is `accepts`-ed by new; `relation` is antisymmetric between `WIDENED` and `NARROWED`. These two functions are the whole algorithm — they get the most test weight.
3. **False-positive corpus.** Extract contracts from 10 real repos at two adjacent commits each and assert zero conflicts where the real commit was backward-compatible. Precision is the metric that decides adoption.
4. **Crash-recovery test.** `SIGKILL` the orchestrator mid-run, then `swarm doctor` must reconcile with zero orphaned worktrees and zero lost branches. Repeat with `ENOSPC` injected.
5. **Live run on a real repo.** 3 agents, any live provider key, real worktrees; verify the shared-prefix cache helps (provider cache-read tokens stay high from the second agent onward), no 429 storms, and a merge that produces an integration branch whose tests pass.
