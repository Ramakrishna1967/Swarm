# swarm

Dependency-free Python package (stdlib only) that orchestrates parallel coding
agents on isolated git worktrees and detects **interface contract conflicts**
between their in-flight changes before merge.

The failure mode it targets: agent A changes `authenticate(user)` to
`authenticate(user, scope)`; agent B, isolated, writes a call to the old
one-arg form. Both worktrees are internally consistent, git sees no textual
conflict, the merge succeeds, and the bug stays silent until runtime. swarm
differs function *shape* across concurrently-modified branches to catch that
gap. Full design: `SWARM_ARCHITECTURE.md`.

swarm never edits code itself and never touches the user's branch.

## Install / run

No third-party dependencies. Python 3.10+.

```powershell
python -m pytest tests -q          # run the test suite
python -m swarm --version
```

## CLI

```
python -m swarm run "goal" --agents 4 --system "custom prompt"
python -m swarm run --task auth="..." --task ui="..." --mock
python -m swarm status <run_id>
python -m swarm conflicts <run_id>        # exit 1 if any HIGH conflict
python -m swarm merge <run_id>            # build swarm/<run_id>/integration
python -m swarm resolve <run_id> --strategy take-a --agent A --paths services/auth.py
python -m swarm extract path/to/dir       # dump the contract index
python -m swarm doctor                    # reconcile manifests vs `git worktree list`
python -m swarm abort <run_id>
python -m swarm log <run_id>
```

Read-only commands accept `--json` for scripting. `run --mock` drives agents
with a no-op MockTransport instead of the real Anthropic tool loop.

## Library surface

- `models.py` — symbol keys, shapes, file contracts, conflict kinds, task/run records
- `shapes.py` — structural `accepts()`, `relation()`, `summarizes()`
- `extract.py` — AST contract extractor (imports, decorators, calls, `__init__` folding)
- `resolve.py` — `build_repo`: import tables, re-export chains, per-symbol definitions
- `conflicts.py` — three-way (base vs A vs B) conflict detector
- `store.py` — `ContractStore`: debounced extraction, syntax-error fallback
- `events.py` — thread-safe JSONL `EventBus` + `fold_status`
- `admission.py` — semaphore + token-bucket admission control
- `transport.py` — `Transport`, `AnthropicTransport`, `MockTransport`
- `gitops.py` — `GitExecutor`, per-worktree manifests, `WorktreeManager`, `doctor()`
- `runner.py` — agent tool loop, sandbox, settle signal, cancellation
- `orchestrator.py` — worktrees + branches, concurrent runs, checkpointing,
  base-contract index, settle→detect→repair injection, cancel
- `merge.py` — severity-ordered integration (own worktree), residual detection,
  take-a / take-b / defer strategies
- `tui.py` — pure `render()` (ASCII fallback) + log watch

## Status

- Contract extraction, resolution, conflict detection, orchestrator, merge, and
  CLI are implemented and covered by unit + end-to-end tests (real temp git
  repo with mock agents).
- `AnthropicTransport` is written but only verified against a live key; run with
  `--mock` to exercise the orchestration without one.
