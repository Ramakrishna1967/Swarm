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

## Providers

swarm drives agents through whatever LLM you point it at. Two real backends
ship, both built so no third-party package is required:

| Provider | Key env var | Notes |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | default; `gpt-4o-mini` by default |
| Groq | `GROQ_API_KEY` | fast llama models |
| OpenRouter | `OPENROUTER_API_KEY` | one key, many models |
| Ollama | — (localhost) | keyless, runs locally |
| LM Studio | — (localhost) | keyless, runs locally |
| Anthropic | `ANTHROPIC_API_KEY` | Claude tool-use |

Any other OpenAI-compatible endpoint (`--base-url`) works the same way.
Select with `--provider`, override the model with `--model`, override the
endpoint with `--base-url`, cap each turn's output with `--max-tokens`
(default 16384, provider-safe for gpt-4o-mini/Groq), or pass a key
explicitly with `--api-key` (reads the provider's env var by default).
`SWARM_PROVIDER`, `SWARM_MODEL`, `SWARM_BASE_URL`, `SWARM_MAX_TOKENS`
set the same options via env for library users of `Orchestrator`. `--mock`
drives agents with a no-op `MockTransport` instead of calling any API.

## CLI

```
python -m swarm run "goal" --agents 4 --system "custom prompt"
python -m swarm run --task auth="..." --task ui="..." --mock
python -m swarm run "goal" --provider groq --model llama-3.3-70b-versatile
python -m swarm status <run_id>
python -m swarm conflicts <run_id>        # exit 1 if any HIGH conflict
python -m swarm merge <run_id>            # build swarm/<run_id>/integration
python -m swarm resolve <run_id> --strategy take-a --agent A --paths services/auth.py
python -m swarm extract path/to/dir       # dump the contract index
python -m swarm doctor                    # reconcile manifests vs `git worktree list`
python -m swarm abort <run_id>
python -m swarm log <run_id>
```

Read-only commands accept `--json` for scripting.

## Library surface

- `models.py` — symbol keys, shapes, file contracts, conflict kinds, task/run records
- `shapes.py` — structural `accepts()`, `relation()`, `summarizes()`
- `extract.py` — AST contract extractor (imports, decorators, calls, `__init__` folding)
- `resolve.py` — `build_repo`: import tables, re-export chains, per-symbol definitions
- `conflicts.py` — three-way (base vs A vs B) conflict detector
- `store.py` — `ContractStore`: debounced extraction, syntax-error fallback
- `events.py` — thread-safe JSONL `EventBus` + `fold_status`
- `admission.py` — semaphore + token-bucket admission control
- `transport.py` — `Transport`, `OpenAICompatTransport` (OpenAI/Groq/OpenRouter/Ollama/vLLM via stdlib HTTP), `AnthropicTransport`, `MockTransport`
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
- The live backends (`OpenAICompatTransport`, `AnthropicTransport`) are written
  but only verified against a live key. Use `--mock` to exercise the
  orchestration without one.
