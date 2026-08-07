"""Command-line interface for swarm (subcommands from §6 of the architecture).

    run         decompose a goal and drive a multi-agent run
    status      project the event log to a run status
    conflicts   show confirmed conflicts (exit 1 if any HIGH)
    extract     dump the contract index for a path (dev/fixture utility)
    merge       build swarm/<run>/integration
    doctor      reconcile manifests against `git worktree list`
    abort       record an abort-intent for a live run
    log         replay the event log for a run
    resolve     apply a take-a/take-b/defer strategy on the integration branch

Most read-only commands accept ``--json`` for scripting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__ as VERSION
from .events import EventBus, fold_status

_RUN = ".swarm"


def _find_run(repo: Path, run_id: str) -> Path:
    d = repo / _RUN / run_id
    if not d.is_dir():
        print(f"no such run: {run_id} (looked in {d})", file=sys.stderr)
        sys.exit(2)
    return d


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _branches(run: Path) -> dict[str, str]:
    """agent-name -> branch, from per-worktree manifests."""
    out: dict[str, str] = {}
    wt = run / "worktrees"
    if not wt.is_dir():
        return out
    for mf in sorted(wt.glob("*.json")):
        branch = _read_json(mf).get("branch")
        if branch:
            out[branch.rsplit("/", 1)[-1]] = branch
    return out


# ---- run ---------------------------------------------------------------------
# Known OpenAI-compatible providers (base_url / key env var / default model).
PROVIDERS: dict[str, dict] = {
    "openai": {"base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY",
               "model": "gpt-4o-mini"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
             "model": "llama-3.3-70b-versatile"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY",
                   "model": "openai/gpt-4o-mini"},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None,
               "model": "qwen2.5-coder:latest"},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "key_env": None,
                 "model": "qwen2.5-coder"},
    "anthropic": {"base_url": None, "key_env": "ANTHROPIC_API_KEY",
                  "model": "claude-opus-5"},
}


def _live_transport(args) -> object:
    """Build the live transport for the requested provider; raises on a
    missing key or unknown provider. ``args.max_tokens`` (or the transport
    default) caps each turn's output tokens."""
    from .transport import AnthropicTransport, OpenAICompatTransport, TransportError

    provider = (args.provider or os.environ.get("SWARM_PROVIDER", "openai")).lower()
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise ValueError(
            f"unknown provider {provider!r}; choose from {', '.join(sorted(PROVIDERS))}")
    model = args.model or os.environ.get("SWARM_MODEL") or spec["model"]
    api_key = args.api_key or (os.environ.get(spec["key_env"])
                               if spec["key_env"] else None)
    max_tokens = args.max_tokens or int(os.environ.get("SWARM_MAX_TOKENS") or 0) or None
    if provider == "anthropic":
        if not api_key:
            raise ValueError(
                "anthropic needs ANTHROPIC_API_KEY (or --api-key)")
        return AnthropicTransport(model=model, api_key=api_key,
                                  **{"max_tokens": max_tokens} if max_tokens else {})
    base_url = args.base_url or os.environ.get("SWARM_BASE_URL") or spec["base_url"]
    if api_key is None and "localhost" not in base_url and "127.0.0.1" not in base_url:
        raise ValueError(
            f"{provider} needs {spec['key_env']} (or --api-key); "
            "keyless endpoints (ollama/lmstudio) run against localhost")
    return OpenAICompatTransport(model=model, base_url=base_url, api_key=api_key,
                                 **{"max_tokens": max_tokens} if max_tokens else {})


def cmd_run(args) -> int:
    from .models import Task
    from .orchestrator import Orchestrator
    from .transport import MockTransport
    from .transport import TransportError

    repo = Path(args.repo).resolve()
    if args.tasks:
        tasks = [Task(id=f"t{i}", title=t.strip(), instructions=t.strip())
                 for i, t in enumerate(args.tasks)]
    elif args.goal:
        words = args.goal.split()
        n = max(1, min(args.agents, len(words)))
        per = max(1, len(words) // n)
        tasks = [Task(id=f"t{i}", title=c, instructions=c)
                 for i, c in enumerate([" ".join(words[k:k+per]) for k in range(0, len(words), per)])]
    else:
        print("provide GOAL or --task", file=sys.stderr)
        return 2

    if args.mock:
        def factory(_name: str):
            return MockTransport([{"tool": "report_done",
                                   "inputs": {"summary": "no-op", "files_changed": []}}])
    else:
        try:
            transport = _live_transport(args)
        except (ValueError, TransportError) as exc:
            print(f"run failed: {exc}", file=sys.stderr)
            return 2

        def factory(_name: str):
            return transport

    orch = Orchestrator(repo, base=args.base or None, transport_factory=factory,
                        auto_repair_rounds=args.auto_repair, max_workers=args.agents,
                        system_prompt=args.system)
    try:
        out = orch.run(tasks)
    except Exception as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        return 1

    states = {k: v.state for k, v in out["results"].items()}
    conflicts = [{"kind": c.kind.value, "symbol": c.symbol, "definer": c.definer,
                  "caller": c.caller, "severity": c.severity} for c in out["conflicts"]]
    if args.json:
        print(json.dumps({"run_id": out["run_id"], "states": states, "conflicts": conflicts}, indent=2))
    else:
        print(f"swarm run {out['run_id']} complete")
        for name, r in out["results"].items():
            print(f"  {name:<12} {r.state:<12} turns {r.turns}  files {len(r.files)}")
        for c in conflicts:
            print(f"  [{c['severity']}] {c['kind']} {c['symbol']} by {c['definer']}->{c['caller']}")
    return 0


# ---- read / admin --------------------------------------------------------------
def cmd_status(args) -> int:
    from .tui import render
    run = _find_run(Path(args.repo).resolve(), args.run_id)
    status = fold_status(EventBus(run / "events.jsonl").read())
    print(json.dumps(status, indent=2) if args.json else render(status))
    return 0


def cmd_extract(args) -> int:
    from .resolve import build_repo
    root = Path(args.path)
    repo = build_repo(root, args.top or None)
    payload = {"modules": len(repo.contracts),
               "defs": sorted(str(k) for k in repo.definitions),
               "unresolved": [f"{c.file}:{c.line}" for c in repo.unresolved]}
    print(json.dumps(payload, indent=2) if args.json else
          "\n".join([f"modules: {payload['modules']}", *[f"  def {d}" for d in payload["defs"]]]))
    return 0


def cmd_conflicts(args) -> int:
    run = _find_run(Path(args.repo).resolve(), args.run_id)
    conflicts = _conflict_list(run)
    if args.json:
        print(json.dumps(conflicts, indent=2))
    else:
        for c in conflicts:
            print(f"[{c.get('severity', '?')}] {c.get('conflict_kind', 'kind')}"
                  f"  {c.get('symbol')}  by {c.get('definer', '-')} -> {c.get('caller', '-')}")
    return 1 if any(c.get("severity") == "high" for c in conflicts) else 0


def cmd_merge(args) -> int:
    from .gitops import GitExecutor
    from .merge import merge_into_integration
    repo = Path(args.repo).resolve()
    run = _find_run(repo, args.run_id)
    git = GitExecutor(repo)
    base = _read_json(run / "run.json").get("base") or git.head()
    rep = merge_into_integration(git, base, args.run_id, _branches(run), repo,
                                 conflicts=None, work_dir=run)
    if args.json:
        print(json.dumps({"merged": rep.merged, "errors": rep.errors, "residual": rep.residual}, indent=2))
    else:
        print("\n".join([f"  merged {a}" for a in rep.merged] +
                        [f"  error {e}" for e in rep.errors] +
                        [f"residual contract conflicts on integration: {len(rep.residual)}"]))
    return 1 if (rep.errors or rep.residual) else 0


def cmd_doctor(args) -> int:
    from .gitops import GitExecutor, doctor
    repo = Path(args.repo).resolve()
    # worktree manifests live per-run under .swarm/<run>/worktrees/*.json
    manifests = sorted(p for p in (repo / _RUN).rglob("worktrees/*.json")
                       if p.is_file())
    lines = doctor(GitExecutor(repo), manifests) or ["ok: (no manifest files)"]
    for line in lines:
        print(line)
    return 0 if all(l.startswith("ok:") for l in lines) else 1


def cmd_abort(args) -> int:
    repo = Path(args.repo).resolve()
    run = _find_run(repo, args.run_id)
    (run / "abort").write_text("abort requested\n", encoding="utf-8")
    print(json.dumps({"aborted": args.run_id}) if args.json else f"abort requested for {args.run_id}")
    return 0


def cmd_log(args) -> int:
    run = _find_run(Path(args.repo).resolve(), args.run_id)
    for ev in EventBus(run / "events.jsonl").read():
        print(ev)
    return 0


def cmd_resolve(args) -> int:
    from .gitops import GitExecutor
    from .merge import apply_strategy
    repo = Path(args.repo).resolve()
    run = _find_run(repo, args.run_id)
    if args.strategy not in {"take-a", "take-b", "defer"}:
        print("strategy must be one of take-a, take-b, defer", file=sys.stderr)
        return 2
    applied = apply_strategy(GitExecutor(repo), args.integration or f"swarm/{args.run_id}/integration",
                             _branches(run), args.strategy, args.agent, args.paths)
    for p in applied:
        print(f"applied {args.strategy} on {p}")
    return 0


def _conflict_list(run: Path) -> list:
    return [e for e in EventBus(run / "events.jsonl").read() if e.get("kind") == "conflict"]


# ---- arg wiring ---------------------------------------------------------------
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", "--root", default=".", help="path to the git repo")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="swarm",
                                     description="Contract-aware parallel Claude coding agents.")
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run"); _add_common(p)
    p.add_argument("goal", nargs="?"); p.add_argument("--agents", type=int, default=4)
    p.add_argument("--task", action="append", dest="tasks"); p.add_argument("--base", default=None)
    p.add_argument("--auto-repair", type=int, default=2); p.add_argument("--system", default="")
    p.add_argument("--provider", default=None,
                   help="openai|groq|openrouter|ollama|lmstudio|anthropic "
                        "(default: env SWARM_PROVIDER or openai)")
    p.add_argument("--model", default=None, help="model name (default: per provider)")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible API root URL")
    p.add_argument("--api-key", default=None, help="provider API key (or the provider's env var)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="max output tokens per turn (default: 16384 / provider-safe)")
    p.add_argument("--mock", action="store_true"); p.add_argument("--json", action="store_true")

    for name in ("status", "conflicts", "merge", "abort", "log"):
        q = sub.add_parser(name); _add_common(q)
        q.add_argument("run_id"); q.add_argument("--json", action="store_true")

    q = sub.add_parser("resolve"); _add_common(q)
    q.add_argument("run_id"); q.add_argument("--strategy", required=True)
    q.add_argument("--agent", default=None); q.add_argument("--paths", nargs="*")
    q.add_argument("--integration", default=None)

    q = sub.add_parser("extract"); _add_common(q)
    q.add_argument("path"); q.add_argument("--top", default=None); q.add_argument("--json", action="store_true")

    (sub.add_parser("doctor")).add_argument("--repo", "--root", default=".")

    args = parser.parse_args(argv)
    handlers = {"run": cmd_run, "status": cmd_status, "conflicts": cmd_conflicts,
                "extract": cmd_extract, "merge": cmd_merge, "doctor": cmd_doctor,
                "abort": cmd_abort, "log": cmd_log, "resolve": cmd_resolve}
    return handlers[args.command](args)