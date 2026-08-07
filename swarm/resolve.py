"""Resolution layer: assign attribute call sites to canonical SymbolKeys.

Implements §2.5's resolution rules over the whole repo:

* A repo-wide unique-method-name index resolves `obj.method()` calls
  (rule 5). If the method name has exactly one definition in the repo, it
  resolves heuristically; otherwise it stays unresolved.
* Re-export chains inside the repo are followed to the defining module
  (so `from .impl import f` in `__init__.py` resolves to `impl.f`).
* Name calls whose target module is not the repo's own top-level are
  dropped (out-of-repo == stdlib / site-packages == never a conflict),
  per rule 1 of §2.5.
"""

from __future__ import annotations

from pathlib import Path

from .extract import extract_source, module_name_for
from .models import CallSite, Conflict, FileContract, SymbolKey


class ResolutionError(Exception):
    pass


class Repo:
    """All contracts in a repo plus the tables the resolver needs."""

    def __init__(self, root: Path, top: str) -> None:
        self.root = root
        self.top = top
        self.contracts: list[FileContract] = []
        self.definitions: dict[SymbolKey, object] = {}
        self.method_index: dict[str, list[SymbolKey]] = {}
        self.reexports: dict[SymbolKey, SymbolKey] = {}
        # All call sites that could not be resolved, with the reason.
        self.unresolved: list[CallSite] = []
        self.call_sites: list[CallSite] = []
        # Set of dotted module names present in the repo. Used to decide
        # whether an imported name is repo-local or out-of-repo (see
        # _follow_local): the repo is the ground truth for what it owns.
        self.modules: set[str] = set()

    def add(self, contract: FileContract) -> None:
        self.contracts.append(contract)
        for key, shape in contract.definitions.items():
            self.definitions[key] = shape
            self.modules.add(key.module)
            if "." in key.qualname:
                self.method_index.setdefault(key.qualname.rsplit(".", 1)[1], []).append(key)
        for key, target in contract.reexports.items():
            self.reexports[key] = target
            self.modules.add(key.module)

    # -- attribute resolution (unique-method-name heuristic, §2.5.5) --------
    def resolve_attribute(self, attr: str) -> SymbolKey | None:
        keys = self.method_index.get(attr, [])
        if len(keys) == 1:
            return keys[0]
        return None

    def _in_repo(self, key: SymbolKey | None) -> bool:
        """Is the module of this key part of this repo's own tree?"""
        if key is None:
            return False
        return any(key.module == m or key.module.startswith(m + ".")
                   for m in self.modules)

    def _follow_reexport(self, key: SymbolKey) -> SymbolKey | None:
        """Follow re-export chains within the repo to the defining module."""
        seen: set[SymbolKey] = set()
        cur = key
        while cur not in seen:
            seen.add(cur)
            if cur in self.definitions:
                return cur
            nxt = self.reexports.get(cur)
            if nxt is None:
                break
            cur = nxt
        return None

    def _follow_local(self, key: SymbolKey | None) -> SymbolKey | None:
        """Resolve a local name to a definition, following re-exports."""
        if not self._in_repo(key):
            return None
        return self._follow_reexport(key)

    def resolve(self) -> None:
        """Run resolution across every contract's call sites."""
        for contract in self.contracts:
            resolved: list[CallSite] = []
            for call in contract.calls:
                if call.reason == "attribute-call":
                    key = self.resolve_attribute(call.attr or "")
                    if key is None:
                        self.unresolved.append(call)
                        continue
                    resolved.append(self._rewrite(call, key, "heuristic"))
                elif call.reason == "name-call":
                    if not self._in_repo(call.key):
                        # out-of-repo target (stdlib/site-packages): the call
                        # site is dropped entirely, never a conflict (§2.5.1)
                        continue
                    key = self._follow_reexport(call.key)
                    if key is None:
                        self.unresolved.append(call)
                        continue
                    resolved.append(self._rewrite(call, key, "exact"))
                else:
                    self.unresolved.append(call)
            contract.calls = resolved

    def _follow_local(self, key: SymbolKey | None) -> SymbolKey | None:
        """Resolve a local name like Key(module=a.b, qualname=g) to a
        definition, following re-export within the repo.

        A target is dropped (None) when its module is not part of this repo:
        that is how the stdlib/site-packages name-collision class is killed
        (rule 1 of §2.5) -- the set of modules we actually indexed *is* the
        repo, so only modules we scanned can ever be repo-local.
        """
        if key is None:
            return None
        if key.module in self.modules or key.module in {m.split(".", 1)[0] for m in self.modules}:
            return self._follow_reexport(key)
        return None

    def _rewrite(self, call: CallSite, key: SymbolKey | None, confidence: str) -> CallSite:
        return CallSite(
            key=key, n_positional=call.n_positional, keywords=call.keywords,
            has_star_args=call.has_star_args, has_star_kwargs=call.has_star_kwargs,
            file=call.file, line=call.line, in_test=call.in_test,
            confidence=confidence, reason=call.reason, attr=call.attr,
        )


def build_repo(root: Path, top: str | None = None) -> Repo:
    top = top or root.name.replace(".", "_")
    repo = Repo(root, top)
    for path in sorted(root.rglob("*.py")):
        if ".git" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        contract = extract_source(source, str(path), module_name_for(path, root))
        repo.add(contract)
    repo.resolve()
    return repo


def merged_contract(repo: Repo, only_files: set[str] | None = None) -> FileContract:
    """Roll a repo's per-file contracts into one agent/base contract.

    ``only_files`` is the set of absolute paths this agent actually touched
    (via ``GitExecutor.changed_vs_base``). When given, only definitions and
    call sites in those files count as *the agent's* -- everything else is
    inherited from the base commit and must not register the agent as a
    definer/caller. When omitted (a standalone repo), every file counts.

    Call sites have already been rewritten by ``Repo.resolve()``: name calls
    are canonical and attribute calls are either resolved (heuristic) or
    dropped into ``repo.unresolved``. Only resolved calls are included so
    out-of-repo (dropped) call sites never reach the detector.
    """
    contract = FileContract("<repo>")
    for fc in repo.contracts:
        if only_files is not None:
            if str(Path(fc.path)) not in only_files:
                continue
        contract.definitions.update(fc.definitions)
        contract.calls.extend(fc.calls)
    return contract


def base_index(repo: Repo) -> tuple[dict[SymbolKey, object], dict[SymbolKey, list]]:
    """Split a repo into (definitions, base_callers) for the detector."""
    definitions: dict[SymbolKey, object] = dict(repo.definitions)
    base_callers: dict[SymbolKey, list] = {}
    for fc in repo.contracts:
        for call in fc.calls:
            if call.key is not None:
                base_callers.setdefault(call.key, []).append(call)
    return definitions, base_callers