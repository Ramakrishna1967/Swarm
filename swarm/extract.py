"""Contract extraction and resolution.

Implements §2.5. The extractor is a lightweight but correctness-first AST
walk. A separate resolution layer decides whether a finding is trustworthy,
and its default is silence: anything it cannot reason about lands in the
"unresolved" bucket rather than producing a false conflict.

The Name resolution resolves imports and re-export chains to a canonical
SymbolKey; attribute calls resolve only when the method name has exactly
one repo-wide definition (the unique-method-name heuristic).
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from .models import CallSite, FileContract, Shape, SymbolKey
from .shapes import ARITY_TRANSPARENT


def _decorator_name(node: ast.expr) -> str:
    return ast.unparse(node)


def _shape(node: ast.FunctionDef | ast.AsyncFunctionDef, qualname: str) -> Shape:
    args = node.args
    positional_nodes = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional_nodes) - len(args.defaults)) + list(args.defaults)
    positional = tuple(a.arg for a in positional_nodes)
    required = sum(d is None for d in defaults)
    kw_required = frozenset(a.arg for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is None)
    kw_optional = frozenset(a.arg for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None)
    is_method = "." in qualname
    first = positional[0] if positional else None
    decorators = tuple(_decorator_name(d) for d in node.decorator_list)
    static = any(d and d.split(".")[-1] == "staticmethod" for d in decorators)
    opaque = any(d and not d.split(".")[-1] in {t.split(".")[-1] for t in ARITY_TRANSPARENT}
                 for d in decorators)
    implicit_self = is_method and first in {"self", "cls"} and not static
    shape: Shape = Shape(
        positional, len(args.posonlyargs), required,
        kw_required, kw_optional, args.vararg is not None, args.kwarg is not None,
        is_method, implicit_self, decorators, opaque,
    )
    return shape


def module_name_for(path: Path, root: Path | None = None) -> str:
    p = path
    if root is not None:
        p = path.relative_to(root)
    parts = list(p.with_suffix("").parts)
    # an __init__.py *is* its package: `from pack import f` names module "pack"
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class _Visitor(ast.NodeVisitor):
    def __init__(self, module: str, path: str, file_contract: FileContract) -> None:
        self.module = module
        self.path = path
        self.contract = file_contract
        self.scope: list[str] = []
        self.imports: dict[str, SymbolKey] = {}
        self._where: list[str] = []

    # --- imports -------------------------------------------------------------
    def _record_reexports(self) -> None:
        # `from x import y` in this module makes `y` a resolvable name here;
        # when another module does `from <this-module> import y` it must be
        # followed to the original definition (rule: re-export chains within
        # the repo). Keyed as (this module, local name) -> target.
        for local, target in self.imports.items():
            if target.qualname:
                self.contract.reexports[SymbolKey(self.module, local)] = target

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            local = item.asname or item.name.split(".")[0]
            self.imports[local] = SymbolKey(item.name, "")
        self._record_reexports()

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            self.generic_visit(node)
            return
        base = node.module
        if node.level:
            # Relative import: qualify against the containing package so
            # `from .impl import f` inside pack/__init__.py names
            # pack.impl:f, not impl:f. `node.level` counts the leading dots.
            base = self._relative_module(node.level, node.module)
        for item in node.names:
            if item.name == "*":
                continue
            self.imports[item.asname or item.name] = SymbolKey(base, item.name)
        self._record_reexports()

    def _relative_module(self, level: int, module: str | None) -> str:
        """Resolve a relative import to its absolute dotted module name.

        level 1 = the current package, level 2 = its parent, and so on,
        matching Python's ``from .x import`` / ``from ..x import`` semantics.
        The visitor knows its own module (``self.module``), so a relative
        name is resolved against it. ``module`` holds everything after the
        dots (may be empty for ``from . import y``).
        """
        pkg_parts = self.module.split(".")[:-1]  # parent package of this file
        if not pkg_parts:
            # __init__.py of a top-level package: relative level 1 is the
            # package itself.
            if level <= 1:
                pkg_parts = self.module.split(".")
            else:
                return module or self.module
        else:
            # go up (level - 1) more packages from the file's parent package
            pkg_parts = pkg_parts[:-(level - 1)] if level > 1 else pkg_parts
            if not pkg_parts and level > 1:
                return module or self.module
        suffix = module.lstrip(".") if module else ""
        return ".".join([*pkg_parts, suffix]) if suffix else ".".join(pkg_parts)

    # --- classes --------------------------------------------------------------
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = [i.id for i in node.bases if isinstance(i, ast.Name)]
        self.scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = ".".join([*self.scope, node.name])
        key = SymbolKey(self.module, qualname)
        self.contract.definitions[key] = _shape(node, qualname)
        self.scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self.scope.pop()

    # --- call sites ------------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        common = dict(
            n_positional=len(node.args),
            keywords=frozenset(k.arg for k in node.keywords if k.arg),
            has_star_args=any(isinstance(a, ast.Starred) for a in node.args),
            has_star_kwargs=any(k.arg is None for k in node.keywords),
            file=self.path,
            line=node.lineno,
            in_test=_in_test_path(self.path),
        )
        func = node.func
        if isinstance(func, ast.Name):
            target = self.imports.get(func.id, SymbolKey(self.module, func.id))
            self.contract.calls.append(CallSite(
                key=target, confidence="exact", reason="name-call", **common,
            ))
        elif isinstance(func, ast.Attribute):
            # Candidate resolution is owned by the resolver: it needs the
            # full repo's definition index, so we leave it unresolved here.
            self.contract.calls.append(CallSite(
                key=None, confidence="unresolved",
                reason="attribute-call", attr=func.attr, **common,
            ))
        else:
            self.contract.calls.append(CallSite(
                key=None, confidence="unresolved", reason="unknown-call", **common,
            ))
        self.generic_visit(node)


def _in_test_path(path: str) -> bool:
    p = Path(path)
    return p.name.startswith("test_") or p.name.endswith("_test") or "tests" in p.parts


def extract_source(source: str, path: str = "<memory>", module: str | None = None) -> FileContract:
    module = module or module_name_for(Path(path))
    contract = FileContract(path)
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        contract.syntax_error = f"{exc.msg} at line {exc.lineno}"
        contract.dirty_unparseable = True
        return contract
    visitor = _Visitor(module, path, contract)
    visitor.visit(tree)
    return contract


def file_digest(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime_ns}:{st.st_size}:{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"