"""One-off repair: collapse try/except ImportError chains and fix module paths."""
import ast, os, sys, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "rl")

def norm(s): return s.replace("_", "").replace("-", "").lower()

# module dotted name -> file
modules = {}
for d, _, fs in os.walk(PKG):
    for f in fs:
        if f.endswith(".py"):
            rel = os.path.relpath(os.path.join(d, f), ROOT)[:-3].replace(os.sep, ".")
            if rel.endswith(".__init__"): rel = rel[:-9]
            modules[rel] = os.path.join(d, f)
by_norm_tail = {}
for m in modules:
    by_norm_tail.setdefault(norm(m.split(".")[-1]), []).append(m)

_defs = {}
def defined(mod):
    if mod not in _defs:
        names = set()
        tree = ast.parse(open(modules[mod]).read())
        for n in tree.body:
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)): names.add(n.name)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name): names.add(t.id)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name): names.add(n.target.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names: names.add(a.asname or a.name.split(".")[0])
            elif isinstance(n, ast.Try):
                for sub in ast.walk(n):
                    if isinstance(sub, (ast.ClassDef,)): names.add(sub.name)
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        for a in sub.names: names.add(a.asname or a.name.split(".")[0])
        _defs[mod] = names
    return _defs[mod]

def resolve(cur_mod, node):
    """Return absolute rl module for an ImportFrom, or None."""
    pkg = cur_mod.split(".")[:-1] if not modules[cur_mod].endswith("__init__.py") else cur_mod.split(".")
    if node.level:
        base = pkg[: len(pkg) - (node.level - 1)]
        parts = base + (node.module.split(".") if node.module else [])
    else:
        if node.module is None: return None
        parts = node.module.split(".")
        if parts[0] not in ("rl",):
            if parts[0] in ("typing", "abc", "__future__", "math", "random", "struct", "sys", "collections",
                            "copy", "uuid", "enum", "functools", "itertools", "dataclasses", "heapq", "bisect", "os", "threading", "operator", "numbers"):
                return "STDLIB"
            parts = ["rl"] + parts
    cand = ".".join(parts)
    if cand in modules: return cand
    # fuzzy: match by normalized tail, prefer same package path
    tail = norm(parts[-1])
    opts = by_norm_tail.get(tail, [])
    if len(opts) == 1: return opts[0]
    for o in opts:
        if norm(o) == norm(cand): return o
    return None

def render(cur_mod, target, names):
    cur_pkg = cur_mod.split(".")[:-1]
    tgt = target.split(".")
    common = 0
    while common < min(len(cur_pkg), len(tgt)) and cur_pkg[common] == tgt[common]: common += 1
    up = len(cur_pkg) - common
    rel = "." * (up + 1) + ".".join(tgt[common:])
    items = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in names)
    return f"from {rel} import {items}"

def try_is_import_fallback(t):
    if not t.handlers: return False
    for h in t.handlers:
        if h.type is None: continue
        nm = ast.unparse(h.type)
        if not any(x in nm for x in ("ImportError", "ModuleNotFoundError")): return False
    return all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in t.body)

report = []
for mod, path in sorted(modules.items()):
    src = open(path).read()
    if not src.strip(): continue
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    edits = []  # (start, end, text)  1-based inclusive lines
    def visit_block(body, indent):
        for n in body:
            if isinstance(n, ast.Try) and try_is_import_fallback(n):
                choice = None
                body_imports = [b for b in n.body if isinstance(b, ast.ImportFrom)]
                rs = [resolve(mod, b) for b in body_imports]
                if body_imports and len(body_imports) == len(n.body) and all(r and r != "STDLIB" for r in rs):
                    choice = "".join(" " * indent + render(mod, r, b.names) + "\n" for r, b in zip(rs, body_imports)).lstrip(" ")
                    missing = [(a.name, r) for r, b in zip(rs, body_imports) for a in b.names if a.name not in defined(r)]
                    if missing: report.append(f"MISSING NAMES {missing} (from {mod}:{n.lineno})")
                elif any(r == "STDLIB" for r in rs) or not body_imports:
                    continue  # genuine optional dependency
                if choice is None:
                    report.append(f"UNRESOLVED try at {mod}:{n.lineno}"); continue
                edits.append((n.lineno, n.end_lineno, " " * indent + choice))
            elif isinstance(n, ast.ImportFrom):
                r = resolve(mod, n)
                if r == "STDLIB": continue
                if r is None:
                    report.append(f"UNRESOLVED import at {mod}:{n.lineno}: {ast.unparse(n)}"); continue
                missing = [a.name for a in n.names if a.name != "*" and a.name not in defined(r)]
                if missing: report.append(f"MISSING NAMES {missing} in {r} (from {mod}:{n.lineno})")
                new = " " * indent + render(mod, r, n.names) + "\n"
                if "".join(lines[n.lineno-1:n.end_lineno]).strip() != new.strip():
                    edits.append((n.lineno, n.end_lineno, new))
            elif isinstance(n, ast.If):
                visit_block(n.body, n.body[0].col_offset); visit_block(n.orelse, n.orelse[0].col_offset if n.orelse else 0)
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                visit_block(n.body, n.body[0].col_offset)
    visit_block(tree.body, 0)
    if edits:
        for s, e, t in sorted(edits, reverse=True):
            lines[s-1:e] = [t]
        if "--write" in sys.argv: open(path, "w").write("".join(lines))
        report.append(f"edited {mod}: {len(edits)}")
print("\n".join(report))
