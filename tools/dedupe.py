"""Replace locally re-declared classes with imports of the canonical definition.

usage: dedupe.py FILE NAME:from_module [NAME:from_module ...]   (NAME:- just deletes)
"""
import ast, sys

path, specs = sys.argv[1], sys.argv[2:]
src = open(path).read()
lines = src.splitlines(keepends=True)
tree = ast.parse(src)
targets = dict(s.split(":", 1) for s in specs)
spans, imports, first = [], {}, None
for n in tree.body:
    if isinstance(n, ast.ClassDef) and n.name in targets:
        start = min([n.lineno] + [d.lineno for d in n.decorator_list])
        # swallow preceding comment lines
        while start > 1 and lines[start - 2].lstrip().startswith("#"):
            start -= 1
        spans.append((start, n.end_lineno))
        first = start if first is None else min(first, start)
        if targets[n.name] != "-":
            imports.setdefault(targets[n.name], []).append(n.name)
missing = set(targets) - {lines and n.name for n in tree.body if isinstance(n, ast.ClassDef)}
if missing: sys.exit(f"{path}: not found {missing}")
for s, e in sorted(spans, reverse=True):
    del lines[s - 1:e]
# insert imports after the last top-level import
last_imp = max((n.end_lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) and n.end_lineno < first), default=0)
ins = "".join(f"from {m} import {', '.join(ns)}\n" for m, ns in imports.items())
lines.insert(last_imp, ins)
out = "".join(lines)
while "\n\n\n\n" in out:
    out = out.replace("\n\n\n\n", "\n\n\n")
open(path, "w").write(out)
ast.parse(out)
print("ok", path)
