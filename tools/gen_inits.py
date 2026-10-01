"""Generate package __init__.py files re-exporting each module's public classes/functions."""
import ast, os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl")
SKIP_PKGS = {"db"}
PRIMITIVES = {"FLOAT", "UINT", "INT", "BYTE", "FEATURE_VECTOR"}  # db/__init__.py is hand-written

def public_names(path, modname):
    tree = ast.parse(open(path).read())
    allv = None
    names = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets):
            allv = [e.value for e in n.value.elts]
        elif isinstance(n, ast.ClassDef) and not n.name.startswith("_"):
            names.append(n.name)
        elif isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
            names.append(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                # aliases like spPolicy = Policy / DEFAULT_TILE_CONT
                if isinstance(t, ast.Name) and (t.id.startswith("sp") or t.id.isupper()) and not t.id.startswith("_"):
                    if isinstance(n.value, (ast.Name, ast.Attribute, ast.Subscript)) or t.id.startswith("DEFAULT"):
                        names.append(t.id)
    imported = set()
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            imported.update(a.asname or a.name for a in n.names)
    if allv is not None:
        names = allv
    # Only export what the module itself defines; primitives come from rl.declares.
    return [x for x in names if x not in imported and x not in PRIMITIVES]

for d, subdirs, files in os.walk(ROOT):
    subdirs[:] = sorted(x for x in subdirs if x != "__pycache__")
    rel = os.path.relpath(d, ROOT)
    if rel.split(os.sep)[0] in SKIP_PKGS: continue
    mods = sorted(f[:-3] for f in files if f.endswith(".py") and f != "__init__.py")
    subs = sorted(s for s in subdirs if os.path.exists(os.path.join(d, s, "__init__.py")) and s != "__pycache__")
    if rel == ".":
        mods = []  # rl.declares is imported explicitly below
    lines = ['"""Auto-generated re-exports. See tools/gen_inits.py."""\n']
    exported = {}
    for s in subs:
        if rel == ".": continue
        lines.append(f"from .{s} import *  # noqa: F401,F403")
        lines.append(f"from .{s} import __all__ as _{s}_all")
    for m in mods:
        names = [n for n in public_names(os.path.join(d, m + ".py"), m) if n not in exported]
        # Class named after module goes first; skip helper TypeVar-ish single letters
        names = [n for n in names if len(n) > 1]
        if not names: continue
        for n in names: exported[n] = m
        lines.append(f"from .{m} import " + ", ".join(names))
    if rel == ".":
        lines.append("from .declares import *  # noqa: F401,F403")
        lines.append("from . import agent, algorithm, coding, db, hash, policy, utility  # noqa: F401")
    elif exported or subs:
        lines.append("")
        lines.append("__all__ = [" + ", ".join(repr(n) for n in exported) + "]"
                     + "".join(f" + _{s}_all" for s in subs))
    open(os.path.join(d, "__init__.py"), "w").write("\n".join(lines) + "\n")
    print(rel, len(exported))
