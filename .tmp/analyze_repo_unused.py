import ast
import os
import sys

ROOT = "."
paths = []
for dirpath, dirnames, filenames in os.walk("src"):
    for fn in filenames:
        if fn.endswith(".py"):
            paths.append(os.path.join(dirpath, fn))

results = {}
for p in paths:
    try:
        with open(p, "r", encoding="utf-8") as f:
            src = f.read()
        mod = ast.parse(src, p)
    except Exception as e:
        results[p] = {"error": str(e)}
        continue
    unused_args = []
    unused_locals = []
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            if args and args[0] == "self":
                args = args[1:]
            used = set()
            assigned = set()
            for n in ast.walk(node):
                if isinstance(n, ast.Name):
                    used.add(n.id)
                if isinstance(n, ast.Assign):
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            assigned.add(t.id)
                        elif isinstance(t, ast.Tuple):
                            for e in t.elts:
                                if isinstance(e, ast.Name):
                                    assigned.add(e.id)
                if isinstance(n, ast.AnnAssign):
                    t = n.target
                    if isinstance(t, ast.Name):
                        assigned.add(t.id)
            unused = [a for a in args if a not in used]
            unused_local = sorted([a for a in assigned if a not in used and not a.startswith("_")])
            if unused:
                unused_args.append({"func": node.name, "lineno": node.lineno, "args": unused})
            if unused_local:
                unused_locals.append({"func": node.name, "lineno": node.lineno, "locals": unused_local})
    results[p] = {"unused_args": unused_args, "unused_locals": unused_locals}

# print concise summary
for p, r in results.items():
    if "error" in r:
        print(p, "ERROR", r["error"])
        continue
    if r["unused_args"] or r["unused_locals"]:
        print("\n", p)
        for ua in r["unused_args"]:
            print(f"  func {ua['func']}@{ua['lineno']}: unused args: {ua['args']}")
        for ul in r["unused_locals"]:
            print(f"  func {ul['func']}@{ul['lineno']}: unused locals assigned: {ul['locals']}")
