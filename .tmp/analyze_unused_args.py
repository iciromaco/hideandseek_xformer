import ast

p = "src/envs/hns_environment.py"
with open(p, "r", encoding="utf-8") as f:
    src = f.read()

mod = ast.parse(src, p)
unused_args = []
for node in ast.walk(mod):
    if isinstance(node, ast.FunctionDef):
        args = [a.arg for a in node.args.args]
        if args and args[0] == "self":
            args = args[1:]
        used = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                used.add(n.id)
        unused = [a for a in args if a not in used]
        if unused:
            unused_args.append((node.name, unused, node.lineno))

print("Detected unused function args:")
for name, unused, ln in unused_args:
    print(f"{name} @ {ln}: {unused}")

# --- detect assigned-but-unused local variables per function ---
print("\nDetecting assigned-but-unused local names per function:")
for node in ast.walk(mod):
    if isinstance(node, ast.FunctionDef):
        assigned = set()
        used = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        assigned.add(t.id)
                    elif isinstance(t, ast.Tuple):
                        for e in t.elts:
                            if isinstance(e, ast.Name):
                                assigned.add(e.id)
            elif isinstance(n, ast.AnnAssign):
                t = n.target
                if isinstance(t, ast.Name):
                    assigned.add(t.id)
            elif isinstance(n, ast.Name):
                used.add(n.id)
        # simple exclusions
        unused_local = sorted([a for a in assigned if a not in used and not a.startswith("_")])
        if unused_local:
            print(f"{node.name} @ {node.lineno}: {unused_local}")
