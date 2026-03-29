import ast
import os

keys_in_info = {}
for dirpath, dirnames, filenames in os.walk("src"):
    for fn in filenames:
        if not fn.endswith(".py"):
            continue
        p = os.path.join(dirpath, fn)
        try:
            src = open(p, "r", encoding="utf-8").read()
            mod = ast.parse(src, p)
        except Exception:
            continue
        # find dict literals assigned to name 'info'
        for node in ast.walk(mod):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "info" and isinstance(node.value, ast.Dict):
                        for k in node.value.keys:
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                keys_in_info.setdefault(p, set()).add(k.value)
            # info[...] usages
            if isinstance(node, ast.Subscript):
                try:
                    if isinstance(node.value, ast.Name) and node.value.id == "info":
                        key = None
                        idx = node.slice
                        if isinstance(idx, ast.Constant) and isinstance(idx.value, str):
                            key = idx.value
                        elif hasattr(ast, "Index") and isinstance(idx, ast.Index) and isinstance(idx.value, ast.Constant) and isinstance(idx.value.value, str):
                            key = idx.value.value
                        if key:
                            keys_in_info.setdefault(p, set()).add(key)
                except Exception:
                    pass
# print summary
all_keys = {}
for p, ks in keys_in_info.items():
    for k in ks:
        all_keys.setdefault(k, set()).add(p)
print("Unique info keys and where used:")
for k, ps in sorted(all_keys.items(), key=lambda x: (-len(x[1]), x[0])):
    print(f"{k}: {len(ps)} files")
    for fp in list(ps)[:5]:
        print("  -", fp)
