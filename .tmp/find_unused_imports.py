#!/usr/bin/env python3
"""
Scan Python files under src/ for unused imports and duplicate imports per-file.
Outputs a human-readable report to stdout and saves JSON to .tmp/imports_report.json
"""

import ast
import json
import os
from pathlib import Path

ROOT = Path("src")
PYFILES = list(ROOT.rglob("*.py"))
report = {"files": {}}

for p in PYFILES:
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        continue
    try:
        tree = ast.parse(text)
    except Exception:
        continue
    imports = []
    # import a, import b as c, from x import y, from x import y as z
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(
                    {
                        "type": "import",
                        "module": None,
                        "name": n.name,
                        "asname": n.asname,
                        "lineno": node.lineno,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module
            for n in node.names:
                imports.append(
                    {
                        "type": "from",
                        "module": mod,
                        "name": n.name,
                        "asname": n.asname,
                        "lineno": node.lineno,
                    }
                )
    # collect all Name usages excluding those in import nodes
    used_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used_names.add(node.id)
    file_report = {"imports": imports, "unused": [], "duplicates": []}
    # detect unused: for each import, determine exposed name(s)
    seen_modules = {}
    for imp in imports:
        if imp["type"] == "import":
            # import a.b as c -> exposed name is asname or top-level name before dot
            exposed = imp["asname"] if imp["asname"] else imp["name"].split(".")[0]
            if exposed not in used_names:
                file_report["unused"].append({**imp, "exposed": exposed})
            # duplicates by name
            key = ("import", imp["name"])
            seen_modules.setdefault(key, []).append(imp["lineno"])
        else:
            # from module import name
            exposed = imp["asname"] if imp["asname"] else imp["name"]
            if exposed not in used_names:
                file_report["unused"].append({**imp, "exposed": exposed})
            key = ("from", imp["module"] or "", imp["name"])
            seen_modules.setdefault(key, []).append(imp["lineno"])
    for k, lines in seen_modules.items():
        if len(lines) > 1:
            file_report["duplicates"].append({"key": k, "lines": lines})
    if file_report["unused"] or file_report["duplicates"]:
        report["files"][str(p)] = file_report

# summary
summary = {
    "file_count": len(report["files"]),
}
print("Unused import / duplicate import scan summary:")
print(json.dumps(summary, indent=2))
for fp, fr in report["files"].items():
    print("\nFile:", fp)
    if fr["unused"]:
        print("  Unused imports:")
        for u in fr["unused"]:
            print(f"    - line {u['lineno']}: {u['type']} {u['module'] or ''} {u['name']} as {u['asname']} exposed={u['exposed']}")
    if fr["duplicates"]:
        print("  Duplicate imports:")
        for d in fr["duplicates"]:
            print("    -", d)

# save json
outp = Path(".tmp") / "imports_report.json"
try:
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSaved JSON report to", str(outp))
except Exception as e:
    print("Failed to save JSON report:", e)
