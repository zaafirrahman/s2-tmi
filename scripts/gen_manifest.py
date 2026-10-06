"""Scan repo -> manifest.json (dipanggil GitHub Action tiap push)."""
import json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SKIP_DIRS = {".git", ".github", "scripts", "node_modules", "__pycache__", ".ipynb_checkpoints"}
SKIP_FILES = {"index.html", "manifest.json", "README.md", ".gitignore"}

files = []
for d, dirs, names in os.walk(ROOT):
    dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))
    for n in sorted(names):
        full = os.path.join(d, n)
        rel = os.path.relpath(full, ROOT).replace(os.sep, "/")
        if rel in SKIP_FILES or n.startswith("."):
            continue
        files.append({"p": rel, "s": os.path.getsize(full)})

with open(os.path.join(ROOT, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump({"files": files}, f, ensure_ascii=False)
print(f"{len(files)} files")
