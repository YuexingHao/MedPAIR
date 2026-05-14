#!/usr/bin/env python3
"""Remove embedded OpenAI ``sk-proj-...`` secrets from ``.py`` and ``.ipynb`` (including outputs).

Excludes: ``Unused_Files/myenv``, ``.git``, ``__pycache__``.

Run::

  python After_PT_Removal/scripts/_strip_embedded_openai_keys.py

Then use ``export OPENAI_API_KEY='sk-...'`` before notebooks/scripts; ``OpenAI()`` reads it automatically.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_PARTS = ("Unused_Files/myenv", ".git", "__pycache__")
KEY_TOKEN = r"sk-proj-[A-Za-z0-9_-]+"
_KEY_RE = re.compile(KEY_TOKEN)


def scrub_text(text: str) -> str:
    text = re.sub(
        rf"OpenAI\(\s*api_key\s*=\s*[\"']{KEY_TOKEN}[\"']\s*\)",
        "OpenAI()",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        rf"openai\.OpenAI\(\s*api_key\s*=\s*[\"']{KEY_TOKEN}[\"']\s*\)",
        "openai.OpenAI()",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(rf",\s*api_key\s*=\s*[\"']{KEY_TOKEN}[\"']", "", text)
    text = re.sub(rf"api_key\s*=\s*[\"']{KEY_TOKEN}[\"']\s*,?", "", text)
    text = re.sub(
        rf"openai\.api_key\s*=\s*[\"']{KEY_TOKEN}[\"'][^\n]*",
        "# Set OPENAI_API_KEY in your environment before running.",
        text,
    )
    text = re.sub(rf"(?m)^\s*[\"']?{KEY_TOKEN}[\"']?\s*\n", "", text)
    if _KEY_RE.search(text):
        text = _KEY_RE.sub("<REMOVED_USE_OPENAI_API_KEY_ENV>", text)
    return text


def scrub_json_obj(o: object) -> object:
    if isinstance(o, dict):
        return {k: scrub_json_obj(v) for k, v in o.items()}
    if isinstance(o, list):
        return [scrub_json_obj(v) for v in o]
    if isinstance(o, str):
        t = scrub_text(o)
        if _KEY_RE.search(t):
            t = _KEY_RE.sub("<REMOVED_USE_OPENAI_API_KEY_ENV>", t)
        return t
    return o


def scrub_notebook_file(nb_path: Path) -> bool:
    raw = nb_path.read_text(encoding="utf-8")
    if "sk-proj-" not in raw:
        return False
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return False
    old = json.dumps(nb, sort_keys=True)
    nb = scrub_json_obj(nb)
    new = json.dumps(nb, sort_keys=True)
    if old == new:
        return False
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main() -> None:
    if not ROOT.is_dir():
        print("Bad ROOT", ROOT, file=sys.stderr)
        sys.exit(1)

    py_changed = ipy_changed = 0
    for path in ROOT.rglob("*.py"):
        if any(sp in path.parts for sp in SKIP_PARTS):
            continue
        if path.name == "_strip_embedded_openai_keys.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if not _KEY_RE.search(text):
            continue
        new_text = scrub_text(text)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            py_changed += 1
            print("updated", path.relative_to(ROOT))

    for path in ROOT.rglob("*.ipynb"):
        if any(sp in path.parts for sp in SKIP_PARTS):
            continue
        try:
            if scrub_notebook_file(path):
                ipy_changed += 1
                print("updated", path.relative_to(ROOT))
        except OSError:
            pass

    print(f"Done. Python files: {py_changed}, notebooks: {ipy_changed}")


if __name__ == "__main__":
    main()
