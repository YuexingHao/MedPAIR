#!/usr/bin/env python3
"""Rewrite legacy 14B path strings in a notebook (optional maintenance).

Run from anywhere: python fix_notebook.py /path/to/notebook.ipynb
Requires Merge/paths.py on sys.path (run with cwd inside Merge or set PYTHONPATH).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

def patch_source(text: str) -> str:
    text = text.replace(
        'input_folder = "Merge_Attribution_Scores_14B_Analysis/Merge Attribution Scores 14B"',
        "input_folder = str(paths.CONTEXTCITE_14B_STAGING)",
    )
    text = text.replace(
        'input_folder = "Merge Attribution Scores 14B"',
        "input_folder = str(paths.CONTEXTCITE_14B_STAGING)",
    )
    text = text.replace(
        'folder_path = "Merge Attribution Scores 14B"',
        "folder_path = str(paths.CONTEXTCITE_14B_STAGING)",
    )
    text = text.replace(
        'output_dir = "Merge Attribution Scores 14B"',
        "output_dir = str(paths.CONTEXTCITE_14B_STAGING)",
    )
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("notebook", type=Path, help="Path to .ipynb")
    args = ap.parse_args()
    path = args.notebook.expanduser().resolve()
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, str):
            new = patch_source(src)
            if new != src:
                cell["source"] = new
                changed += 1
        else:
            full = "".join(src)
            new = patch_source(full)
            if new != full:
                cell["source"] = [new]
                changed += 1
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {path} ({changed} code cells touched).")


if __name__ == "__main__":
    main()
