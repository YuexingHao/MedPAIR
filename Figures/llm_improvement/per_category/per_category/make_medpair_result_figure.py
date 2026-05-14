#!/usr/bin/env python3
"""
Thin wrapper: the implementation is ``Figures/llm_improvement/per_category/make_medpair_result_figure.py``.

Run from repo root::

  python Figures/llm_improvement/per_category/make_medpair_result_figure.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_IMPL = Path(__file__).resolve().parents[1] / "make_medpair_result_figure.py"
if not _IMPL.is_file():
    print(f"Missing implementation script: {_IMPL}", file=sys.stderr)
    sys.exit(1)

_spec = importlib.util.spec_from_file_location("_medpair_per_category_impl", _IMPL)
if _spec is None or _spec.loader is None:
    print(f"Could not load {_IMPL}", file=sys.stderr)
    sys.exit(1)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    _mod.main()
