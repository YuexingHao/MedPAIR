"""Shared After_PT_Removal assets (import from notebooks after sys.path bootstrap).

- DATA: Centaur / cross-model CSVs used by multiple notebooks (data/)
- NOTEBOOKS: shared notebooks (notebooks/)
- SCRIPTS: shared scripts (scripts/)
- GPT4O_PREDICTIONS: sibling ``GPT4o/results/predictions`` (GPT-4o prediction runs)
"""
from pathlib import Path

SHARED_ROOT = Path(__file__).resolve().parent
DATA = SHARED_ROOT / "data"
NOTEBOOKS = SHARED_ROOT / "notebooks"
SCRIPTS = SHARED_ROOT / "scripts"
GPT4O_ROOT = SHARED_ROOT.parent / "GPT4o"
GPT4O_PREDICTIONS = GPT4O_ROOT / "results" / "predictions"
