"""Qwen2.5-14B-Instruct project layout (import from notebooks after sys.path bootstrap).

- DATA: input / reference CSVs (data/raw/)
- PREDICTIONS: model outputs and progress checkpoints (results/predictions/)
- TABLES: analysis exports (results/tables/)
"""
from pathlib import Path

QWEN14_ROOT = Path(__file__).resolve().parent
DATA = QWEN14_ROOT / "data" / "raw"
RESULTS = QWEN14_ROOT / "results"
PREDICTIONS = RESULTS / "predictions"
TABLES = RESULTS / "tables"
