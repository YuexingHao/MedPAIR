"""Llama-70B project layout (import from notebooks after sys.path bootstrap).

- DATA: input / reference CSVs (data/raw/)
- PREDICTIONS: model outputs and progress checkpoints (results/predictions/)
- TABLES: analysis exports (results/tables/)
"""
from pathlib import Path

LLAMA70_ROOT = Path(__file__).resolve().parent
DATA = LLAMA70_ROOT / "data" / "raw"
RESULTS = LLAMA70_ROOT / "results"
PREDICTIONS = RESULTS / "predictions"
TABLES = RESULTS / "tables"
