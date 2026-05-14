"""MedGemma-27b-text-it project layout (import from notebooks after sys.path bootstrap).

- DATA: input / reference CSVs (data/raw/)
- PREDICTIONS: model outputs and progress checkpoints (results/predictions/)
- TABLES: analysis exports (results/tables/)
"""
from pathlib import Path

MEDGEMMA27_ROOT = Path(__file__).resolve().parent
DATA = MEDGEMMA27_ROOT / "data" / "raw"
RESULTS = MEDGEMMA27_ROOT / "results"
PREDICTIONS = RESULTS / "predictions"
TABLES = RESULTS / "tables"
