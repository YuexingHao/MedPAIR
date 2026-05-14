"""GPT5 project layout (import from notebooks after sys.path bootstrap).

- DATA: input CSVs (data/raw/)
- PREDICTIONS / RELEVANCY / TABLES: under results/
- GPT4O_PREDICTIONS: sibling ``After_PT_Removal/GPT4o/results/predictions`` (GPT-4o runs, including
  ``gpt4o_predictions_on_14b_removed.csv`` / ``gpt4o_predictions_on_gpt5_removed.csv``)
- ARCHIVE: old or exploratory outputs (results/archive/)
- FIGURES: Sankey HTML/PNG/PDF exports (workspace ``Figures/MedPAIR_Sankey/figures/``)
"""
from pathlib import Path

GPT5_ROOT = Path(__file__).resolve().parent
_WORKSPACE_ROOT = GPT5_ROOT.parent.parent
DATA = GPT5_ROOT / "data" / "raw"
RESULTS = GPT5_ROOT / "results"
# GPT-4o prediction runs that live under After_PT_Removal/GPT4o (not GPT5/results/predictions)
GPT4O_ROOT = GPT5_ROOT.parent / "GPT4o"
GPT4O_PREDICTIONS = GPT4O_ROOT / "results" / "predictions"

# Result subfolders (keep RESULTS as the parent for figures layout)
PREDICTIONS = RESULTS / "predictions"
RELEVANCY = RESULTS / "relevancy"
TABLES = RESULTS / "tables"
ARCHIVE = RESULTS / "archive"
FIGURES = _WORKSPACE_ROOT / "Figures" / "MedPAIR_Sankey" / "figures"
