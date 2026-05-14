"""GPT4o After_PT_Removal layout (import from notebooks after sys.path bootstrap).

- RAW: merge / round annotation CSVs (data/raw/)
- SR_CONCORDANCE_DATA: inputs for SR concordance (data/sr_concordance/)
- REMOVE_LOW_IRR_DATA: remove-low/irrelevant experiment inputs (results/predictions/remove_low_irr/)
- PREDICTIONS: GPT-4o model outputs (results/predictions/); GPT-5 outputs live under sibling ``GPT5/results/predictions`` (see ``GPT5_PREDICTIONS``)
- TABLES: summary tables (results/tables/)
- FIGURES: figures (results/figures/)
"""
from pathlib import Path

GPT4O_ROOT = Path(__file__).resolve().parent
DATA = GPT4O_ROOT / "data"
RAW = DATA / "raw"
SR_CONCORDANCE_DATA = DATA / "sr_concordance"
RESULTS = GPT4O_ROOT / "results"
PREDICTIONS = RESULTS / "predictions"
REMOVE_LOW_IRR_DATA = PREDICTIONS / "remove_low_irr"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
NOTEBOOKS = GPT4O_ROOT / "notebooks"
SCRIPTS = GPT4O_ROOT / "scripts"
GPT5_ROOT = GPT4O_ROOT.parent / "GPT5"
GPT5_PREDICTIONS = GPT5_ROOT / "results" / "predictions"
