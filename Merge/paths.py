"""Merge project layout (import from notebooks after sys.path bootstrap).

- DATA: merged CSVs and analysis tables (data/csv/)
- FIGURES: PDF figures (figures/)
- NOTEBOOKS: notebooks (notebooks/) — Jupyter cwd is often notebooks/ or repo root
- SCRIPTS: Python utilities (scripts/)
- ATTRIBUTION: ContextCite / physician match bundles (attribution/)
"""
from pathlib import Path

MERGE_ROOT = Path(__file__).resolve().parent
DATA = MERGE_ROOT / "data" / "csv"
FIGURES = MERGE_ROOT / "figures"
NOTEBOOKS = MERGE_ROOT / "notebooks"
SCRIPTS = MERGE_ROOT / "scripts"
ATTRIBUTION = MERGE_ROOT / "attribution"
ATTRIBUTION_QWEN72B = ATTRIBUTION / "qwen72b_contextcite"
# Long-form 14B attribution (one row per sentence); canonical replacement for per-question Merge_Q*.csv
MERGED_ATTRIBUTION_SCORES_14B = DATA / "merged_attribution_scores_14B.csv"
# If you re-run ContextCite notebooks that emit one CSV per question, point output here (gitignored).
CONTEXTCITE_14B_STAGING = DATA / "contextcite_14b_staging"
