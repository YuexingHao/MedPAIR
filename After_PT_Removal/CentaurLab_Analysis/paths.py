"""CentaurLab_Analysis layout — use from project root (see notebook bootstrap cell)."""
from pathlib import Path

CENTAUR_ROOT = Path(__file__).resolve().parent
DATA = CENTAUR_ROOT / "data" / "raw"
RESULTS = CENTAUR_ROOT / "results"
TABLES = RESULTS / "tables"
IMAGE_NECESSARY = RESULTS / "image_necessary"
FIGURES = RESULTS / "figures"
