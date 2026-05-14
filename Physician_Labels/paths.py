"""Stable paths under ``Physician_Labels/`` (import from repo code or notebooks).

``Mar2_2026_Data/`` is unchanged and remains the canonical location for the 933 majority-vote
CSV used across ``After_PT_Removal`` scripts.
"""
from pathlib import Path

PHYSICIAN_LABELS_ROOT = Path(__file__).resolve().parent
MAR2_DATA = PHYSICIAN_LABELS_ROOT / "Mar2_2026_Data"
RESULTS = PHYSICIAN_LABELS_ROOT / "results"
REFERENCE = PHYSICIAN_LABELS_ROOT / "reference"
NOTEBOOKS = PHYSICIAN_LABELS_ROOT / "notebooks"

MAJORITY_VOTE_933 = MAR2_DATA / "933_Clinician_Student_Majority_Vote.csv"
