"""
Canonical filesystem paths for the replication repository.

Every script in `code/` imports the constants below so the project is fully
relocatable: clone the repo anywhere, and `python code/04_event_study.py` will
read and write into the sibling `data/` and `report/` directories without any
configuration.

The paths are anchored to this file (which lives at the repo root) using
`Path(__file__).resolve().parent`. Do not move `paths.py`.
"""

from pathlib import Path

# ── Repo root ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent

# ── Code ─────────────────────────────────────────────────────────────────────
CODE_DIR = REPO_ROOT / "code"

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_DIR = REPO_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
RESULTS_DIR = DATA_DIR / "results"

# raw inputs
TRUTH_ARCHIVE_JSON = RAW_DIR / "truth_archive.json"
TRUTH_ARCHIVE_CSV = RAW_DIR / "truth_archive.csv"
POSTS_PARQUET = RAW_DIR / "posts_60d.parquet"           # window + topic-tagged subset
MINUTE_BARS_5M = RAW_DIR / "minute_bars_5m"             # one parquet per ticker

# interim derived
SIGNALS_5M = INTERIM_DIR / "signals_5m"                 # one parquet per ticker

# results (JSON / CSV outputs of analysis stages)
EVENT_STUDY_5M_JSON = RESULTS_DIR / "orderflow_event_study_5m.json"
PLACEBO_5M_JSON = RESULTS_DIR / "orderflow_placebo_5m.json"
SENSITIVITY_5M_JSON = RESULTS_DIR / "orderflow_sensitivity_5m.json"
FDR_5M_JSON = RESULTS_DIR / "orderflow_fdr_5m.json"
FDR_5M_CSV = RESULTS_DIR / "orderflow_fdr_5m_trim.csv"
FINDING4_OIL_JSON = RESULTS_DIR / "finding4_oil_complex.json"
DOLLAR_BOUND_JSON = RESULTS_DIR / "dollar_upper_bound_strategies.json"
TIMESHIFT_JSON = RESULTS_DIR / "phase2b_timeshift.json"
LOO_JSON = RESULTS_DIR / "phase2c_loo.json"

# ── Report ───────────────────────────────────────────────────────────────────
REPORT_DIR = REPO_ROOT / "report"
REPORT_TEX = REPORT_DIR / "report.tex"
REPORT_PDF = REPORT_DIR / "report.pdf"
FIGURES_DIR = REPORT_DIR / "figures"


def ensure_dirs():
    """Create any missing project directories. Called by stage-1 scripts."""
    for d in (RAW_DIR, INTERIM_DIR, RESULTS_DIR, MINUTE_BARS_5M,
              SIGNALS_5M, REPORT_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"REPO_ROOT     = {REPO_ROOT}")
    print(f"DATA_DIR      = {DATA_DIR}")
    print(f"RESULTS_DIR   = {RESULTS_DIR}")
    print(f"REPORT_DIR    = {REPORT_DIR}")
    print(f"FIGURES_DIR   = {FIGURES_DIR}")
