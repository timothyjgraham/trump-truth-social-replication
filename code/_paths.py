"""Tiny shim: make the repo-root paths.py importable from any script in code/.

Each pipeline script does:
    from _paths import (...)

This avoids needing to install the repo as a package.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# re-export everything from the canonical paths module
from paths import *  # noqa: F401,F403
from paths import (  # noqa: F401  (explicit re-export for IDEs)
    REPO_ROOT, CODE_DIR, DATA_DIR,
    RAW_DIR, INTERIM_DIR, RESULTS_DIR,
    TRUTH_ARCHIVE_JSON, TRUTH_ARCHIVE_CSV, POSTS_PARQUET,
    MINUTE_BARS_5M, SIGNALS_5M,
    EVENT_STUDY_5M_JSON, PLACEBO_5M_JSON, SENSITIVITY_5M_JSON,
    FDR_5M_JSON, FDR_5M_CSV, FINDING4_OIL_JSON,
    DOLLAR_BOUND_JSON, TIMESHIFT_JSON, LOO_JSON,
    REPORT_DIR, REPORT_TEX, REPORT_PDF, FIGURES_DIR,
    ensure_dirs,
)
