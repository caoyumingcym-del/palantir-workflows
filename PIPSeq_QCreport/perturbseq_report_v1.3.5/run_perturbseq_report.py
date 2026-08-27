#!/usr/bin/env python3
"""
Launcher — run the pipeline without installing anything.

    python run_perturbseq_report.py --manifest /path/to/sample_manifest.csv

Keep this file next to the ``perturbseq_report/`` package directory. If you do
install the package (``pip install -e .``), the ``perturbseq-report`` command
does the same thing and this file is unnecessary.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perturbseq_report.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
