"""
Regression tests for DRAGEN metrics-file discovery (seqmetrics.py).

DRAGEN sometimes names the per-library metrics file after a longer string
than the manifest's own `prefix` column -- e.g. a hashing/feature-library
tag appended ("HR20260218_1_hashingABCD.scRNA_metrics.csv" for manifest
prefix "HR20260218_1"). This surfaced as "Sequencing QC -- not produced"
even with a correctly-resolved dragen_root, because _find_metric_file only
tried an exact `{prefix}.scRNA_metrics.csv` match.

Run with:  python -m pytest tests/test_seqmetrics.py -v
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from perturbseq_report.seqmetrics import (          # noqa: E402
    _find_metric_file, load_sequencing_metrics,
)

DRAGEN_CSV = (
    "GEX,{prefix},Total input reads,{total}\n"
    "GEX,{prefix},Number of cells,100\n"
)


def test_exact_prefix_match_still_wins() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "HR1.scRNA_metrics.csv").write_text(
            DRAGEN_CSV.format(prefix="HR1", total=500)
        )
        (d / "HR1_hashingABCD.scRNA_metrics.csv").write_text(
            DRAGEN_CSV.format(prefix="HR1_hashingABCD", total=999)
        )
        found = _find_metric_file(d, "HR1")
        assert found is not None and found.name == "HR1.scRNA_metrics.csv"


def test_suffixed_filename_is_found() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "HR20260218_1_hashingABCD.scRNA_metrics.csv").write_text(
            DRAGEN_CSV.format(prefix="HR20260218_1_hashingABCD", total=1234)
        )
        found = _find_metric_file(d, "HR20260218_1")
        assert found is not None
        assert found.name == "HR20260218_1_hashingABCD.scRNA_metrics.csv"


def test_numeric_prefix_collision_is_not_matched() -> None:
    """"HR20260218_1" must resolve to its own file, not "..._10"'s, when
    both suffixed files sit in the same (shared DRAGEN run) directory --
    a bare "{prefix}*" would let "_1" match the "_10" file too."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "HR20260218_1_hashingABCD.scRNA_metrics.csv").write_text(
            DRAGEN_CSV.format(prefix="HR20260218_1_hashingABCD", total=1000)
        )
        (d / "HR20260218_10_hashingXYZ.scRNA_metrics.csv").write_text(
            DRAGEN_CSV.format(prefix="HR20260218_10_hashingXYZ", total=2000)
        )
        found_1 = _find_metric_file(d, "HR20260218_1")
        found_10 = _find_metric_file(d, "HR20260218_10")
        assert found_1 is not None and found_1.name == "HR20260218_1_hashingABCD.scRNA_metrics.csv"
        assert found_10 is not None and found_10.name == "HR20260218_10_hashingXYZ.scRNA_metrics.csv"


def test_load_sequencing_metrics_resolves_suffixed_files_for_both_prefixes() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "HR20260218_1_hashingABCD.scRNA_metrics.csv").write_text(
            DRAGEN_CSV.format(prefix="HR20260218_1_hashingABCD", total=1000)
        )
        (d / "HR20260218_2_hashingABCD.scRNA_metrics.csv").write_text(
            DRAGEN_CSV.format(prefix="HR20260218_2_hashingABCD", total=2000)
        )
        runs = [
            {"sample": "s1", "prefix": "HR20260218_1", "dragen_path": d},
            {"sample": "s1", "prefix": "HR20260218_2", "dragen_path": d},
        ]
        sm = load_sequencing_metrics(runs)
        assert not sm.empty
        assert sm.missing == []
        assert set(sm.files) == {"HR20260218_1", "HR20260218_2"}


if __name__ == "__main__":
    test_exact_prefix_match_still_wins()
    test_suffixed_filename_is_found()
    test_numeric_prefix_collision_is_not_matched()
    test_load_sequencing_metrics_resolves_suffixed_files_for_both_prefixes()
    print("all checks passed")
