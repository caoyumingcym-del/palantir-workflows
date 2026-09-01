"""
Regression tests for the DRAGEN cellhashing.tsv fallback (hto_dragen.py).

    python tests/test_hto_dragen.py

Locks in the behaviour established against MDL-1856 via
``diagnose_cellhashing_tsv.py``: cellhashing.tsv is barcodes-as-rows with one
column per hashtag, barcodes can be far longer than a 10x 16bp barcode (28bp
combinatorial barcodes here, with no "-1" suffix), and -- the one that
actually changes the numbers -- multiple manifest rows for one sample are
multiple library preparations of the SAME cells, so a barcode recovered from
more than one run's file must be SUMMED across runs, not deduplicated by
keeping the first run seen.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturbseq_report.config import PipelineConfig            # noqa: E402
from perturbseq_report.hto_dragen import find_cellhashing_file  # noqa: E402


def test_finds_demux_tsv_with_suffixed_dragen_filename(tmp_path) -> None:
    """DRAGEN's actual per-cell hashtag file can be named "*.scRNA.demux.tsv"
    (not "*cellhashing.tsv"), and its own name can carry a longer tag past
    the manifest's prefix (e.g. "HR20260218_1_hashingABCD...tsv" for
    manifest prefix "HR20260218_1") -- see the WALKUP-19889 report."""
    p = tmp_path / "HR20260218_1_hashingABCD.scRNA.demux.tsv"
    p.write_text("Barcode\tprot:hash.A\tprot:hash.B\n"
                 "AAACAAACAAACACAAACACAACAAATG\t0\t3\n")
    cfg = PipelineConfig()
    found = find_cellhashing_file(tmp_path, "HR20260218_1", cfg.modality.hto_dragen_file_patterns)
    assert found == p

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"  {detail}"))
    if not cond:
        FAILURES.append(name)


def _write_cellhashing_tsv(
    path: Path, barcodes: list[str], hashtag_counts: dict[str, list[int]],
    extra_columns: dict[str, list] | None = None,
) -> None:
    df = pd.DataFrame({"Barcode": barcodes, **hashtag_counts, **(extra_columns or {})})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


# ===========================================================================
def test_barcodes_longer_than_10x_are_recognised() -> None:
    """28bp combinatorial barcodes must not be mistaken for 'not a barcode'.

    The original BARCODE_RE only allowed 6-20bp (standard 10x), which made
    orientation detection fail ("neither axis looks like barcodes") on
    MDL-1856's 28bp combinatorial barcodes even though the file was
    unambiguously barcodes-as-rows.
    """
    print("\n[28bp barcodes recognised, orientation resolved]")
    from perturbseq_report.hto_dragen import load_cellhashing_file

    rng = np.random.default_rng(0)
    bcs = ["".join(rng.choice(list("ACGT"), 28)) for _ in range(20)]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "run.scRNA.cellhashing.tsv"
        _write_cellhashing_tsv(
            p, bcs, {"prot:hash.A": rng.poisson(5, 20).tolist(),
                     "prot:hash.B": rng.poisson(50, 20).tolist()}
        )
        parsed = load_cellhashing_file(p)
        check("orientation resolved to barcodes_as_rows",
              parsed.orientation == "barcodes_as_rows", parsed.orientation)
        check("all 20 barcodes present", len(parsed.counts) == 20)
        check("both hashtag columns kept",
              list(parsed.counts.columns) == ["prot:hash.A", "prot:hash.B"])


def test_non_numeric_column_is_dropped_not_coerced() -> None:
    print("\n[non-numeric column dropped, not silently zeroed]")
    from perturbseq_report.hto_dragen import load_cellhashing_file

    rng = np.random.default_rng(1)
    bcs = ["".join(rng.choice(list("ACGT"), 16)) for _ in range(10)]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "run.scRNA.cellhashing.tsv"
        _write_cellhashing_tsv(
            p, bcs, {"prot:hash.A": rng.poisson(5, 10).tolist()},
            extra_columns={"notes": ["ok"] * 10},
        )
        parsed = load_cellhashing_file(p)
        check("'notes' column dropped", "notes" not in parsed.counts.columns)
        check("'notes' column reported in dropped_columns",
              "notes" in parsed.dropped_columns)
        check("hashtag column survives", "prot:hash.A" in parsed.counts.columns)


def test_orientation_raises_when_neither_axis_is_barcodes() -> None:
    print("\n[garbage file raises rather than guessing]")
    from perturbseq_report.hto_dragen import CellHashingLoadError, load_cellhashing_file

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "run.scRNA.cellhashing.tsv"
        pd.DataFrame({"col1": ["x", "y"], "col2": ["a", "b"]}).to_csv(
            p, sep="\t", index=False
        )
        try:
            load_cellhashing_file(p)
            check("raises CellHashingLoadError on unrecognisable layout", False)
        except CellHashingLoadError:
            check("raises CellHashingLoadError on unrecognisable layout", True)


def test_barcode_transform_is_chosen_by_best_overlap() -> None:
    print("\n[barcode transform: best overlap wins, not the first guess]")
    from perturbseq_report.hto_dragen import best_barcode_transform

    rng = np.random.default_rng(2)
    bcs = ["".join(rng.choice(list("ACGT"), 16)) for _ in range(30)]
    # Target has the "-1" suffix the file lacks; also include unrelated
    # barcodes so a lower-overlap transform can't win by accident.
    target = pd.Index([b + "-1" for b in bcs[:-5]] + ["ZZZZZZZZZZZZZZZZ-1"] * 3)
    match, transformed = best_barcode_transform(pd.Index(bcs), target)
    check("chosen transform is 'add -1 suffix'", match.transform == "add -1 suffix",
          match.transform)
    check("recovers all but the 5 held-out barcodes",
          match.n_matched == len(bcs) - 5, match.n_matched)


def test_same_barcode_summed_across_runs_for_one_sample() -> None:
    """The behaviour this module exists for: 4 runs, same cells, SUM not first.

    Confirmed against MDL-1856 via diagnose_cellhashing_tsv.py section G: a
    barcode recovered from every run showed comparable non-trivial counts in
    every run (same cells resequenced), not signal in one run and noise in
    the rest (which would instead argue for keeping only one run's counts).
    """
    print("\n[hashtag counts summed across multiple runs of the same sample]")
    from perturbseq_report.config import ModalityConfig
    from perturbseq_report.hto_dragen import build_hto_modality_from_dragen

    rng = np.random.default_rng(3)
    n_real = 15
    real_bcs = ["".join(rng.choice(list("ACGT"), 28)) for _ in range(n_real)]
    target = list(real_bcs)

    tmp = Path(tempfile.mkdtemp())
    try:
        expected_totals = {bc: 0 for bc in real_bcs}
        runs = []
        for i in range(1, 5):
            d = tmp / f"PHrun_{i}" / "dragen_output"
            counts_a = rng.poisson(20, n_real)
            counts_b = rng.poisson(20, n_real)
            for bc, a, b in zip(real_bcs, counts_a, counts_b):
                expected_totals[bc] += int(a) + int(b)
            # background noise barcodes, never in target
            noise_bcs = ["".join(rng.choice(list("ACGT"), 28)) for _ in range(200)]
            all_bcs = real_bcs + noise_bcs
            all_a = list(counts_a) + rng.poisson(1, 200).tolist()
            all_b = list(counts_b) + rng.poisson(1, 200).tolist()
            _write_cellhashing_tsv(
                d / f"PHrun_{i}.scRNA.cellhashing.tsv", all_bcs,
                {"prot:hash.A": all_a, "prot:hash.B": all_b},
            )
            runs.append({"sample": "s1", "prefix": f"PHrun_{i}", "dragen_path": d})

        cfg = ModalityConfig()
        mod, notes = build_hto_modality_from_dragen(runs, target, cfg)
        check("modality recovered", mod is not None and mod.present)
        frame = mod.to_frame()
        observed = frame.sum(axis=1)
        expected = pd.Series(expected_totals)
        max_diff = float((observed.reindex(expected.index) - expected).abs().max())
        check("summed totals match exactly across all 4 runs", max_diff == 0.0,
              f"max abs diff = {max_diff}")
        check("notes mention summing across runs",
              any("SUMMED" in n for n in notes), notes)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_single_run_still_works_unchanged() -> None:
    """A single-run recovery (no cross-run duplicates) is untouched by the
    summing change -- summing a barcode found in exactly one run is a no-op."""
    print("\n[single-run recovery still works]")
    from perturbseq_report.config import ModalityConfig
    from perturbseq_report.hto_dragen import build_hto_modality_from_dragen

    rng = np.random.default_rng(4)
    n = 25
    bcs = ["".join(rng.choice(list("ACGT"), 16)) for _ in range(n)]
    tmp = Path(tempfile.mkdtemp())
    try:
        d = tmp / "PH1_1" / "dragen_output"
        _write_cellhashing_tsv(
            d / "PH1_1.scRNA.cellhashing.tsv", bcs,
            {"prot:hash.A": rng.poisson(6, n).tolist(),
             "prot:hash.B": rng.poisson(60, n).tolist()},
        )
        target = [b + "-1" for b in bcs[:-3]] + [f"EXTRA{i}-1" for i in range(5)]
        cfg = ModalityConfig()
        runs = [{"sample": "s1", "prefix": "PH1_1", "dragen_path": d}]
        mod, notes = build_hto_modality_from_dragen(runs, target, cfg)
        check("modality recovered", mod is not None and mod.present)
        check("covers exactly the cells present in the file",
              int(mod.to_frame().sum(axis=1).gt(0).sum()) == n - 3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_max_mode_keeps_strongest_run_not_sum() -> None:
    """dragen_runs_share_cells=no -> keep the strongest run, discard the rest.

    Mirror of the "sum" test but for genuinely independent runs/pools: a
    shared barcode should end up with exactly ONE run's counts (whichever had
    the larger total), never the sum of both.
    """
    print("\n[combine_mode='max': strongest run kept, not summed]")
    from perturbseq_report.config import ModalityConfig
    from perturbseq_report.hto_dragen import build_hto_modality_from_dragen

    rng = np.random.default_rng(5)
    bc_strong_in_1 = "".join(rng.choice(list("ACGT"), 28))
    bc_strong_in_2 = "".join(rng.choice(list("ACGT"), 28))
    target = [bc_strong_in_1, bc_strong_in_2]

    tmp = Path(tempfile.mkdtemp())
    try:
        d1 = tmp / "R1" / "dragen_output"
        d2 = tmp / "R2" / "dragen_output"
        # R1: real signal for bc_strong_in_1, near-noise for the other.
        _write_cellhashing_tsv(
            d1 / "R1.scRNA.cellhashing.tsv",
            [bc_strong_in_1, bc_strong_in_2],
            {"prot:hash.A": [50, 1], "prot:hash.B": [40, 0]},
        )
        # R2: real signal for bc_strong_in_2, near-noise for the other.
        _write_cellhashing_tsv(
            d2 / "R2.scRNA.cellhashing.tsv",
            [bc_strong_in_1, bc_strong_in_2],
            {"prot:hash.A": [2, 45], "prot:hash.B": [1, 38]},
        )
        runs = [
            {"sample": "s1", "prefix": "R1", "dragen_path": d1},
            {"sample": "s1", "prefix": "R2", "dragen_path": d2},
        ]
        cfg = ModalityConfig()
        mod, notes = build_hto_modality_from_dragen(
            runs, target, cfg, combine_mode="max"
        )
        frame = mod.to_frame()
        check("bc_strong_in_1 keeps R1's counts, not R1+R2",
              int(frame.loc[bc_strong_in_1].sum()) == 90,
              int(frame.loc[bc_strong_in_1].sum()))
        check("bc_strong_in_2 keeps R2's counts, not R1+R2",
              int(frame.loc[bc_strong_in_2].sum()) == 83,
              int(frame.loc[bc_strong_in_2].sum()))
        check("notes mention independence, not summing",
              any("independent" in n for n in notes), notes)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_multi_sample_refuses_without_declaration() -> None:
    """A sample with >1 run and no dragen_runs_share_cells is skipped, not guessed."""
    print("\n[multi-sample: undeclared multi-run sample is skipped]")
    from perturbseq_report.config import ModalityConfig
    from perturbseq_report.hto_dragen import build_hto_modality_multi_sample

    rng = np.random.default_rng(6)
    bcs = ["".join(rng.choice(list("ACGT"), 16)) for _ in range(5)]
    tmp = Path(tempfile.mkdtemp())
    try:
        for run in ("R1", "R2"):
            d = tmp / run / "dragen_output"
            _write_cellhashing_tsv(
                d / f"{run}.scRNA.cellhashing.tsv", bcs,
                {"prot:hash.A": rng.poisson(10, 5).tolist()},
            )
        runs = [
            {"sample": "sampleA", "prefix": "R1", "dragen_path": tmp / "R1" / "dragen_output"},
            {"sample": "sampleA", "prefix": "R2", "dragen_path": tmp / "R2" / "dragen_output"},
        ]
        cfg = ModalityConfig()
        mod, notes = build_hto_modality_multi_sample(
            runs, bcs, sample_of_cell=["sampleA"] * len(bcs),
            share_cells_by_sample={"sampleA": None}, cfg=cfg,
        )
        check("nothing recovered when undeclared", mod is None)
        check("note names the missing manifest column",
              any("dragen_runs_share_cells" in n for n in notes), notes)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_multi_sample_combines_two_samples_with_different_modes() -> None:
    """sampleA (share_cells=yes, 2 runs) and sampleB (1 run) recovered together."""
    print("\n[multi-sample: two samples with different run structures]")
    from perturbseq_report.config import ModalityConfig
    from perturbseq_report.hto_dragen import build_hto_modality_multi_sample

    rng = np.random.default_rng(8)
    a_bcs = ["".join(rng.choice(list("ACGT"), 20)) for _ in range(6)]
    b_bcs = ["".join(rng.choice(list("ACGT"), 20)) for _ in range(4)]
    tmp = Path(tempfile.mkdtemp())
    try:
        expected_a = {bc: 0 for bc in a_bcs}
        for run in ("A1", "A2"):
            d = tmp / run / "dragen_output"
            counts = rng.poisson(15, len(a_bcs))
            for bc, c in zip(a_bcs, counts):
                expected_a[bc] += int(c)
            _write_cellhashing_tsv(
                d / f"{run}.scRNA.cellhashing.tsv", a_bcs,
                {"prot:hash.A": counts.tolist()},
            )
        d_b = tmp / "B1" / "dragen_output"
        b_counts = rng.poisson(15, len(b_bcs))
        _write_cellhashing_tsv(
            d_b / "B1.scRNA.cellhashing.tsv", b_bcs,
            {"prot:hash.A": b_counts.tolist()},
        )

        runs = [
            {"sample": "sampleA", "prefix": "A1", "dragen_path": tmp / "A1" / "dragen_output"},
            {"sample": "sampleA", "prefix": "A2", "dragen_path": tmp / "A2" / "dragen_output"},
            {"sample": "sampleB", "prefix": "B1", "dragen_path": d_b},
        ]
        target = a_bcs + b_bcs
        sample_of_cell = ["sampleA"] * len(a_bcs) + ["sampleB"] * len(b_bcs)
        cfg = ModalityConfig()
        mod, notes = build_hto_modality_multi_sample(
            runs, target, sample_of_cell=sample_of_cell,
            share_cells_by_sample={"sampleA": True, "sampleB": None}, cfg=cfg,
        )
        check("modality recovered", mod is not None and mod.present)
        frame = mod.to_frame()
        max_diff_a = max(
            abs(int(frame.loc[bc].sum()) - expected_a[bc]) for bc in a_bcs
        )
        check("sampleA counts summed across its 2 runs", max_diff_a == 0, max_diff_a)
        max_diff_b = max(
            abs(int(frame.loc[bc].sum()) - int(c))
            for bc, c in zip(b_bcs, b_counts)
        )
        check("sampleB (single run, no declaration needed) recovered correctly",
              max_diff_b == 0, max_diff_b)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
def main() -> int:
    for fn in (
        test_barcodes_longer_than_10x_are_recognised,
        test_non_numeric_column_is_dropped_not_coerced,
        test_orientation_raises_when_neither_axis_is_barcodes,
        test_barcode_transform_is_chosen_by_best_overlap,
        test_same_barcode_summed_across_runs_for_one_sample,
        test_single_run_still_works_unchanged,
        test_max_mode_keeps_strongest_run_not_sum,
        test_multi_sample_refuses_without_declaration,
        test_multi_sample_combines_two_samples_with_different_modes,
    ):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: " + ", ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
