"""
Regression tests for v1.3.1 -- the multiple-testing fixes.

    python tests/test_v131_changes.py

Every check here locks in a number that was previously wrong. The two BH bugs
were both invisible to the existing suite, and for instructive reasons:

* ``test_de_sparse_equals_dense`` compared the paths with ``np.nanmax``, so the
  one column where they disagreed -- untested genes, NaN in the dense path and
  1.0 in the sparse one -- was masked by the very NaN that marked the
  disagreement. The check below compares NaN *patterns* explicitly.
* Nothing asserted what the BH family actually was, so applying the correction
  over 60 hand-picked genes instead of the 4,000 screened went unnoticed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"  {detail}"))
    if not cond:
        FAILURES.append(name)


def _counts(rng, n, k, dens=0.07, scale=4):
    M = (rng.random((n, k)) < dens) * rng.poisson(scale, size=(n, k))
    return np.log1p(M.astype(float))


# ===========================================================================
def test_untested_genes_are_nan_not_one() -> None:
    """A gene detected in no cell was never a hypothesis; it must not be one.

    ``benjamini_hochberg`` excludes NaN from the ranking but counts 1.0, so
    emitting 1.0 for an untestable gene inflates every other gene's adjusted p
    by the ratio of screened to testable genes.
    """
    print("\n[untested genes -> NaN, not p = 1.0]")
    from perturbseq_report.stats import (
        mannwhitney_u_columns, mannwhitney_u_sparse_columns,
    )
    from test_v121_changes import MiniCSC

    rng = np.random.default_rng(0)
    n1, n2, k = 120, 300, 30
    Bg, Br = _counts(rng, n1, k), _counts(rng, n2, k)
    zero = [3, 11, k - 1]
    Bg[:, zero] = 0.0
    Br[:, zero] = 0.0

    _u, p_dense = mannwhitney_u_columns(Bg, Br)
    _u, p_sparse = mannwhitney_u_sparse_columns(
        MiniCSC(Bg), MiniCSC(Br), n1, n2
    )
    check("dense path returns NaN for all-zero genes",
          bool(np.isnan(p_dense[zero]).all()), str(p_dense[zero]))
    check("sparse path returns NaN for all-zero genes",
          bool(np.isnan(p_sparse[zero]).all()), str(p_sparse[zero]))
    tested = [j for j in range(k) if j not in zero]
    check("testable genes still get finite p in both",
          bool(np.isfinite(p_dense[tested]).all()
               and np.isfinite(p_sparse[tested]).all()))


def test_bh_family_excludes_untestable() -> None:
    print("\n[BH family = testable genes only]")
    from perturbseq_report.stats import benjamini_hochberg, differential_expression

    rng = np.random.default_rng(1)
    n1, n2, k = 150, 500, 240
    Bg, Br = _counts(rng, n1, k), _counts(rng, n2, k)
    zero = rng.choice(k, int(0.4 * k), replace=False)
    Bg[:, zero] = 0.0
    Br[:, zero] = 0.0
    genes = [f"G{i}" for i in range(k)]

    tab = differential_expression(Bg, Br, genes, log_input=True).table
    p = tab["pvalue"].to_numpy()
    padj = tab["padj"].to_numpy()
    ok = np.isfinite(p)

    check("untested genes carry NaN padj", bool(np.isnan(padj[zero]).all()))
    check("padj equals BH over exactly the testable genes",
          bool(np.allclose(padj[ok], benjamini_hochberg(p)[ok], atol=1e-15,
                           equal_nan=True)))

    # What the old code produced, for the record: untested genes at p = 1.0.
    #
    # The inflation factor is n_screened / n_testable for any gene whose
    # adjusted value is not pulled down by the monotone step-up, and less than
    # that for the ones it is. So the honest assertion is the bound, plus the
    # fact that it really does bind somewhere -- not equality, which depends on
    # where `minimum.accumulate` happens to clip in a given random draw.
    p_old = np.where(ok, p, 1.0)
    old = benjamini_hochberg(p_old)
    ratio = old[ok] / np.maximum(padj[ok], 1e-300)
    bound = k / int(ok.sum())
    check(f"old padj was inflated, by at most {bound:.3f}x",
          bool(1.0 < float(np.nanmax(ratio)) <= bound + 1e-6),
          f"max {float(np.nanmax(ratio)):.4f}, bound {bound:.4f}")
    check("no gene's padj got smaller under the fix",
          bool((ratio >= 1.0 - 1e-12).all()))


def test_sparse_and_dense_agree_including_nan() -> None:
    """The check the original suite could not make, because nanmax hid it."""
    print("\n[sparse == dense, NaN patterns included]")
    from perturbseq_report.stats import differential_expression
    from test_v121_changes import MiniCSC

    rng = np.random.default_rng(3)
    n1, n2, k = 200, 700, 120
    Bg, Br = _counts(rng, n1, k, dens=0.06), _counts(rng, n2, k, dens=0.06)
    zero = rng.choice(k, 40, replace=False)
    Bg[:, zero] = 0.0
    Br[:, zero] = 0.0
    genes = [f"G{i}" for i in range(k)]

    d = differential_expression(Bg, Br, genes, log_input=True).table
    s = differential_expression(MiniCSC(Bg), MiniCSC(Br), genes,
                                log_input=True, block=17).table
    worst = 0.0
    pattern_ok = True
    for col in ("log2fc", "pvalue", "padj", "mean_group", "mean_ref",
                "frac_detected_group", "frac_detected_ref"):
        a, b = d[col].to_numpy(), s[col].to_numpy()
        pattern_ok &= np.array_equal(np.isnan(a), np.isnan(b))
        diff = np.abs(a - b)
        if np.isfinite(diff).any():
            worst = max(worst, float(np.nanmax(diff)))
    check("NaN pattern identical in every column", pattern_ok)
    check("every column agrees to 1e-12", worst < 1e-12, f"max {worst:.2e}")


def test_marker_padj_uses_the_screened_family() -> None:
    """Markers are corrected over every gene screened, not the top few.

    The original pre-ranked by fold change, tested ``max(n_genes * 6, 60)``,
    and applied BH with that as n -- a denominator up to ~67x too small, over a
    subset chosen using the same data as the test.
    """
    print("\n[marker padj: BH over the screened family]")
    from perturbseq_report.gex import _marker_pvalues, compute_markers
    from perturbseq_report.stats import benjamini_hochberg

    rng = np.random.default_rng(11)
    n_cells, k, n_clust = 1200, 400, 4
    labels = pd.Series(rng.integers(0, n_clust, n_cells).astype(str))
    X = rng.poisson(rng.gamma(0.6, 0.9, k), (n_cells, k)).astype(float)
    planted = {c: np.arange(c * 8, c * 8 + 8) for c in range(n_clust)}
    for c in range(n_clust):
        m = (labels == str(c)).to_numpy()
        X[np.ix_(m, planted[c])] *= 6
    # Undetected genes, as in a real object -- but not the planted markers.
    # Zeroing those would be the test sabotaging its own premise.
    all_planted = np.concatenate(list(planted.values()))
    candidates = np.setdiff1d(np.arange(k), all_planted)
    X[:, rng.choice(candidates, int(0.3 * k), replace=False)] = 0.0
    X = np.log1p(X)
    excluded = np.zeros(k, bool)

    mk = compute_markers(X, [f"G{i}" for i in range(k)], labels, excluded,
                         n_genes=6, max_genes_tested=k)
    check("markers were found", not mk.empty)
    check("planted markers dominate the result",
          float(mk["gene"].isin({f"G{i}" for i in all_planted}).mean()) > 0.5,
          f"{float(mk['gene'].isin({f'G{i}' for i in all_planted}).mean()):.3f}")
    check("only enriched genes are returned",
          bool((mk["log2fc"] > 0).all()))
    check("no NaN padj survives into the marker table",
          bool(np.isfinite(mk["padj"].to_numpy()).all()))

    # Recompute one cluster's family independently and confirm the reported
    # padj is BH over all testable screened genes, not over the returned rows.
    cl = str(mk["cluster"].iloc[0])
    m = (labels == cl).to_numpy()
    pv = _marker_pvalues(X, np.flatnonzero(m), np.flatnonzero(~m))
    expect = benjamini_hochberg(pv)
    names = np.array([f"G{i}" for i in range(k)])
    got = mk[mk["cluster"] == cl]
    idx = [int(np.flatnonzero(names == g)[0]) for g in got["gene"]]
    check(f"padj matches BH over all {int(np.isfinite(pv).sum())} testable "
          f"genes (not the {len(got)} reported)",
          bool(np.allclose(got["padj"].to_numpy(), expect[idx], atol=1e-12)))

    # And the denominator is emphatically not the row count.
    small = benjamini_hochberg(pv[idx])
    check("padj is NOT BH over just the reported rows",
          not bool(np.allclose(got["padj"].to_numpy(), small, atol=1e-12)))


def test_guide_metrics_say_what_they_measure() -> None:
    print("\n[guide metrics]")
    from perturbseq_report.guide import guide_efficiency_by_group

    per_cell = pd.DataFrame({
        "guide_is_assigned": [True, True, False, True, False, True],
        "guide_total_umis": [0, 3, 12, 40, 9, 10],
        "n_guides_detected": [0, 1, 2, 1, 1, 1],
    })
    grp = pd.Series(["a"] * 6)

    eff = guide_efficiency_by_group(per_cell, grp, min_reads=10)
    row = eff.iloc[0]
    # >= 10: the 12, 40 and 10 -> 3 of 6
    check("pct_above_min_reads honours the threshold",
          abs(float(row["pct_above_min_reads"]) - 50.0) < 1e-9,
          str(row["pct_above_min_reads"]))
    # > 0: everything except the single 0 -> 5 of 6
    check("pct_with_any_guide_umi is reported separately",
          abs(float(row["pct_with_any_guide_umi"]) - 100.0 * 5 / 6) < 1e-9,
          str(row["pct_with_any_guide_umi"]))
    check("the two are not the same number",
          float(row["pct_above_min_reads"]) != float(row["pct_with_any_guide_umi"]))

    no_thr = guide_efficiency_by_group(per_cell, grp).iloc[0]
    check("without a threshold, pct_above_min_reads is NaN rather than wrong",
          bool(np.isnan(float(no_thr["pct_above_min_reads"]))))


def test_n_targets_excludes_the_control_pool() -> None:
    print("\n[n_targets excludes NTC]")
    src = (Path(__file__).resolve().parents[1]
           / "perturbseq_report" / "guide.py").read_text()
    i = src.index("n_targets = int(")
    window = src[max(0, i - 500):i + 300]
    check("the distinct-target count filters the NTC label",
          "ntc_label" in window or "is_ntc" in window,
          "NTC pool is still counted as a target")


# ===========================================================================
def main() -> int:
    for fn in (
        test_untested_genes_are_nan_not_one,
        test_bh_family_excludes_untestable,
        test_sparse_and_dense_agree_including_nan,
        test_marker_padj_uses_the_screened_family,
        test_guide_metrics_say_what_they_measure,
        test_n_targets_excludes_the_control_pool,
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
