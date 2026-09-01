"""
Unit tests for the numerical kernels, manifest handling, modality splitting and
guide parsing.

Each numeric test checks a value derived independently of the implementation --
by hand, by an algebraic identity, or from a planted effect -- rather than
merely asserting the code returns something.

Run with:  python -m pytest tests/ -v
      or:  python tests/test_units.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_anndata import make_combined                              # noqa: E402
from perturbseq_report import stats as S                            # noqa: E402
from perturbseq_report.config import (                              # noqa: E402
    THRESHOLD_KEYS, GuideConfig, HTOConfig, ModalityConfig, QCThresholds,
    build_config, decide_run_mode,
)
from perturbseq_report.guide import GuideParser                     # noqa: E402
from perturbseq_report.manifest import ManifestError, read_manifest # noqa: E402
from perturbseq_report.modalities import split_modalities           # noqa: E402


def approx(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


# ===========================================================================
# Guide statistics
# ===========================================================================
def test_guide_stats_hand_computed():
    X = np.array(
        [
            [100, 2, 0],    # clear winner
            [50, 50, 0],    # tie -> purity 50%
            [0, 0, 0],      # no reads -> NaN ratios
            [5, 0, 0],      # single guide -> purity 100%
            [10, 1, 1],
        ],
        dtype=float,
    )
    g = S.compute_guide_stats(X)
    assert list(g.total) == [102, 100, 0, 5, 12]
    assert list(g.top1) == [100, 50, 0, 5, 10]
    assert list(g.top2) == [2, 50, 0, 0, 1]
    assert approx(g.top1_over_top2[0], 100 * 100 / 102, 1e-6)
    assert approx(g.top1_over_top2[1], 50.0, 1e-6)
    assert np.isnan(g.top1_over_top2[2]), "no-read cell must be NaN, not 0 or 100"
    assert approx(g.top1_over_top2[3], 100.0)
    assert list(g.n_detected) == [2, 2, 0, 1, 3]
    assert g.top1_index[2] == -1


def test_guide_assignment_rule():
    X = np.array([[100, 2], [50, 50], [5, 0], [20, 1]], dtype=float)
    g = S.compute_guide_stats(X)
    out = S.assign_guides(g, ["gA", "gB"], min_reads=10, purity_min=75.0)
    # row 0: 102 reads, 98% pure -> assigned
    # row 1: 100 reads, 50% pure -> not assigned (impure)
    # row 2: 5 reads -> not eligible
    # row 3: 21 reads, 95% pure -> assigned
    assert list(out["guide_is_assigned"]) == [True, False, False, True]
    assert list(out["assigned_guide"]) == ["gA", None, None, "gA"]


def test_assignment_sweep_is_monotone():
    rng = np.random.default_rng(0)
    X = rng.poisson(5, size=(500, 6)).astype(float)
    X[np.arange(500), rng.integers(0, 6, 500)] += 60
    g = S.compute_guide_stats(X)
    sweep = S.assignment_sweep(g, [0, 25, 50, 75, 90, 100], min_reads=10)
    frac = sweep["frac_assigned"].to_numpy()
    assert np.all(np.diff(frac) <= 1e-12), "raising the purity cut must not assign more"


# ===========================================================================
# Energy distance
# ===========================================================================
def test_energy_distance_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(400, 5))
    B = rng.normal(size=(400, 5))
    assert abs(S.energy_distance(A, B)) < 0.3


def test_energy_distance_matches_closed_form():
    """For a pure mean shift d in p dimensions, E = 2*||d||^2 in expectation."""
    rng = np.random.default_rng(1)
    p, shift = 5, 2.0
    A = rng.normal(size=(4000, p))
    B = rng.normal(size=(4000, p)) + shift
    expected = 2.0 * (p * shift**2)
    observed = S.energy_distance(A, B)
    assert abs(observed - expected) / expected < 0.05, (observed, expected)


def test_energy_distance_permutation_pvalue():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(120, 4))
    B = rng.normal(size=(120, 4))
    _, p_null = S.edistance_permutation_pvalue(A, B, n_perm=200, random_state=0)
    C = rng.normal(size=(120, 4)) + 1.5
    _, p_real = S.edistance_permutation_pvalue(A, C, n_perm=200, random_state=0)
    assert p_null > 0.05, f"identical distributions gave p={p_null}"
    assert p_real < 0.02, f"clearly shifted distributions gave p={p_real}"


# ===========================================================================
# Statistics
# ===========================================================================
def test_benjamini_hochberg_known_values():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212,
                  0.216])
    adj = S.benjamini_hochberg(p)
    assert approx(adj[0], 0.01, 1e-9)
    assert approx(adj[1], 0.04, 1e-9)
    assert np.all(np.diff(adj) >= -1e-12), "BH output must be monotone"
    assert np.all(adj <= 1.0)


def test_benjamini_hochberg_preserves_nan():
    p = np.array([0.01, np.nan, 0.5])
    adj = S.benjamini_hochberg(p)
    assert np.isnan(adj[1]), "NaN must stay NaN, not be treated as p=1"
    assert np.isfinite(adj[0]) and np.isfinite(adj[2])


def test_mannwhitney_separated_groups():
    a = np.arange(1, 21, dtype=float)
    b = np.arange(31, 51, dtype=float)
    _, p = S.mannwhitney_u(a, b)
    assert p < 1e-4
    _, p_same = S.mannwhitney_u(a, a.copy())
    assert p_same > 0.5


def test_spearman_matrix_perfect_monotone():
    M = np.column_stack([
        np.arange(20, dtype=float),
        np.arange(20, dtype=float) ** 3,        # monotone -> rho = 1
        -np.arange(20, dtype=float),            # anti-monotone -> rho = -1
    ])
    C = S.spearman_matrix(M)
    assert approx(C[0, 1], 1.0, 1e-9)
    assert approx(C[0, 2], -1.0, 1e-9)


def test_jaccard_matrix():
    J = S.jaccard_matrix([{1, 2, 3}, {2, 3, 4}, set()])
    assert approx(J[0, 1], 0.5)      # |{2,3}| / |{1,2,3,4}|
    assert approx(J[0, 0], 1.0)
    assert approx(J[2, 2], 0.0)      # empty set has no self-similarity


def test_gini():
    assert approx(S.gini(np.ones(10)), 0.0, 1e-9)
    assert S.gini(np.array([0.0, 0, 0, 100])) > 0.7


def test_safe_divide_handles_nan_denominator():
    """The original's `if total else nan` guard passed NaN through, because
    bool(np.nan) is True."""
    out = S.safe_divide(np.array([1.0, 1.0, 1.0]), np.array([2.0, 0.0, np.nan]))
    assert approx(out[0], 0.5)
    assert np.isnan(out[1]) and np.isnan(out[2])


# ===========================================================================
# Thresholds
# ===========================================================================
def test_mad_bounds_on_lognormal():
    rng = np.random.default_rng(0)
    x = rng.lognormal(mean=8.0, sigma=0.4, size=20000)
    lo, hi = S.mad_bounds(x, 3.0, 5.0, log=True)
    assert lo > 0, "log-scale bounds must never produce a negative lower gate"
    assert lo < np.median(x) < hi
    frac_kept = float(((x >= lo) & (x <= hi)).mean())
    assert frac_kept > 0.97, f"3/5 MAD kept only {frac_kept:.3f} of clean data"


def test_otsu_finds_valley():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.4, 2000), rng.normal(6, 0.4, 2000)])
    t = S.otsu_threshold(x)
    assert 1.5 < t < 4.5, f"threshold {t} is not between the two modes"


def test_split_bimodal_is_deterministic_and_correct():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.3, 900), rng.normal(5, 0.4, 100)])
    bg1, m0a, m1a = S.split_bimodal_1d(x, random_state=0)
    bg2, m0b, m1b = S.split_bimodal_1d(x, random_state=999)
    assert np.array_equal(bg1, bg2), "result must not depend on the seed"
    assert approx(m0a, m0b) and approx(m1a, m1b)
    assert 0.85 < bg1.mean() < 0.95
    assert m0a < m1a


def test_split_bimodal_degenerate_is_all_background():
    bg, m0, m1 = S.split_bimodal_1d(np.full(100, 3.0))
    assert bg.all(), "a constant feature must not produce positive calls"


# ===========================================================================
# CLR
# ===========================================================================
def test_clr_by_feature_centres_each_column():
    rng = np.random.default_rng(0)
    X = rng.poisson(10, size=(200, 4)).astype(float)
    Z = S.clr_by_feature(X)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-9)
    # Adding a constant multiple to one hashtag must not change other columns.
    X2 = X.copy()
    X2[:, 0] += 50
    Z2 = S.clr_by_feature(X2)
    assert np.allclose(Z[:, 1:], Z2[:, 1:], atol=1e-9)


def test_clr_by_cell_centres_each_row():
    rng = np.random.default_rng(0)
    X = rng.poisson(10, size=(50, 6)).astype(float)
    Z = S.clr_by_cell(X)
    assert np.allclose(Z.mean(axis=1), 0.0, atol=1e-9)


# ===========================================================================
# Knockdown
# ===========================================================================
def test_percent_knockdown_inverts_log_before_averaging():
    """Averaging in log space understates knockdown; this checks we don't."""
    control_lin = np.array([10.0, 12.0, 11.0, 9.0])
    pert_lin = np.array([1.0, 1.0, 2.0, 0.0])
    res = S.percent_knockdown(np.log1p(pert_lin), np.log1p(control_lin),
                              log_input=True)
    expected = (1 - pert_lin.mean() / control_lin.mean()) * 100
    assert approx(res["pct_knockdown"], expected, 1e-6)
    # The naive log-space calculation gives a materially different answer.
    naive = (1 - np.log1p(pert_lin).mean() / np.log1p(control_lin).mean()) * 100
    assert abs(naive - expected) > 5, "test case does not discriminate the two"


def test_percent_knockdown_undefined_when_target_absent():
    res = S.percent_knockdown(np.zeros(10), np.zeros(10), log_input=True)
    assert np.isnan(res["pct_knockdown"]), \
        "knockdown of an undetectable gene must be NaN, not 0 or 100"


def test_resampling_test_calibration():
    """Under the null the resampling p-value must not be systematically small."""
    rng = np.random.default_rng(0)
    pool = np.log1p(rng.poisson(4, size=3000).astype(float))
    ps = []
    for i in range(30):
        idx = rng.choice(pool.size, 80, replace=False)
        ps.append(
            S.resampling_test(pool[idx], pool, n_resample=300, random_state=i)[
                "p_resample"
            ]
        )
    ps = np.array(ps)
    assert (ps < 0.05).mean() < 0.20, \
        f"null p-values rejected {100 * (ps < 0.05).mean():.0f}% of the time"


def test_resampling_test_detects_real_effect():
    rng = np.random.default_rng(0)
    ctrl = np.log1p(rng.poisson(20, size=2000).astype(float))
    pert = np.log1p(rng.poisson(3, size=200).astype(float))
    res = S.resampling_test(pert, ctrl, n_resample=500, random_state=0)
    assert res["p_resample"] < 0.01
    assert res["log2fc"] < -1.0


def test_deg_selection_and_ranking():
    de = pd.DataFrame(
        {
            "gene": ["OWN", "A", "B", "C", "D"],
            "log2fc": [-3.0, 2.0, 0.2, -1.5, 4.0],
            "padj": [0.001, 0.002, 0.001, 0.01, 0.20],
            "low_expression": [False, False, False, False, True],
        }
    )
    sel = S.select_degs(de, padj_max=0.05, abs_log2fc_min=0.5)
    genes = set(sel["gene"])
    assert "B" not in genes, "|log2FC| below the cut must be excluded"
    assert "D" not in genes, "non-significant gene must be excluded"
    ranked = S.rank_degs(sel, top_n=3, always_first="OWN")
    assert ranked["gene"].iloc[0] == "OWN", "target gene must be hoisted first"


# ===========================================================================
# Manifest
# ===========================================================================
def _write(tmp: Path, text: str, name: str = "m.csv") -> Path:
    p = tmp / name
    p.write_text(text, encoding="utf-8")
    return p


GOOD = (
    "sample,h5ad_path,output_path,gRNA_method,HTO,cell_input,prefix,dragen_path\n"
    "s1,data.h5ad,out,PCR,yes,1000,P1,d1\n"
    "s1,data.h5ad,out,PCR,yes,1000,P2,d2\n"
    "s2,data.h5ad,out,direct,yes,1000,P3,d3\n"
)


def test_manifest_reads_and_resolves(tmp_path):
    m = read_manifest(_write(tmp_path, GOOD))
    assert m.n_samples == 2 and m.n_runs == 3
    assert m.sample_col == "sample"
    # Relative paths resolve against the MANIFEST's directory, not the CWD.
    assert m.h5ad_path == (tmp_path / "data.h5ad").resolve()
    assert m.declares_hto() is True
    assert "gRNA_method" in m.condition_columns()
    assert len(m.dragen_runs()) == 3


def test_dragen_runs_root_override_matches_selected_dir_itself(tmp_path):
    """ICA's picker selects a whole directory, not "the parent of one" --
    if the manifest's dragen_path basename IS the selected root (user
    picked the run's own output dir directly), root/basename must not be
    used (doubles the last path segment into a nonexistent dir)."""
    real_dir = tmp_path / "some" / "long" / "dragen_output"
    real_dir.mkdir(parents=True)
    manifest_csv = (
        "sample,h5ad_path,output_path,prefix,dragen_path\n"
        "s1,data.h5ad,out,P1,/original/unmounted/path/dragen_output\n"
    )
    m = read_manifest(_write(tmp_path, manifest_csv, name="m2.csv"))
    runs = m.dragen_runs(root_override=[real_dir])
    assert runs[0]["dragen_path"] == real_dir


def test_manifest_strips_padded_headers(tmp_path):
    """A padded header silently reported zero samples in the original."""
    padded = GOOD.replace("sample,", " sample ,", 1)
    m = read_manifest(_write(tmp_path, padded))
    assert m.n_samples == 2, "padded header broke sample parsing"


def test_manifest_rejects_inconsistent_global_column(tmp_path):
    bad = GOOD.replace("s2,data.h5ad,out,", "s2,other.h5ad,out,")
    try:
        read_manifest(_write(tmp_path, bad))
    except ManifestError as exc:
        assert "identical on every row" in str(exc)
    else:
        raise AssertionError("differing h5ad_path should have been rejected")


def test_manifest_rejects_missing_required_column(tmp_path):
    bad = GOOD.replace(",output_path", ",other_col")
    try:
        read_manifest(_write(tmp_path, bad))
    except ManifestError as exc:
        assert "output_path" in str(exc)
    else:
        raise AssertionError("missing output_path should have been rejected")


def test_manifest_tab_delimited(tmp_path):
    m = read_manifest(_write(tmp_path, GOOD.replace(",", "\t"), "m.tsv"))
    assert m.n_samples == 2 and m.delimiter == "\t"


def test_manifest_blank_tokens_treated_as_missing(tmp_path):
    text = GOOD.replace(",yes,1000,P1,d1", ",yes,1000,P1,NA")
    m = read_manifest(_write(tmp_path, text))
    assert len(m.dragen_runs()) == 2, "'NA' should count as a blank cell"


def test_dragen_runs_share_cells_undeclared_is_none(tmp_path):
    """No column at all -> None, not a default guess either way."""
    m = read_manifest(_write(tmp_path, GOOD))
    assert m.dragen_runs_share_cells("s1") is None
    assert m.dragen_runs_share_cells("s2") is None


def test_dragen_runs_share_cells_declared_per_sample(tmp_path):
    """s1 (2 runs) declared yes, s2 (1 run) declared no -- read back correctly."""
    text = (
        "sample,h5ad_path,output_path,gRNA_method,HTO,cell_input,prefix,"
        "dragen_path,dragen_runs_share_cells\n"
        "s1,data.h5ad,out,PCR,yes,1000,P1,d1,yes\n"
        "s1,data.h5ad,out,PCR,yes,1000,P2,d2,yes\n"
        "s2,data.h5ad,out,direct,yes,1000,P3,d3,no\n"
    )
    m = read_manifest(_write(tmp_path, text))
    assert m.dragen_runs_share_cells("s1") is True
    assert m.dragen_runs_share_cells("s2") is False
    # A declaration about library preparation, not a biological condition --
    # must not be picked up as a comparison axis even though it varies by
    # sample here.
    assert "dragen_runs_share_cells" not in m.condition_columns()
    assert "dragen_runs_share_cells" not in m.metadata_columns()


def test_dragen_runs_share_cells_rejects_conflicting_rows(tmp_path):
    """s1's two run-rows disagreeing must fail loudly, not average out."""
    text = (
        "sample,h5ad_path,output_path,gRNA_method,HTO,cell_input,prefix,"
        "dragen_path,dragen_runs_share_cells\n"
        "s1,data.h5ad,out,PCR,yes,1000,P1,d1,yes\n"
        "s1,data.h5ad,out,PCR,yes,1000,P2,d2,no\n"
    )
    m = read_manifest(_write(tmp_path, text))
    try:
        m.dragen_runs_share_cells("s1")
    except ManifestError as exc:
        assert "conflicting" in str(exc)
    else:
        raise AssertionError("conflicting yes/no for one sample should raise")


def test_manifest_threshold_roundtrip_and_backup(tmp_path):
    p = _write(tmp_path, GOOD)
    m = read_manifest(p)
    th = QCThresholds(min_genes=100, max_genes=200, min_counts=300,
                      max_counts=400, max_mito=5)
    backup = m.write_thresholds(th)
    assert backup is not None and backup.exists()
    again = read_manifest(p)
    assert again.read_thresholds().min_genes == 100
    assert len(again.df) == 3, "rows must survive the rewrite"


def test_manifest_rejects_conflicting_thresholds(tmp_path):
    text = GOOD.rstrip("\n").replace("prefix,dragen_path", "prefix,dragen_path,min_genes")
    lines = text.split("\n")
    lines[1] += ",100"
    lines[2] += ",900"
    lines[3] += ",100"
    m = read_manifest(_write(tmp_path, "\n".join(lines) + "\n"))
    try:
        m.read_thresholds()
    except ManifestError as exc:
        assert "conflicting" in str(exc)
    else:
        raise AssertionError("conflicting per-row thresholds should be rejected")


# ===========================================================================
# Modality splitting
# ===========================================================================
def test_modality_layouts_all_resolve():
    mc, gc = ModalityConfig(), GuideConfig()
    for layout in ("feature_types", "obsm", "obsm_dict_uns", "obs_cols"):
        ad = make_combined(layout=layout)
        r = split_modalities(ad, mc, gc.guide_id_regexes)
        assert r.gex.n_vars == 50, layout
        assert r.guide.n_features == 8, layout
        assert r.hto.n_features == 4, layout
        assert r.guide.names[0].startswith("TGT0"), (layout, r.guide.names[:2])
        assert r.hto.names == ["HTO1", "HTO2", "HTO3", "HTO4"], layout


def test_modality_index_keyed_uns_dict():
    """uns[key] = {0: 'name', ...} silently produced integer names before."""
    ad = make_combined(layout="obsm_dict_uns")
    r = split_modalities(ad, ModalityConfig(), GuideConfig().guide_id_regexes)
    assert not any(n.isdigit() for n in r.guide.names), \
        f"index-keyed uns dict yielded numeric names: {r.guide.names[:3]}"


def test_modality_absent_reports_reason():
    ad = make_combined(layout="gex_only")
    r = split_modalities(ad, ModalityConfig(), GuideConfig().guide_id_regexes)
    assert not r.has_guides and not r.has_hto
    assert "not found" in r.guide.source
    assert "obsm" in r.guide.source, "the reason should name what was checked"


def test_modality_excludes_clr_obs_columns():
    ad = make_combined(layout="obs_cols")
    r = split_modalities(ad, ModalityConfig(), GuideConfig().guide_id_regexes)
    assert r.hto.n_features == 4, "the *_CLR companion columns must be excluded"


# ===========================================================================
# Guide ID parsing
# ===========================================================================
def test_guide_parser_targets_and_controls():
    p = GuideParser(GuideConfig())
    cases = {
        "TP53|ENSG00000141510|ACGT": ("TP53", "ENSG00000141510", False),
        "TP53_ENSG00000141510_sg1": ("TP53", "ENSG00000141510", False),
        "MYC_singleguide": ("MYC", None, False),
        "KRAS_1_ACGTACGTACGTACGTA": ("KRAS", None, False),
        "EGFR_sg2": ("EGFR", None, False),
        "NonTargetingControl_1_ACGT": ("NTC", None, True),
        "NTC_003": ("NTC", None, True),
        "ONE_INTERGENIC_SITE": ("NTC", None, True),
        "safe_harbor_1": ("NTC", None, True),
        "scrambled_2": ("NTC", None, True),
    }
    for gid, (gene, ensg, is_ntc) in cases.items():
        r = p.parse(gid)
        assert r.target_gene == gene, (gid, r.target_gene)
        assert r.target_ensg == ensg, (gid, r.target_ensg)
        assert r.is_ntc == is_ntc, (gid, r.is_ntc)


def test_guide_parser_does_not_overmatch_ntc():
    """A gene whose name merely contains 'ntc' must not become a control."""
    p = GuideParser(GuideConfig())
    assert not p.parse("PNTCX_sg1").is_ntc
    assert p.parse("PNTCX_sg1").target_gene == "PNTCX"


def test_guide_parser_dual_guide_detection():
    p = GuideParser(GuideConfig())
    dual = p.parse_all(["G1.iBAR.1", "G1.iBAR.2", "G2.iBAR.1", "G2.iBAR.2"])
    assert p.detect_dual_guide(dual)
    # The common single-guide naming GENE_1 / GENE_2 must NOT look dual-guide.
    single = p.parse_all(["A_1", "A_2", "B_1", "B_2"])
    assert not p.detect_dual_guide(single), \
        "bare numeric suffixes were misread as iBAR positions"


def test_guide_parser_ibar_targets():
    p = GuideParser(GuideConfig())
    r = p.parse("BRCA1.iBAR.2")
    assert r.target_gene == "BRCA1" and r.construct == "BRCA1" and r.ibar == "2"


# ===========================================================================
# Config
# ===========================================================================
def test_config_flat_keys_route_to_subconfigs():
    cfg = build_config({"min_genes": 800, "purity_min": 60.0, "n_pcs": 20})
    assert cfg.qc.min_genes == 800
    assert cfg.guide.purity_min == 60.0
    assert cfg.embedding.n_pcs == 20


def test_config_rejects_unknown_key():
    try:
        build_config({"not_a_real_setting": 1})
    except ValueError as exc:
        assert "Unknown configuration key" in str(exc)
    else:
        raise AssertionError("unknown config keys should be rejected")


def test_config_nested_form():
    cfg = build_config({"qc": {"max_mito": 12.0}, "figures": {"dpi": 120}})
    assert cfg.qc.max_mito == 12.0 and cfg.figures.dpi == 120


# ===========================================================================
# Run-mode decision (explore-first)
# ===========================================================================
NONE_TH = {k: None for k in THRESHOLD_KEYS}
FULL_TH = {k: 1.0 for k in THRESHOLD_KEYS}


def test_mode_explores_when_nothing_is_set():
    """The default for a fresh manifest is the QC review step, not a full run."""
    m = decide_run_mode(NONE_TH, NONE_TH)
    assert m.explore
    assert m.thresholds_from == "none yet"
    assert len(m.missing) == 5


def test_mode_runs_full_when_manifest_complete():
    m = decide_run_mode(FULL_TH, NONE_TH)
    assert not m.explore
    assert m.thresholds_from == "manifest"
    assert m.missing == ()


def test_mode_explores_on_partially_filled_manifest():
    """A half-filled block is an unfinished edit, not an instruction to run.

    The previous pipeline ran the full analysis when given a single threshold,
    silently defaulting the other four.
    """
    partial = dict(NONE_TH, min_genes=500.0)
    m = decide_run_mode(partial, NONE_TH)
    assert m.explore
    assert set(m.missing) == {"max_genes", "min_counts", "max_counts", "max_mito"}
    assert "1 of 5" in m.reason


def test_mode_cli_flag_runs_full_and_says_what_is_derived():
    m = decide_run_mode(NONE_TH, dict(NONE_TH, max_mito=10.0))
    assert not m.explore
    assert m.thresholds_from == "cli"
    assert "max_mito" in m.reason
    assert "derived from the data" in m.reason


def test_mode_auto_flag_runs_full():
    m = decide_run_mode(NONE_TH, NONE_TH, auto_flag=True)
    assert not m.explore
    assert m.thresholds_from == "auto"


def test_mode_explore_flag_wins_over_complete_manifest():
    m = decide_run_mode(FULL_TH, NONE_TH, explore_flag=True)
    assert m.explore, "--explore must force the review step"


def test_mode_explore_flag_wins_over_auto_flag():
    m = decide_run_mode(NONE_TH, NONE_TH, explore_flag=True, auto_flag=True)
    assert m.explore, "--explore is the more conservative of the two"


def test_report_filenames_differ_by_mode():
    cfg = build_config({"output_path": "/tmp/x"})
    assert cfg.report_path.name == "qc_report.html"
    cfg.explore_only = True
    assert cfg.report_path.name == "qc_explore.html", \
        "a full run must not be able to overwrite the explore report"


if __name__ == "__main__":
    import tempfile
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        needs_tmp = "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]
        try:
            if needs_tmp:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
            failed += 1
        except Exception:
            print(f"ERROR {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
