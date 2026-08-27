"""
End-to-end test: run the full pipeline on synthetic data with known ground
truth and assert that the planted effects are recovered.

This is the test the original pipeline could not have: because the analysis was
a notebook executed by string injection, there was no way to call a stage with
known inputs and check its outputs. Every claim about correctness rested on a
human looking at a plot.

Run with:  python -m pytest tests/ -v
      or:  python tests/test_end_to_end.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_anndata import FakeAnnData                          # noqa: E402
from perturbseq_report import synthetic                        # noqa: E402
from perturbseq_report.config import (                         # noqa: E402
    THRESHOLD_KEYS, build_config, decide_run_mode,
)
from perturbseq_report.manifest import read_manifest           # noqa: E402
from perturbseq_report.pipeline import run_with_adata          # noqa: E402


def build_adata(bundle: synthetic.Bundle):
    """Assemble the bundle into an AnnData-like object.

    Uses the real AnnData when available so the test exercises the same code
    path as production; falls back to the duck-typed stand-in otherwise.
    """
    X = np.hstack([bundle.counts, bundle.guide_counts, bundle.hto_counts])
    var = pd.DataFrame(
        {
            "feature_types": (
                ["Gene Expression"] * len(bundle.gene_names)
                + ["CRISPR Guide Capture"] * len(bundle.guide_names)
                + ["Antibody Capture"] * len(bundle.hto_names)
            )
        },
        index=bundle.gene_names + bundle.guide_names + bundle.hto_names,
    )
    try:
        import anndata as ad
        return ad.AnnData(X=X, obs=bundle.obs.copy(), var=var)
    except ImportError:
        return FakeAnnData(X=X, obs=bundle.obs.copy(), var=var)


_CACHE: dict = {}


def run_pipeline(tmp: Path, **kwargs):
    """Run the pipeline once and reuse it across tests.

    Cached because a dozen assertions about one run should not mean a dozen
    runs. The cache is keyed on the kwargs, so a test that wants different
    generation parameters still gets its own run. Output goes to a persistent
    temp directory rather than the per-test one, since the result outlives the
    test that triggered it.
    """
    import tempfile

    key = tuple(sorted(kwargs.items()))
    if key in _CACHE:
        return _CACHE[key]

    workdir = Path(tempfile.mkdtemp(prefix="perturbseq-e2e-"))
    bundle = synthetic.make_bundle(seed=7, **kwargs)
    truth = bundle.truth
    out = workdir / "out"
    dragen = synthetic.write_dragen_metrics(
        workdir / "dragen", [f"LANE{i+1}" for i in range(4)]
    )
    manifest_path = synthetic.write_manifest(
        workdir / "sample_manifest.csv", workdir / "synthetic.h5ad", out,
        n_samples=2, n_lanes=2, dragen_dir=dragen,
    )
    manifest = read_manifest(manifest_path)
    # auto_thresholds=True because these tests assert on the FULL pipeline.
    # The CLI is explore-first, so a plain run would stop after QC; library
    # callers state their intent explicitly instead.
    cfg = build_config({
        "output_path": out, "verbose": False, "resample_n": 200,
        "auto_thresholds": True,
    })
    cfg.manifest_path = manifest_path
    result = run_with_adata(cfg, manifest, build_adata(bundle))
    _CACHE[key] = (result, truth, cfg)
    return _CACHE[key]


# ===========================================================================
# Tests
# ===========================================================================
def test_pipeline_runs_and_writes_report(tmp_path):
    result, truth, cfg = run_pipeline(tmp_path)

    assert result.report_path is not None and result.report_path.exists()
    html = result.report_path.read_text(encoding="utf-8")
    assert len(html) > 50_000, "report suspiciously small"
    # Self-contained: figures embedded, no external network resources.
    assert "data:image/png;base64," in html
    assert "fonts.googleapis" not in html
    assert "http://" not in html.replace("http://www.w3.org", "")
    # Registry claims match the filesystem.
    assert result.registry.verify() == [], result.registry.verify()


def test_qc_removes_low_quality_cells(tmp_path):
    result, truth, cfg = run_pipeline(tmp_path)
    removed = 1.0 - result.n_cells_after / result.n_cells_before
    # We planted 12% low-quality cells. QC should remove a comparable amount:
    # enough to catch them, not so much that it is cutting into good cells.
    assert 0.04 <= removed <= 0.45, f"removed {removed:.1%}"
    assert result.n_cells_after > 0.5 * truth.n_cells


def test_thresholds_are_recorded_with_provenance(tmp_path):
    result, truth, cfg = run_pipeline(tmp_path)
    th = result.thresholds
    assert th.is_complete()
    assert set(th.source) >= {
        "min_genes", "max_genes", "min_counts", "max_counts", "max_mito"
    }
    assert th.min_counts < th.max_counts
    assert th.min_genes < th.max_genes
    assert 0 < th.max_mito <= 100


def test_knockdown_recovers_planted_effect(tmp_path):
    result, truth, cfg = run_pipeline(tmp_path)
    kd_art = result.registry.get("perturbation", "knockdown_table")
    assert kd_art is not None and kd_art.skipped_reason is None, \
        "knockdown table was not produced"
    kd = pd.DataFrame(kd_art.data["rows"]).set_index("target_gene")

    strong = [t for t in truth.strong_targets() if t in kd.index]
    null = [t for t in truth.null_targets() if t in kd.index]
    assert strong, "no strong perturbation survived to the knockdown table"

    for t in strong:
        expected = (1.0 - truth.target_knockdown[t]) * 100.0
        observed = kd.loc[t, "pct_knockdown"]
        assert observed > 40, f"{t}: expected ~{expected:.0f}% KD, got {observed:.0f}%"
        assert abs(observed - expected) < 30, \
            f"{t}: expected ~{expected:.0f}% KD, got {observed:.0f}%"

    for t in null:
        observed = kd.loc[t, "pct_knockdown"]
        assert abs(observed) < 25, \
            f"{t} was planted with no knockdown but measured {observed:.0f}%"


def test_strong_perturbations_are_significant(tmp_path):
    result, truth, cfg = run_pipeline(tmp_path)
    kd = pd.DataFrame(
        result.registry.get("perturbation", "knockdown_table").data["rows"]
    ).set_index("target_gene")
    strong = [t for t in truth.strong_targets() if t in kd.index]
    null = [t for t in truth.null_targets() if t in kd.index]
    for t in strong:
        assert kd.loc[t, "padj_resample"] < 0.05, f"{t} not significant"
    for t in null:
        assert kd.loc[t, "padj_resample"] > 0.01, \
            f"{t} planted null but called significant (false positive)"


def test_guide_assignment_rate_near_planted(tmp_path):
    result, truth, cfg = run_pipeline(tmp_path)
    metrics = result.registry.metrics()
    observed = metrics.get("pct_cells_with_guide")
    assert observed is not None
    expected = truth.guide_capture_rate * 100
    # Ambient contamination costs some cells their purity, so the observed rate
    # should sit at or slightly below the planted capture rate -- never above.
    assert expected - 30 <= observed <= expected + 6, \
        f"planted {expected:.0f}% capture, measured {observed:.0f}% assigned"


def test_hashtag_rates_near_planted(tmp_path):
    result, truth, cfg = run_pipeline(tmp_path)
    metrics = result.registry.metrics()
    singlet = metrics.get("pct_hto_singlet")
    multiplet = metrics.get("pct_hto_multiplet")
    assert singlet is not None and multiplet is not None
    assert singlet > 45, f"singlet rate {singlet:.0f}% is implausibly low"
    # One of four populations has a deliberately broken hashtag, so ~25% of
    # cells cannot be called; the singlet rate should reflect that, not exceed it.
    assert singlet < 92, f"singlet rate {singlet:.0f}% too high given a broken hashtag"
    assert multiplet < 30, f"multiplet rate {multiplet:.0f}% implausibly high"


def test_broken_hashtag_is_flagged(tmp_path):
    result, truth, cfg = run_pipeline(tmp_path)
    art = result.registry.get("hashtags", "thresholds")
    assert art is not None
    th = pd.DataFrame(art.data["rows"]).set_index("hashtag")
    for name in truth.broken_hashtags:
        assert name in th.index
        assert not bool(th.loc[name, "well_separated"]), \
            f"{name} was planted broken but was not flagged"
    working = [h for h in th.index if h not in truth.broken_hashtags]
    assert all(bool(th.loc[h, "well_separated"]) for h in working), \
        "a working hashtag was wrongly flagged as failed"


def test_clustering_recovers_populations(tmp_path):
    result, truth, cfg = run_pipeline(tmp_path)
    metrics = result.registry.metrics()
    n_clusters = metrics.get("n_clusters")
    assert n_clusters is not None and n_clusters >= 2, \
        f"only {n_clusters} cluster(s) found; the planted populations were not resolved"


def test_sections_present_and_skips_explained(tmp_path):
    result, truth, cfg = run_pipeline(tmp_path)
    by_section = result.registry.by_section()
    for expected in ("summary", "cell_qc", "transcriptome", "guides",
                     "perturbation", "hashtags"):
        assert expected in by_section, f"section {expected} missing entirely"
    # Anything not produced must carry a reason -- never a silent omission.
    for art in result.registry:
        if art.skipped_reason is not None:
            assert len(art.skipped_reason) > 20, \
                f"{art.section}/{art.key} skipped without a useful reason"


def test_experiment_without_hashtags_degrades_gracefully(tmp_path):
    """An experiment with no hashtags must still produce a full report."""
    bundle = synthetic.make_bundle(seed=3, n_cells=1200, n_genes=400, n_htos=2)
    # Drop the hashtag block entirely.
    X = np.hstack([bundle.counts, bundle.guide_counts])
    var = pd.DataFrame(
        {"feature_types": (["Gene Expression"] * len(bundle.gene_names)
                           + ["CRISPR Guide Capture"] * len(bundle.guide_names))},
        index=bundle.gene_names + bundle.guide_names,
    )
    try:
        import anndata as ad
        adata = ad.AnnData(X=X, obs=bundle.obs.copy(), var=var)
    except ImportError:
        adata = FakeAnnData(X=X, obs=bundle.obs.copy(), var=var)

    out = tmp_path / "out_nohto"
    mp = synthetic.write_manifest(
        tmp_path / "m.csv", tmp_path / "s.h5ad", out, hto="no"
    )
    manifest = read_manifest(mp)
    cfg = build_config({"output_path": out, "verbose": False, "resample_n": 100})
    cfg.manifest_path = mp
    result = run_with_adata(cfg, manifest, adata)

    assert result.report_path.exists()
    hto_arts = [a for a in result.registry if a.section == "hashtags"]
    assert hto_arts, "hashtag section should record why it was skipped"
    assert all(a.skipped_reason for a in hto_arts if a.kind == "figure")
    # And the rest of the report is intact.
    assert result.registry.get("cell_qc", "hexbin_prefilter") is not None


def test_explore_mode_stops_after_qc(tmp_path):
    bundle = synthetic.make_bundle(seed=5, n_cells=900, n_genes=300)
    out = tmp_path / "out_explore"
    mp = synthetic.write_manifest(tmp_path / "m.csv", tmp_path / "s.h5ad", out)
    manifest = read_manifest(mp)
    cfg = build_config({"output_path": out, "verbose": False, "explore_only": True})
    cfg.manifest_path = mp
    result = run_with_adata(cfg, manifest, build_adata(bundle))

    assert result.report_path.exists()
    assert result.report_path.name == "qc_explore.html", \
        "explore output must not overwrite the full report"
    assert result.registry.get("cell_qc", "hexbin_prefilter") is not None
    assert result.registry.get("transcriptome", "umaps") is None, \
        "explore mode should not have run the transcriptome stage"


# ===========================================================================
# Explore-first workflow
# ===========================================================================
def test_explore_report_contains_the_review_panels(tmp_path):
    """The explore report must carry the counts / genes / %mito distributions."""
    bundle = synthetic.make_bundle(seed=5, n_cells=1000, n_genes=300)
    out = tmp_path / "out"
    mp = synthetic.write_manifest(tmp_path / "m.csv", tmp_path / "s.h5ad", out)
    cfg = build_config({"output_path": out, "verbose": False, "explore_only": True})
    cfg.manifest_path = mp
    result = run_with_adata(cfg, read_manifest(mp), build_adata(bundle))

    root = out / "analysis_outputs"
    for key in ("hexbin_prefilter", "hist_prefilter", "thresholds", "retention"):
        art = result.registry.get("cell_qc", key)
        assert art is not None and art.skipped_reason is None, f"missing {key}"
        if art.kind == "figure":
            assert (root / art.path).exists(), f"{key} figure not on disk"

    html = result.report_path.read_text(encoding="utf-8")
    # Figure titles and captions are HTML text; axis labels live inside the PNGs,
    # so assert on the former.
    for label in ("QC metrics and thresholds", "Marginal QC distributions",
                  "total UMI counts", "genes detected", "% mitochondrial"):
        assert label in html, f"{label!r} not named in the explore report"

    # All five thresholds must be listed, with provenance.
    rows = result.registry.get("cell_qc", "thresholds").data["rows"]
    assert {r["threshold"] for r in rows} == set(THRESHOLD_KEYS)
    assert all(r["source"] for r in rows)


def test_explore_prefills_manifest_thresholds(tmp_path):
    bundle = synthetic.make_bundle(seed=5, n_cells=1000, n_genes=300)
    out = tmp_path / "out"
    mp = synthetic.write_manifest(tmp_path / "m.csv", tmp_path / "s.h5ad", out)

    before = pd.read_csv(mp)
    assert not any(c in before.columns for c in
                   ("min_genes", "max_genes", "min_counts", "max_counts", "max_mito"))

    cfg = build_config({"output_path": out, "verbose": False, "explore_only": True})
    cfg.manifest_path = mp
    run_with_adata(cfg, read_manifest(mp), build_adata(bundle))

    after = pd.read_csv(mp)
    for col in ("min_genes", "max_genes", "min_counts", "max_counts", "max_mito"):
        assert col in after.columns, f"{col} was not added to the manifest"
        assert after[col].notna().all(), f"{col} was not filled in"
    assert len(after) == len(before), "manifest rows were lost"
    assert (out / "analysis_outputs" / "threshold_state.json").exists()


def test_two_command_workflow(tmp_path):
    """Explore, then the identical config runs the full pipeline."""
    bundle = synthetic.make_bundle(seed=5, n_cells=1200, n_genes=350)
    out = tmp_path / "out"
    mp = synthetic.write_manifest(tmp_path / "m.csv", tmp_path / "s.h5ad", out)

    # Pass 1: nothing set anywhere -> explore.
    m1 = read_manifest(mp)
    mode1 = decide_run_mode(m1.read_thresholds().as_dict(),
                            {k: None for k in THRESHOLD_KEYS})
    assert mode1.explore, mode1.reason
    cfg1 = build_config({"output_path": out, "verbose": False,
                         "explore_only": True})
    cfg1.manifest_path = mp
    r1 = run_with_adata(cfg1, m1, build_adata(bundle))
    assert r1.report_path.name == "qc_explore.html"

    # Pass 2: the manifest now carries all five -> full pipeline.
    m2 = read_manifest(mp)
    th2 = m2.read_thresholds()
    assert th2.is_complete(), "explore should have filled every threshold"
    mode2 = decide_run_mode(th2.as_dict(), {k: None for k in THRESHOLD_KEYS})
    assert not mode2.explore, mode2.reason

    cfg2 = build_config({"output_path": out, "verbose": False, "resample_n": 100})
    cfg2.manifest_path = mp
    for key in THRESHOLD_KEYS:
        setattr(cfg2.qc, key, getattr(th2, key))
        cfg2.qc.source[key] = "manifest"
    r2 = run_with_adata(cfg2, m2, build_adata(bundle))

    assert r2.report_path.name == "qc_report.html"
    assert r1.report_path.exists(), "the explore report was overwritten"
    assert r2.registry.get("transcriptome", "umaps") is not None
    # Same thresholds either way, so the same cells must survive.
    assert r1.n_cells_after == r2.n_cells_after


def test_unreviewed_thresholds_are_flagged(tmp_path):
    """A full run on untouched auto values must say so in the report."""
    bundle = synthetic.make_bundle(seed=5, n_cells=1200, n_genes=350)
    out = tmp_path / "out"
    mp = synthetic.write_manifest(tmp_path / "m.csv", tmp_path / "s.h5ad", out)

    cfg1 = build_config({"output_path": out, "verbose": False,
                         "explore_only": True})
    cfg1.manifest_path = mp
    run_with_adata(cfg1, read_manifest(mp), build_adata(bundle))

    m2 = read_manifest(mp)
    th2 = m2.read_thresholds()
    cfg2 = build_config({"output_path": out, "verbose": False, "resample_n": 100})
    cfg2.manifest_path = mp
    for key in THRESHOLD_KEYS:
        setattr(cfg2.qc, key, getattr(th2, key))
        cfg2.qc.source[key] = "manifest"
    r2 = run_with_adata(cfg2, m2, build_adata(bundle))

    note = r2.registry.get("cell_qc", "thresholds_unchanged")
    assert note is not None, "unchanged auto thresholds were not flagged"
    assert note.data.get("level") == "warn"


def test_edited_thresholds_are_recognised(tmp_path):
    bundle = synthetic.make_bundle(seed=5, n_cells=1200, n_genes=350)
    out = tmp_path / "out"
    mp = synthetic.write_manifest(tmp_path / "m.csv", tmp_path / "s.h5ad", out)

    cfg1 = build_config({"output_path": out, "verbose": False,
                         "explore_only": True})
    cfg1.manifest_path = mp
    run_with_adata(cfg1, read_manifest(mp), build_adata(bundle))

    # A human edits one threshold.
    df = pd.read_csv(mp)
    df["max_mito"] = 42.0
    df.to_csv(mp, index=False)

    m2 = read_manifest(mp)
    th2 = m2.read_thresholds()
    cfg2 = build_config({"output_path": out, "verbose": False, "resample_n": 100})
    cfg2.manifest_path = mp
    for key in THRESHOLD_KEYS:
        setattr(cfg2.qc, key, getattr(th2, key))
        cfg2.qc.source[key] = "manifest"
    r2 = run_with_adata(cfg2, m2, build_adata(bundle))

    assert r2.registry.get("cell_qc", "thresholds_unchanged") is None
    note = r2.registry.get("cell_qc", "thresholds_edited")
    assert note is not None, "the edit was not recognised"
    assert "max_mito" in note.data.get("body", "")


def test_auto_thresholds_flag_warns_loudly(tmp_path):
    bundle = synthetic.make_bundle(seed=5, n_cells=1200, n_genes=350)
    out = tmp_path / "out"
    mp = synthetic.write_manifest(tmp_path / "m.csv", tmp_path / "s.h5ad", out)
    cfg = build_config({"output_path": out, "verbose": False, "resample_n": 100,
                        "auto_thresholds": True})
    cfg.manifest_path = mp
    r = run_with_adata(cfg, read_manifest(mp), build_adata(bundle))

    assert r.report_path.name == "qc_report.html"
    assert r.registry.get("transcriptome", "umaps") is not None
    note = r.registry.get("cell_qc", "thresholds_unreviewed")
    assert note is not None and note.data.get("level") == "warn"
    assert any("not reviewed" in w for w in r.warnings)


def test_report_escapes_hostile_names(tmp_path):
    """A gene named with HTML metacharacters must not corrupt the report."""
    bundle = synthetic.make_bundle(seed=11, n_cells=800, n_genes=300)
    bundle.gene_names[50] = '<script>alert("x")</script>'
    bundle.gene_names[51] = "A&B<C"
    out = tmp_path / "out_esc"
    mp = synthetic.write_manifest(tmp_path / "m.csv", tmp_path / "s.h5ad", out)
    manifest = read_manifest(mp)
    cfg = build_config({"output_path": out, "verbose": False, "resample_n": 100})
    cfg.manifest_path = mp
    result = run_with_adata(cfg, manifest, build_adata(bundle))

    html = result.report_path.read_text(encoding="utf-8")
    assert "<script>alert" not in html, "unescaped gene name injected into the report"


def test_manifest_threshold_writeback_is_safe(tmp_path):
    """Thresholds are written back atomically, with a backup, preserving rows."""
    result, truth, cfg = run_pipeline(tmp_path)
    mp = cfg.manifest_path
    after = pd.read_csv(mp)
    assert len(after) == 4, "manifest rows were lost during write-back"
    assert {"min_genes", "max_genes", "min_counts", "max_counts",
            "max_mito"} <= set(after.columns)
    backups = list(mp.parent.glob(mp.name + ".bak-*"))
    assert backups, "no backup was taken before rewriting the manifest"
    before = pd.read_csv(backups[0])
    assert len(before) == len(after)
    assert set(before["sample"]) == set(after["sample"])


# ===========================================================================
# Already-analysed input objects
# ===========================================================================
def _analysed_adata(bundle, with_counts_layer=True):
    """An object in the state a previously-analysed h5ad arrives in."""
    X = np.hstack([bundle.counts, bundle.guide_counts, bundle.hto_counts])
    var = pd.DataFrame(
        {"feature_types": (
            ["Gene Expression"] * len(bundle.gene_names)
            + ["CRISPR Guide Capture"] * len(bundle.guide_names)
            + ["Antibody Capture"] * len(bundle.hto_names))},
        index=bundle.gene_names + bundle.guide_names + bundle.hto_names,
    )
    Xn = X / np.clip(X.sum(1, keepdims=True), 1, None) * 1e4
    rng = np.random.default_rng(0)
    a = FakeAnnData(X=np.log1p(Xn), obs=bundle.obs.copy(), var=var)
    a.uns["log1p"] = {"base": None}
    a.obs["total_counts"] = X.sum(1)
    a.obs["n_genes_by_counts"] = (X > 0).sum(1)
    a.obs["pct_counts_mt"] = rng.uniform(1, 12, a.n_obs)
    a.obs["leiden"] = rng.choice(list("0123"), a.n_obs)
    a.obsm["X_pca"] = rng.normal(size=(a.n_obs, 30))
    a.obsm["X_umap"] = rng.normal(size=(a.n_obs, 2))
    if with_counts_layer:
        a.layers["counts"] = X
    return a


def test_analysed_input_reuses_embedding(tmp_path):
    """An already-analysed h5ad must not be re-embedded or re-normalised."""
    bundle = synthetic.make_bundle(seed=7, n_cells=1200, n_genes=350)
    out = tmp_path / "out"
    mp = synthetic.write_manifest(tmp_path / "m.csv", tmp_path / "s.h5ad", out)
    cfg = build_config({"output_path": out, "verbose": False,
                        "resample_n": 50, "auto_thresholds": True})
    cfg.manifest_path = mp
    r = run_with_adata(cfg, read_manifest(mp), _analysed_adata(bundle))

    assert r.report_path.exists()
    notes = " ".join(
        a.data.get("body", "") for a in r.registry
        if a.kind == "note"
    )
    assert "REUSED" in notes, "the existing embedding was not reused"
    assert "reused" in notes.lower()
    # Cluster count must match what was in the input, not a fresh clustering.
    assert r.registry.metrics()["n_clusters"] == 4


def test_analysed_input_reads_guides_from_counts_layer(tmp_path):
    """Guide/HTO counts must come from raw counts, not log-transformed X.

    Regression test: reading them from a log-transformed X turns UMI counts
    into values of order 0-5, so the '>10 reads' gate rejects nearly every
    cell and the assignment rate collapses to ~1% -- which looks like a failed
    experiment rather than a bug.
    """
    bundle = synthetic.make_bundle(seed=7, n_cells=1200, n_genes=350)
    out = tmp_path / "out"
    mp = synthetic.write_manifest(tmp_path / "m.csv", tmp_path / "s.h5ad", out)
    cfg = build_config({"output_path": out, "verbose": False,
                        "resample_n": 50, "auto_thresholds": True})
    cfg.manifest_path = mp
    r = run_with_adata(cfg, read_manifest(mp), _analysed_adata(bundle))

    pct = r.registry.metrics()["pct_cells_with_guide"]
    expected = bundle.truth.guide_capture_rate * 100
    assert pct > expected - 30, (
        f"guide assignment {pct:.1f}% against a planted capture rate of "
        f"{expected:.0f}% -- guide counts are probably being read from "
        f"transformed values"
    )
    singlet = r.registry.metrics()["pct_hto_singlet"]
    assert singlet > 45, f"hashtag singlet rate {singlet:.1f}% implausibly low"


def test_analysed_input_without_counts_layer_warns(tmp_path):
    """No raw counts anywhere: proceed, but say the thresholds are not UMIs."""
    bundle = synthetic.make_bundle(seed=7, n_cells=900, n_genes=300)
    out = tmp_path / "out"
    mp = synthetic.write_manifest(tmp_path / "m.csv", tmp_path / "s.h5ad", out)
    cfg = build_config({"output_path": out, "verbose": False,
                        "resample_n": 50, "auto_thresholds": True})
    cfg.manifest_path = mp
    r = run_with_adata(
        cfg, read_manifest(mp), _analysed_adata(bundle, with_counts_layer=False)
    )

    assert r.report_path.exists(), "should still produce a report"
    assert any("not raw counts" in w or "NO raw-counts" in w for w in r.warnings), \
        "missing raw counts was not reported"


def test_force_recompute_ignores_prior_results(tmp_path):
    bundle = synthetic.make_bundle(seed=7, n_cells=1000, n_genes=300)
    out = tmp_path / "out"
    mp = synthetic.write_manifest(tmp_path / "m.csv", tmp_path / "s.h5ad", out)
    cfg = build_config({"output_path": out, "verbose": False,
                        "resample_n": 50, "auto_thresholds": True,
                        "force_recompute": True})
    cfg.manifest_path = mp
    r = run_with_adata(cfg, read_manifest(mp), _analysed_adata(bundle))

    notes = " ".join(a.data.get("body", "") for a in r.registry if a.kind == "note")
    assert "force-recompute" in notes
    # Must NOT claim to have reused the embedding.
    assert "will be REUSED" not in notes


def test_subsample_cells(tmp_path):
    bundle = synthetic.make_bundle(seed=7, n_cells=2000, n_genes=300)
    out = tmp_path / "out"
    mp = synthetic.write_manifest(tmp_path / "m.csv", tmp_path / "s.h5ad", out)
    cfg = build_config({"output_path": out, "verbose": False,
                        "resample_n": 30, "auto_thresholds": True,
                        "subsample_cells": 700})
    cfg.manifest_path = mp
    r = run_with_adata(cfg, read_manifest(mp), build_adata(bundle))

    assert r.n_cells_before <= 700, \
        f"subsample not applied: {r.n_cells_before} cells entered QC"
    assert r.registry.get("summary", "subsampled") is not None, \
        "the subsample must be stated in the report"


if __name__ == "__main__":
    import tempfile
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        with tempfile.TemporaryDirectory() as td:
            try:
                fn(Path(td))
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
