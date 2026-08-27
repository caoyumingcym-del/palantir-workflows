"""
Regression tests for v1.3.0.

    python tests/test_v130_changes.py

Covers the 20 Aug review: batch_key and harmony (A1/A4), covariate regression
(A2), the z-score clip (A3), cluster merging (A5b), performance wiring (A5c),
the doublet gate (A6), HTO normalisation (A7), and the Part B control-pool and
comparability work.

Several checks here are deliberately *source* assertions rather than behavioural
ones. The bugs this version fixes were all of the same shape -- a setting that
reads correctly and is wired to nothing, or a docstring that claims more than
the code does. A behavioural test cannot catch "nobody calls this"; reading the
source can.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1] / "perturbseq_report"

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"  {detail}"))
    if not cond:
        FAILURES.append(name)


def src(name: str) -> str:
    return (ROOT / name).read_text()


# ===========================================================================
# A1 / A4 -- batch_key and harmony
# ===========================================================================
def test_batch_key_dominance() -> None:
    print("\n[A1: pick_batch_key implements the check its docstring promised]")
    from perturbseq_report.gex import pick_batch_key

    # The exact case the old code let through: two levels, 95/5.
    obs = pd.DataFrame({
        "sample": ["A"] * 950 + ["B"] * 50,
        "lane": ["1"] * 500 + ["2"] * 500,
    })
    key, reasons = pick_batch_key(obs, ("sample", "lane"), 0.90)
    check("a 95/5 split is rejected as a batch", key == "lane", f"got {key!r}")
    check("the rejection is explained", any("dominance" in r for r in reasons))

    key, _ = pick_batch_key(obs, ("sample",), 0.90)
    check("with no alternative it returns None rather than a bad key",
          key is None, f"got {key!r}")

    key, _ = pick_batch_key(obs, ("sample",), 0.99)
    check("the limit is configurable", key == "sample")

    one = pd.DataFrame({"sample": ["A"] * 100})
    check("a single-level column is still rejected",
          pick_batch_key(one, ("sample",), 0.90)[0] is None)

    check("a missing column is reported, not silently skipped",
          any("not present" in r
              for r in pick_batch_key(one, ("nope",), 0.9)[1]))


def test_harmony_is_opt_in() -> None:
    print("\n[A4: harmony is off by default and refused on a condition column]")
    from perturbseq_report.config import PipelineConfig
    from perturbseq_report.gex import resolve_batch_correction

    cfg = PipelineConfig()
    check("batch_correct defaults to 'none'",
          cfg.embedding.batch_correct == "none",
          cfg.embedding.batch_correct)

    obs = pd.DataFrame({"sample": ["A"] * 500 + ["B"] * 500})

    notes: list[str] = []
    key, may = resolve_batch_correction(obs, cfg.embedding, [], notes)
    check("default run resolves a key but does not correct",
          key == "sample" and may is False, f"{key!r} {may}")
    check("the resolved key is reported even though nothing uses it",
          any("resolved to 'sample'" in n for n in notes))
    check("the report says correction was not applied",
          any("No batch correction was applied" in n for n in notes))

    # 'auto' used to mean "harmony whenever possible" and was the default.
    ecfg_auto = PipelineConfig().embedding
    ecfg_auto.batch_correct = "auto"
    notes = []
    _k, may = resolve_batch_correction(obs, ecfg_auto, [], notes)
    check("'auto' no longer means harmony", may is False)
    check("'auto' being read as 'none' is stated",
          any("read as 'none'" in n for n in notes))

    ecfg_h = PipelineConfig().embedding
    ecfg_h.batch_correct = "harmony"
    notes = []
    _k, may = resolve_batch_correction(obs, ecfg_h, [], notes)
    check("explicit --batch-correct harmony is honoured", may is True)

    notes = []
    _k, may = resolve_batch_correction(obs, ecfg_h, ["sample"], notes)
    check("harmony is REFUSED when the key is a declared condition",
          may is False)
    check("the refusal is explicit in the report",
          any("REFUSED" in n for n in notes))
    check("the refusal does not stop the run (returns, does not raise)",
          _k == "sample")


def test_hvg_is_not_batch_aware_by_default() -> None:
    print("\n[A1: HVG selection stops being batch-aware by accident]")
    from perturbseq_report.config import PipelineConfig

    cfg = PipelineConfig()
    check("hvg_batch_key exists and is None by default",
          cfg.embedding.hvg_batch_key is None)

    s = src("gex.py")
    check("highly_variable_genes no longer receives the resolved batch_key",
          "batch_key=batch_key if batch_key else None" not in s)
    check("it receives hvg_batch instead", "batch_key=hvg_batch" in s)
    check("there are exactly two HVG call sites, both using hvg_batch",
          s.count("batch_key=hvg_batch") == 2,
          str(s.count("batch_key=hvg_batch")))
    check("--hvg-batch-key is exposed", "--hvg-batch-key" in src("cli.py"))


def test_edistance_space_is_stated() -> None:
    print("\n[A4: E-distance says which space it was measured in]")
    s = src("perturb.py")
    check("run_perturbation_stage takes batch_corrected",
          "batch_corrected: str = \"none\"" in s)
    check("a note about the E-distance space is registered",
          "edist_space" in s)
    check("the uncorrected case is stated positively",
          "nothing has been integrated" in s)
    check("pipeline passes the embedding's correction state through",
          "batch_corrected=emb.batch_corrected" in src("pipeline.py"))


# ===========================================================================
# A2 / A3 / A5b -- regression, clip, merging
# ===========================================================================
def test_regress_out_default_empty() -> None:
    print("\n[A2: nothing is regressed out unless asked]")
    from perturbseq_report.cli import _collect_overrides, build_parser
    from perturbseq_report.config import PipelineConfig

    check("default is empty", tuple(PipelineConfig().embedding.regress_out) == ())

    p = build_parser()
    o = _collect_overrides(p.parse_args(["--manifest", "m.csv", "--regress-qc"]))
    check("--regress-qc gives depth and %mito only",
          tuple(o["regress_out"]) == ("total_counts", "pct_counts_mt"),
          str(o.get("regress_out")))
    check("cell-cycle regression is not reachable from the CLI",
          "S_score" not in src("cli.py"))
    check("cell-cycle SCORING is still on",
          PipelineConfig().embedding.score_cell_cycle is True)

    s = src("gex.py")
    check("the no-covariate case is reported, not silent",
          "No covariates were regressed out" in s)
    check("the configured-but-absent case is reported too",
          "none of\n" in s or "were present on obs" in s)


def test_scale_clip_documented() -> None:
    print("\n[A3: the z-score clip is explained and measurable]")
    from perturbseq_report.config import PipelineConfig
    from perturbseq_report.gex import _regress_and_scale

    check("still 10 by default", PipelineConfig().embedding.scale_max_value == 10.0)
    s = src("gex.py")
    check("the report says where 10 comes from", "ScaleData(scale.max = 10)" in s)
    check("the number of clipped entries is reported", "sit at the cap" in s)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)) * 50
    Z = _regress_and_scale(X, None, 2.0)
    check("clipping is applied", float(np.max(np.abs(Z))) <= 2.0 + 1e-9)
    Zn = _regress_and_scale(X, None, None)
    check("max_value=None disables the clip without crashing",
          float(np.max(np.abs(Zn))) > 2.0)


def test_no_cluster_merging() -> None:
    print("\n[A5b: small clusters are reported, not absorbed]")
    from perturbseq_report.config import PipelineConfig
    from perturbseq_report.gex import merge_tiny_clusters, small_cluster_notes

    ecfg = PipelineConfig().embedding
    check("min_cluster_frac defaults to 0", ecfg.min_cluster_frac == 0.0)

    labels = np.array(["0"] * 900 + ["1"] * 95 + ["2"] * 5)
    pca = np.random.default_rng(0).normal(size=(1000, 5))
    out, notes = merge_tiny_clusters(labels, pca, 0.0)
    check("with 0.0 nothing is merged",
          len(set(out)) == 3 and not notes, f"{len(set(out))} clusters")

    # The merge code stays available for anyone who explicitly wants it.
    out2, notes2 = merge_tiny_clusters(labels, pca, 0.05)
    check("an explicit threshold still merges", len(set(out2)) < 3)

    msgs = small_cluster_notes(labels, ecfg)
    check("fragmentation is reported", any("2" in m for m in msgs), str(msgs))
    check("the note explains why nothing was merged",
          any("NOT merged" in m for m in msgs))

    big = np.array(["0"] * 500 + ["1"] * 500)
    check("a clean split reports no fragmentation",
          any("no cluster fragmentation" in m
              for m in small_cluster_notes(big, ecfg)))


# ===========================================================================
# A5c -- performance
# ===========================================================================
def test_performance_wiring() -> None:
    print("\n[A5c: the settings that looked live are now live]")
    from perturbseq_report.config import PipelineConfig

    cfg = PipelineConfig()
    check("n_jobs defaults to all cores", cfg.n_jobs == -1, str(cfg.n_jobs))
    check("copy_input defaults to False", cfg.embedding.copy_input is False)

    s = src("gex.py")
    check("cfg.n_jobs is actually assigned to scanpy",
          "sc.settings.n_jobs" in s)
    check("PCA uses the randomized solver",
          'svd_solver="randomized"' in s)
    check("there is a fallback if the solver is unsupported",
          s.count("sc.tl.pca(") >= 2, str(s.count("sc.tl.pca(")))

    p = src("pipeline.py")
    check("_step reports the previous step's duration", "_fmt_duration" in p)
    check("durations are only printed once a step is slow enough",
          "elapsed >= 1.0" in p)

    from perturbseq_report.pipeline import _fmt_duration
    check("duration formatting: seconds", _fmt_duration(9.4) == "9.4s")
    check("duration formatting: minutes", _fmt_duration(125) == "2m05s")
    check("duration formatting: hours", _fmt_duration(7300) == "2h01m")


# ===========================================================================
# A6 -- doublets
# ===========================================================================
def test_doublet_gate() -> None:
    print("\n[A6: off means off]")
    from perturbseq_report.config import PipelineConfig

    check("still off by default",
          PipelineConfig().embedding.detect_doublets is False)

    s = src("gex.py")
    check("the scanpy-path fallback is gated on detect_doublets",
          'if ecfg.detect_doublets and "predicted_doublet" not in obs.columns:' in s)
    check("the ungated form is gone",
          'if "predicted_doublet" not in obs.columns:\n        score, call' not in s)
    check("the skipped panel is registered with a reason",
          '"transcriptome", "doublets"' in s)
    check("the note warns that the fallback rate is not a measurement",
          "near 5-10%" in s or "5-10%" in s)
    check("remove_doublets dead config is gone",
          "remove_doublets" not in s and "remove_doublets" not in src("config.py"))

    # Both paths must use the same wording, so a reader cannot tell which
    # detector ran from the tone of the note alone.
    check("one shared note template for the fallback",
          s.count("DOUBLET_FALLBACK_NOTE") >= 3,
          str(s.count("DOUBLET_FALLBACK_NOTE")))


# ===========================================================================
# A7 -- HTO normalisation
# ===========================================================================
def test_hto_normalisation() -> None:
    print("\n[A7: CLR naming, the real Seurat transform, and a comparison]")
    from perturbseq_report.config import HTOConfig
    from perturbseq_report.stats import (
        clr_by_cell, clr_by_feature, clr_true_seurat,
    )

    s = src("stats.py")
    check("the docstring no longer claims to BE Seurat's CLR",
          "This is Seurat's ``NormalizeData" not in s)
    check("the difference from Seurat is spelled out",
          "not the same transform" in s or "differs from" in s)

    check("normalisation is configurable",
          HTOConfig().normalisation == "mean_centred_log1p",
          HTOConfig().normalisation)

    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.poisson(2, size=(300, 3)),
        rng.poisson(200, size=(300, 3)),
    ]).astype(float)
    X[:, 2] = 0.0                       # a hashtag that never fired

    A = clr_by_feature(X)
    B = clr_true_seurat(X)
    C = clr_by_cell(X)
    for nm, M in (("mean_centred_log1p", A), ("seurat_clr", B),
                  ("compositional", C)):
        check(f"{nm}: shape preserved and finite",
              M.shape == X.shape and np.isfinite(M).all())

    check("Seurat's transform maps zero counts to exactly 0",
          float(np.max(np.abs(B[X == 0]))) < 1e-12)
    check("mean-centred log1p maps zero counts BELOW zero (the real difference)",
          float(np.max(A[:, 0][X[:, 0] == 0], initial=-1.0)) <= 0.0)
    check("an all-zero hashtag does not produce NaN in either",
          np.isfinite(A[:, 2]).all() and np.isfinite(B[:, 2]).all())

    # Both are monotone in the raw count for a fixed feature, which is why the
    # classification structure is unaffected and only the cutoff moves.
    col = 0
    o = np.argsort(X[:, col])
    for nm, M in (("mean_centred_log1p", A), ("seurat_clr", B)):
        d = np.diff(M[o, col])
        check(f"{nm} is monotone in the raw count",
              bool((d >= -1e-9).all()))


def test_hto_transform_comparison() -> None:
    print("\n[A7: the transform comparison is a number, not an argument]")
    from perturbseq_report.config import HTOConfig
    from perturbseq_report.hto import compare_normalisations

    rng = np.random.default_rng(1)
    X = np.vstack([
        rng.poisson(1, size=(400, 2)),
        rng.poisson(150, size=(400, 2)),
    ]).astype(float)
    tab = compare_normalisations(X, ["hto_A", "hto_B"], HTOConfig())
    check("a row per hashtag per transform",
          len(tab) == 6, f"{len(tab)} rows")
    for c in ("normalisation", "hashtag", "threshold", "pct_positive"):
        check(f"column {c!r} present", c in tab.columns)

    # The claim being tested is specific: the two PER-FEATURE transforms differ
    # only in where the cut-off lands, so their call rates should agree
    # closely. The compositional one is a different question entirely and is
    # expected to disagree -- which is the whole reason this table exists.
    p = tab.pivot(index="hashtag", columns="normalisation",
                  values="pct_positive")
    gap = float((p["mean_centred_log1p"] - p["seurat_clr"]).abs().max())
    check("the two per-feature transforms give near-identical call rates",
          gap < 2.0, f"largest gap {gap:.2f} pp")

    # And on a hashtag that never fired, per-feature says 0% while the
    # compositional CLR calls half the cells positive -- a concrete reason not
    # to use it for demultiplexing, found by this diagnostic rather than argued.
    dead = np.zeros((600, 2))
    dead[:, 0] = np.r_[rng.poisson(1, 300), rng.poisson(150, 300)]
    t2 = compare_normalisations(dead, ["live", "dead"], HTOConfig())
    d = t2[t2["hashtag"] == "dead"].set_index("normalisation")["pct_positive"]
    check("per-feature transforms call 0% positive on a dead hashtag",
          float(d["mean_centred_log1p"]) == 0.0
          and float(d["seurat_clr"]) == 0.0,
          str(d.to_dict()))
    check("the compositional CLR does not, and the table shows it",
          float(d["compositional"]) > 10.0, str(d.to_dict()))


# ===========================================================================
# Part B
# ===========================================================================
def test_depth_matched_controls() -> None:
    print("\n[B1: fallback controls are depth-matched, not just substituted]")
    from perturbseq_report.controls import depth_matched_indices

    rng = np.random.default_rng(0)
    # The real situation: unassigned cells run at about half the depth.
    target_depth = rng.lognormal(np.log(5446), 0.4, size=2000)
    pool_depth = rng.lognormal(np.log(2674), 0.5, size=8000)

    idx, info = depth_matched_indices(target_depth, pool_depth, random_state=0)
    check("some cells were selected", idx.size > 0, str(idx.size))
    check("selection is a subset of the pool",
          idx.max() < pool_depth.size and idx.min() >= 0)
    check("no cell is used twice", len(set(idx.tolist())) == idx.size)

    before = abs(float(np.median(pool_depth)) - float(np.median(target_depth)))
    after = abs(float(np.median(pool_depth[idx])) - float(np.median(target_depth)))
    check("matching moves the median much closer",
          after < before * 0.4, f"before {before:.0f}, after {after:.0f}")
    check("the achieved match is reported",
          "median_ratio_after" in info and "n_selected" in info, str(info))

    # Degenerate inputs must not raise.
    e = np.array([])
    check("empty pool returns empty, does not raise",
          depth_matched_indices(target_depth, e, random_state=0)[0].size == 0)
    check("empty target returns empty",
          depth_matched_indices(e, pool_depth, random_state=0)[0].size == 0)

    # A pool that cannot match must say so rather than silently returning junk.
    _i, info2 = depth_matched_indices(
        target_depth, rng.lognormal(np.log(50), 0.2, size=500), random_state=0
    )
    check("an unmatchable pool is flagged", bool(info2.get("poor_match")),
          str(info2))


def _kd_fixture(seed: int = 0):
    """Two families, A with controls and B without, plus unassigned cells."""
    rng = np.random.default_rng(seed)
    vn = [f"G{i}" for i in range(20)]
    targets = pd.Series(
        ["G1_A"] * 200 + ["NTC_A"] * 200 + ["G2_B"] * 200 + [np.nan] * 200
    )
    mapping = pd.DataFrame({
        "target_key": ["G1_A", "NTC_A", "G2_B"],
        "target_gene": ["G1", "NTC", "G2"],
        "family": ["A", "A", "B"],
        "is_ntc": [False, True, False],
    })
    X = rng.lognormal(0.7, 0.3, size=(800, 20))
    X[600:, :] *= 0.5          # unassigned cells run shallower, as in reality
    return X, vn, targets, mapping


def test_fallback_is_only_used_when_needed() -> None:
    """The depth-matching step must not touch a normal dataset."""
    print("\n[B1: fallback fires only where a family has no controls]")
    from perturbseq_report.config import PipelineConfig
    from perturbseq_report.perturb import knockdown_table, target_annotations

    cfg = PipelineConfig().perturb
    rng = np.random.default_rng(0)
    vn = [f"G{i}" for i in range(20)]

    # --- a normal dataset: the one family present HAS controls -----------
    t = pd.Series(["G1_A"] * 300 + ["NTC_A"] * 300 + [np.nan] * 200)
    mp = pd.DataFrame({
        "target_key": ["G1_A", "NTC_A"], "target_gene": ["G1", "NTC"],
        "family": ["A", "A"], "is_ntc": [False, True],
    })
    X = rng.lognormal(0.7, 0.3, size=(800, 20))
    X[:300, 1] *= 0.3
    X[600:, :] *= 0.5
    d = X.sum(axis=1)
    un = t.isna().to_numpy()

    fb: dict = {}
    with_fb, _ = knockdown_table(X, vn, t, "NTC", cfg, mp, fallback_pool=un,
                                 depth=d, fallback_out=fb)
    without_fb, _ = knockdown_table(X, vn, t, "NTC", cfg, mp)

    check("fallback is not invoked when controls exist", fb == {}, str(fb))
    check("no row is marked as resting on a fallback pool",
          set(with_fb["control_is_fallback"]) == {False})
    check("the real controls are used, not a matched subset",
          int(with_fb["n_control_cells"].iloc[0]) == 300,
          str(int(with_fb["n_control_cells"].iloc[0])))

    # The strong form: supplying the machinery changes nothing at all.
    shared = [c for c in without_fb.columns if c in with_fb.columns]
    identical = all(
        np.allclose(with_fb[c].astype(float), without_fb[c].astype(float),
                    equal_nan=True)
        if pd.api.types.is_numeric_dtype(without_fb[c])
        else (with_fb[c].astype(str) == without_fb[c].astype(str)).all()
        for c in shared
    )
    check("results are byte-identical to not passing the fallback at all",
          identical)

    # --- a mixed dataset: family B has no controls -----------------------
    X2, vn2, t2, mp2 = _kd_fixture()
    d2 = X2.sum(axis=1)
    un2 = t2.isna().to_numpy()
    fb2: dict = {}
    kd2, _ = knockdown_table(X2, vn2, t2, "NTC", cfg, mp2, fallback_pool=un2,
                             depth=d2, fallback_out=fb2)
    check("fallback fires for the family that needs it", list(fb2) == ["B"],
          str(list(fb2)))
    by_key = kd2.set_index("target_key")
    check("only that family's rows are marked",
          bool(by_key.loc["G2_B", "control_is_fallback"])
          and not bool(by_key.loc["G1_A", "control_is_fallback"]))
    check("the family WITH controls keeps its real ones",
          int(by_key.loc["G1_A", "n_control_cells"]) == 200)

    # --- and without the machinery, B is excluded, never borrowed -------
    kd3, exc3 = knockdown_table(X2, vn2, t2, "NTC", cfg, mp2)
    check("with no fallback supplied, the family is EXCLUDED not borrowed",
          list(kd3["target_key"]) == ["G1_A"], str(list(kd3["target_key"])))
    check("and the exclusion says why",
          "cannot borrow another family's NTCs" in " ".join(exc3["reason"]))


def test_no_silent_cross_family_borrowing() -> None:
    """A single NTC key must not mean 'every family may use it'."""
    print("\n[B1: cross-family borrowing requires the explicit flag]")
    from perturbseq_report.perturb import target_annotations

    mapping = pd.DataFrame({
        "target_key": ["G1_A", "NTC_A", "G2_B"],
        "target_gene": ["G1", "NTC", "G2"],
        "family": ["A", "A", "B"],
        "is_ntc": [False, True, False],
    })

    # Up to v1.3.0 this was inferred from len(ntc_keys) == 1, which is also
    # true when pooling was NOT requested and only one family has controls --
    # exactly the mixed experiment the fallback exists for. Family B silently
    # borrowed family A's NTCs, so the fallback could never fire.
    ann = target_annotations(mapping, "NTC", pool_across_families=False)
    check("family B gets no control pool by default",
          ann.ntc_key_by_family.get("B") is None,
          str(ann.ntc_key_by_family))
    check("family A keeps its own", ann.ntc_key_by_family.get("A") == "NTC_A")

    pooled = target_annotations(mapping, "NTC", pool_across_families=True)
    check("with pooling explicitly on, B may use A's controls",
          pooled.ntc_key_by_family.get("B") == "NTC_A",
          str(pooled.ntc_key_by_family))

    s = src("perturb.py")
    check("the flag is threaded, not inferred",
          "if pool_across_families and len(ntc_keys) == 1:" in s)
    check("the stage passes the real config value",
          "cfg.guide.pool_ntc_across_families" in s)


def test_control_pool_warnings() -> None:
    print("\n[B2/B4: control-pool composition is reported]")
    from perturbseq_report.controls import summarise_control_pool

    guides = pd.Series(
        ["NTC_1"] * 500 + ["NTC_2"] * 400 + ["NTC_3"] * 300
    )
    s = summarise_control_pool({"A": guides, "C": pd.Series(["NTC_9"] * 974)})
    check("one row per family", len(s) == 2, str(len(s)))
    row_c = s.set_index("family").loc["C"]
    check("a single-guide pool is flagged", bool(row_c["single_guide"]))
    row_a = s.set_index("family").loc["A"]
    check("a multi-guide pool is not flagged", not bool(row_a["single_guide"]))
    check("guide counts are recorded", int(row_a["n_guides"]) == 3)
    check("the dominant guide's share is recorded",
          abs(float(row_a["top_guide_frac"]) - 500 / 1200) < 1e-6,
          str(row_a["top_guide_frac"]))

    # The softer version of the same problem: three guides, but one supplies
    # almost all the cells, so it behaves like a single-guide pool.
    conc = summarise_control_pool({
        "D": pd.Series(["NTC_1"] * 900 + ["NTC_2"] * 60 + ["NTC_3"] * 40),
    })
    check("a concentrated pool is flagged separately",
          bool(conc.iloc[0]["concentrated"])
          and not bool(conc.iloc[0]["single_guide"]),
          str(conc.iloc[0].to_dict()))


def test_leave_one_out_consistency() -> None:
    print("\n[B4: leave-one-out control consistency]")
    from perturbseq_report.controls import leave_one_out_consistency

    rng = np.random.default_rng(0)
    # Three well-behaved control guides, plus one that is clearly different.
    X = np.vstack([
        rng.normal(0.0, 1.0, size=(200, 40)),
        rng.normal(0.0, 1.0, size=(200, 40)),
        rng.normal(0.0, 1.0, size=(200, 40)),
        rng.normal(3.0, 1.0, size=(200, 40)),
    ])
    g = np.array(["g1"] * 200 + ["g2"] * 200 + ["g3"] * 200 + ["odd"] * 200)
    tab = leave_one_out_consistency(X, pd.Series(g))
    check("one row per control guide", len(tab) == 4, str(len(tab)))
    worst = tab.sort_values("max_abs_shift", ascending=False).iloc[0]
    check("the outlier guide is identified", worst["guide"] == "odd",
          str(worst.to_dict()))
    check("well-behaved guides shift the pool much less",
          float(tab.set_index("guide").loc["g1", "max_abs_shift"])
          < float(worst["max_abs_shift"]) / 2.0)


def test_pseudobulk_comparability() -> None:
    print("\n[B3: pseudobulk GEX and gRNA comparability]")
    from perturbseq_report.pseudobulk import (
        gRNA_composition, pseudobulk_by_group,
    )

    rng = np.random.default_rng(0)
    n_genes = 200
    # A shared expression profile, so genes actually rank consistently -- an
    # i.i.d. matrix has no structure to correlate and every rho would sit at 0
    # regardless of whether the code works.
    profile = rng.lognormal(0.0, 1.5, size=n_genes)
    X = np.vstack([
        rng.poisson(profile, size=(200, n_genes)),
        rng.poisson(profile, size=(200, n_genes)),
        # The 'odd' condition gets its profile SHUFFLED, i.e. genuinely
        # different biology rather than a rescaling (which Spearman would not
        # see at all, since it preserves ranks).
        rng.poisson(rng.permutation(profile), size=(200, n_genes)),
    ]).astype(float)
    grp = pd.Series(["CSU"] * 200 + ["IVT"] * 200 + ["odd"] * 200)

    pb = pseudobulk_by_group(X, [f"G{i}" for i in range(n_genes)], grp)
    check("one column per group", list(pb.columns) == ["CSU", "IVT", "odd"],
          str(list(pb.columns)))
    check("one row per gene", len(pb) == n_genes, str(len(pb)))

    from perturbseq_report.pseudobulk import correlation_matrix
    cm = correlation_matrix(pb)
    check("correlation matrix is square and symmetric",
          cm.shape == (3, 3)
          and np.allclose(cm.to_numpy(), cm.to_numpy().T, atol=1e-9))
    check("the odd condition is the least similar",
          float(cm.loc["CSU", "IVT"]) > float(cm.loc["CSU", "odd"]),
          str(cm.to_dict()))

    from perturbseq_report.pseudobulk import comparability_notes, composition_drift
    notes = comparability_notes(cm, pd.DataFrame(), "fixation")
    check("a weak correlation is called out in words",
          any("NOT well correlated" in n for n in notes), str(notes))

    guides = pd.Series(rng.choice(["a", "b", "c"], size=600))
    comp = gRNA_composition(guides, grp)
    check("gRNA composition is a guide x group table",
          comp.shape[1] == 3 and comp.shape[0] == 3, str(comp.shape))
    check("each group's composition sums to ~100%",
          bool(np.allclose(comp.sum(axis=0).to_numpy(), 100.0, atol=1e-6)))

    # Differential dropout: one guide is selectively lost from one condition.
    biased = pd.Series(
        list(rng.choice(["a", "b", "c"], size=400))
        + list(rng.choice(["a", "b"], size=200))
    )
    drift = composition_drift(gRNA_composition(biased, grp))
    check("a guide lost from one condition tops the drift table",
          str(drift.iloc[0]["guide"]) == "c", str(drift.head(3).to_dict()))
    dn = comparability_notes(pd.DataFrame(), drift, "fixation")
    check("differential dropout is explained, not just tabulated",
          any("non-randomly" in n for n in dn), str(dn))


def test_per_condition_perturbation() -> None:
    print("\n[B5: per-condition perturbation comparison]")
    s = src("perturb.py")
    check("a per-condition entry point exists",
          "def per_condition_knockdown" in s)
    check("the stage registers a per-condition table",
          "per_condition_" in s and "condition_effect_spread" in s)
    check("the pipeline hands the condition axes to the stage",
          "group_columns=group_f" in src("pipeline.py"))

    from perturbseq_report.config import PipelineConfig
    from perturbseq_report.perturb import (
        condition_effect_spread, condition_spread_notes, knockdown_table,
        per_condition_knockdown,
    )

    rng = np.random.default_rng(0)
    n_genes = 20
    n = 800
    var_names = [f"G{i}" for i in range(n_genes)]
    X = rng.lognormal(0.7, 0.3, size=(n, n_genes))

    # Controls are family-scoped, so a realistic mapping is required: bare
    # target strings with no mapping give ntc_key_by_family == {None: 'NTC'}
    # and nothing can be quantified.
    targets = pd.Series(["G1_A"] * 400 + ["NTC_A"] * 400)
    mapping = pd.DataFrame({
        "target_key": ["G1_A", "NTC_A"],
        "target_gene": ["G1", "NTC"],
        "family": ["A", "A"],
        "is_ntc": [False, True],
    })
    cond = pd.Series(["CSU"] * 200 + ["IVT"] * 200 + ["CSU"] * 200 + ["IVT"] * 200)
    # G1 knocked down hard in CSU only. Pooled this averages to something
    # unremarkable, which is exactly the failure this panel exists to catch.
    X[:200, 1] *= 0.15

    pooled, _ = knockdown_table(
        X, var_names, targets, "NTC", PipelineConfig().perturb, mapping,
    )
    check("the pooled table still works", not pooled.empty)

    tab = per_condition_knockdown(
        X, var_names, targets, cond, "NTC", PipelineConfig().perturb, mapping,
    )
    check("a row per target per condition level", len(tab) == 2, str(len(tab)))
    check("the condition is named in the output", "condition" in tab.columns)
    if "condition" in tab.columns and {"CSU", "IVT"} <= set(tab["condition"]):
        by = tab.set_index("condition")["pct_knockdown"].astype(float)
        check("a condition-specific effect is not averaged away",
              float(by["CSU"]) > float(by["IVT"]) + 20.0,
              f"CSU {by['CSU']:.1f}% vs IVT {by['IVT']:.1f}%")
        pooled_kd = float(pooled["pct_knockdown"].iloc[0])
        check("and pooling really does hide it",
              float(by["IVT"]) < pooled_kd < float(by["CSU"]),
              f"IVT {by['IVT']:.1f} < pooled {pooled_kd:.1f} < CSU {by['CSU']:.1f}")

        spread = condition_effect_spread(tab)
        check("the spread table reports the gap", len(spread) == 1, str(len(spread)))
        check("range_pp is the best-worst gap",
              abs(float(spread.iloc[0]["range_pp"])
                  - abs(float(by["CSU"]) - float(by["IVT"]))) < 1e-6)
        check("the best arm is identified",
              str(spread.iloc[0]["best_condition"]) == "CSU",
              str(spread.iloc[0].to_dict()))
        notes = condition_spread_notes(spread, "fixation")
        check("the finding is explained in words",
              any("describes neither" in x for x in notes), str(notes))

    # A single-level axis must produce nothing rather than a spurious row.
    one = per_condition_knockdown(
        X, var_names, targets, pd.Series(["only"] * n), "NTC",
        PipelineConfig().perturb, mapping,
    )
    check("a one-level condition yields one row, not a comparison",
          len(one) <= 1, str(len(one)))


# ===========================================================================
# B8 -- dead config
# ===========================================================================
def test_no_dead_config() -> None:
    print("\n[B8: every config field is read by something]")
    from dataclasses import fields

    from perturbseq_report import config as C

    all_src = "\n".join(
        p.read_text() for p in sorted(ROOT.glob("*.py"))
    )
    # Fields that are legitimately declarative: consumed by name-matching
    # loops, serialised into the report, or provenance only.
    allow = {
        "source", "title", "subtitle",
    }
    dead: list[str] = []
    for cls in (C.QCThresholds, C.ModalityConfig, C.GuideConfig,
                C.PerturbConfig, C.HTOConfig, C.EmbeddingConfig,
                C.FigureConfig, C.ReportConfig, C.PipelineConfig):
        for f in fields(cls):
            if f.name in allow:
                continue
            # Count references outside the declaration itself.
            refs = all_src.count(f".{f.name}") + all_src.count(f'"{f.name}"')
            if refs == 0:
                dead.append(f"{cls.__name__}.{f.name}")
    check("no config field is wired to nothing",
          not dead, "unreferenced: " + ", ".join(dead))

    check("n_jobs is read", "sc.settings.n_jobs" in all_src)
    check("use_checkpoints is read", "cfg.use_checkpoints" in all_src)


def main() -> int:
    for fn in (
        test_batch_key_dominance,
        test_harmony_is_opt_in,
        test_hvg_is_not_batch_aware_by_default,
        test_edistance_space_is_stated,
        test_regress_out_default_empty,
        test_scale_clip_documented,
        test_no_cluster_merging,
        test_performance_wiring,
        test_doublet_gate,
        test_hto_normalisation,
        test_hto_transform_comparison,
        test_depth_matched_controls,
        test_fallback_is_only_used_when_needed,
        test_no_silent_cross_family_borrowing,
        test_control_pool_warnings,
        test_leave_one_out_consistency,
        test_pseudobulk_comparability,
        test_per_condition_perturbation,
        test_no_dead_config,
    ):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
