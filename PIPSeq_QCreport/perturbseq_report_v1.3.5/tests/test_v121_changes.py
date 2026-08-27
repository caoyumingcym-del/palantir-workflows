"""
Regression tests for v1.2.1.

    python tests/test_v121_changes.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturbseq_report.config import GuideConfig, PerturbConfig
from perturbseq_report.guide import GuideParser
from perturbseq_report.manifest import ManifestError, read_table
from perturbseq_report.sanity import check_expression_matrix
from perturbseq_report.stats import (
    differential_expression, mannwhitney_u, mannwhitney_u_columns,
    mannwhitney_u_sparse_columns, rank_columns, rankdata,
)
from perturbseq_report.whitelists import load_guide_whitelist

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"  {detail}"))
    if not cond:
        FAILURES.append(name)


def write(text: str) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    f.write(text)
    f.close()
    return Path(f.name)


class MiniCSC:
    """Minimal CSC so the sparse DE path is exercised without scipy.

    ``toarray`` raises: if the sparse path ever densifies, the test fails
    rather than silently losing the optimisation.
    """

    format = "csc"

    def __init__(self, M=None, data=None, indptr=None, shape=None):
        if M is not None:
            M = np.asarray(M, float)
            n, k = M.shape
            d, ip = [], [0]
            for j in range(k):
                col = M[:, j]
                nz = col[col != 0]
                d.append(nz)
                ip.append(ip[-1] + nz.size)
            self.data = np.concatenate(d) if d else np.zeros(0)
            self.indptr = np.array(ip)
            self.shape = (n, k)
        else:
            self.data, self.indptr, self.shape = data, indptr, shape

    def toarray(self):
        raise AssertionError("sparse DE path densified the matrix")

    def tocsr(self):
        return self

    def tocsc(self):
        return self

    def __getitem__(self, key):
        _rows, cols = key
        a, b = cols.start or 0, cols.stop
        s, e = self.indptr[a], self.indptr[b]
        return MiniCSC(data=self.data[s:e],
                       indptr=self.indptr[a:b + 1] - self.indptr[a],
                       shape=(self.shape[0], b - a))


# ===========================================================================
def test_rank_columns_matches_scalar() -> None:
    print("\n[vectorised ranking]")
    rng = np.random.default_rng(0)
    worst_rank, tie_ok = 0.0, True
    for trial in range(8):
        n, k = int(rng.integers(20, 400)), int(rng.integers(2, 25))
        A = rng.poisson(rng.uniform(0.05, 3), size=(n, k)).astype(float)
        if trial % 2:
            A = np.log1p(A)             # heavy ties at zero, like real data
        R, tie = rank_columns(A)
        for j in range(k):
            worst_rank = max(worst_rank, np.abs(R[:, j] - rankdata(A[:, j])).max())
            _v, c = np.unique(A[:, j], return_counts=True)
            tie_ok &= abs(tie[j] - float((c ** 3 - c).sum())) < 1e-6
    check("column ranks identical to scalar rankdata", worst_rank == 0.0,
          f"max diff {worst_rank:.3e}")
    check("tie terms exact", tie_ok)


def test_mannwhitney_equivalence() -> None:
    """The whole DE speedup is gated on this: same statistic, same p-value."""
    print("\n[Mann-Whitney equivalence]")
    rng = np.random.default_rng(1)
    wu = wp = 0.0
    n_cmp = 0
    for _ in range(6):
        n1, n2, k = (int(rng.integers(30, 300)), int(rng.integers(30, 600)),
                     int(rng.integers(3, 20)))
        Bg = np.log1p(rng.poisson(rng.uniform(0.05, 2), size=(n1, k)).astype(float))
        Br = np.log1p(rng.poisson(rng.uniform(0.05, 2), size=(n2, k)).astype(float))
        u, p = mannwhitney_u_columns(Bg, Br)
        us, ps = mannwhitney_u_sparse_columns(MiniCSC(Bg), MiniCSC(Br), n1, n2)
        for j in range(k):
            u0, p0 = mannwhitney_u(Bg[:, j], Br[:, j])
            if not (np.isfinite(p0) and np.isfinite(p[j])):
                continue
            wu = max(wu, abs(u[j] - u0), abs(us[j] - u0))
            wp = max(wp, abs(p[j] - p0), abs(ps[j] - p0))
            n_cmp += 1
    check(f"dense and sparse match the per-gene test on {n_cmp} genes",
          wu == 0.0 and wp == 0.0, f"max dU {wu:.2e} max dp {wp:.2e}")


def test_de_sparse_equals_dense() -> None:
    print("\n[differential_expression: sparse == dense]")
    rng = np.random.default_rng(3)
    n1, n2, k = 200, 700, 60
    Bg = np.log1p((rng.random((n1, k)) < 0.06) * rng.poisson(4, size=(n1, k))).astype(float)
    Br = np.log1p((rng.random((n2, k)) < 0.06) * rng.poisson(4, size=(n2, k))).astype(float)
    genes = [f"G{i}" for i in range(k)]
    dense = differential_expression(Bg, Br, genes, log_input=True).table
    sparse = differential_expression(MiniCSC(Bg), MiniCSC(Br), genes,
                                     log_input=True, block=17).table
    ok = True
    for col in ("log2fc", "pvalue", "padj", "mean_group", "mean_ref",
                "frac_detected_group", "frac_detected_ref"):
        d = float(np.nanmax(np.abs(dense[col].to_numpy() - sparse[col].to_numpy())))
        ok &= d < 1e-12
    check("every DE column agrees to 1e-12", ok)
    check("low_expression flag agrees",
          bool((dense.low_expression == sparse.low_expression).all()))


def test_de_is_actually_faster() -> None:
    """A speedup that isn't one is not worth the risk of new code.

    The first attempt -- a dense column-wise ranking -- measured 1.0x, because
    argsort over the full block costs what the per-gene loop cost. Exploiting
    the zeros is what makes it worthwhile, so the margin is asserted.
    """
    print("\n[DE speedup is real]")
    rng = np.random.default_rng(5)
    n1, n2, k = 800, 12000, 300
    Bg = np.log1p((rng.random((n1, k)) < 0.04) * rng.poisson(3, size=(n1, k))).astype(float)
    Br = np.log1p((rng.random((n2, k)) < 0.04) * rng.poisson(3, size=(n2, k))).astype(float)
    t = time.time()
    mannwhitney_u_sparse_columns(MiniCSC(Bg), MiniCSC(Br), n1, n2)
    t_sparse = time.time() - t
    n_probe = 20
    t = time.time()
    for j in range(n_probe):
        mannwhitney_u(Bg[:, j], Br[:, j])
    t_loop = (time.time() - t) / n_probe * k
    speedup = t_loop / max(t_sparse, 1e-9)
    print(f"        per-gene {t_loop:.2f}s  vs  sparse {t_sparse:.2f}s")
    check(f"sparse path is >=5x faster (measured {speedup:.1f}x)", speedup >= 5.0)


def test_single_family_labels() -> None:
    print("\n[single-family labels]")
    ids = [
        "ABT1_target_version_1.0_spacer_number_2_spacer_target_ENSG00000146109_TCCATGTTGACTGACACGAG",
        "ABT1_target_version_1.1_spacer_number_2_spacer_target_ONE_INTERGENIC_SITE_GATGTTACTCACAACCAACC",
        "CDH1_1_TGAACCACCAGGGTATACGT",
        "CD55_singleguide",
        "NTC_10_ACGTTGACCATGCTAAGGCA",
    ]
    m = GuideParser(GuideConfig()).parse_all(ids).set_index("guide_id")
    check("no whitelist -> no 'unassigned' in labels",
          m.loc[ids[0], "short_label"] == "ABT1_v1.0s2",
          m.loc[ids[0], "short_label"])
    check("no whitelist -> target_key is the bare gene",
          set(m["target_key"]) == {"ABT1", "NTC", "CDH1", "CD55"},
          str(sorted(set(m["target_key"]))))

    wl1 = load_guide_whitelist(write(
        "guide_id,family\n" + "".join(f"{g},A\n" for g in ids)))
    m1 = GuideParser(GuideConfig(), wl1).parse_all(ids).set_index("guide_id")
    check("one declared family -> suffix still suppressed",
          m1.loc[ids[2], "short_label"] == "CDH1_g1"
          and m1.loc[ids[2], "target_key"] == "CDH1",
          f"{m1.loc[ids[2], 'short_label']} / {m1.loc[ids[2], 'target_key']}")

    wl2 = load_guide_whitelist(write(
        "guide_id,family\n" + "".join(
            f"{g},{'A' if i < 3 else 'B'}\n" for i, g in enumerate(ids))))
    m2 = GuideParser(GuideConfig(), wl2).parse_all(ids).set_index("guide_id")
    check("two families -> suffix returns",
          m2.loc[ids[2], "short_label"] == "CDH1_A_g1"
          and m2.loc[ids[2], "target_key"] == "CDH1_A",
          f"{m2.loc[ids[2], 'short_label']} / {m2.loc[ids[2], 'target_key']}")
    check("two families -> control pools stay separate",
          {"NTC_A", "NTC_B"} <= set(m2["target_key"]))


def test_input_sanity_gate() -> None:
    print("\n[input sanity gate]")
    rng = np.random.default_rng(0)
    hk = ["ACTB", "GAPDH", "B2M", "EEF1A1", "RPL13A", "RPLP0", "PPIA", "TPT1",
          "RPS18", "RPL10", "UBC", "HSP90AB1", "PGK1", "TUBB", "MALAT1"]
    names = hk + [f"GENE{i}" for i in range(400)]
    n, k = 2500, len(names)
    good = np.zeros((n, k))
    for j in range(len(hk)):
        good[:, j] = rng.poisson(20, n)
    for j in range(len(hk), k):
        good[:, j] = rng.poisson(0.05, n)
    var = pd.DataFrame({"n_cells_by_counts": (good > 0).sum(0)}, index=names)

    r = check_expression_matrix(good, names, var)
    check("a real matrix passes", r.ok, str(r.failures[:1]))
    check("var statistics agree", r.var_stat_agreement is not None
          and 0.9 < r.var_stat_agreement < 1.1)

    perm = rng.permutation(k)
    r2 = check_expression_matrix(good[:, perm], names, var)
    check("permuted columns are caught", not r2.ok)
    check("failure names the likely cause",
          any("permuted" in f for f in r2.failures))

    shallow = np.zeros((n, k))
    for j in range(len(hk)):
        shallow[:, j] = (rng.random(n) < 0.55) * rng.poisson(3, n)
    r3 = check_expression_matrix(shallow, names, None)
    check("a shallow but valid matrix is not rejected", r3.ok,
          f"median detection {r3.median_detection:.3f}")

    r4 = check_expression_matrix(good, [f"g{i}" for i in range(k)], None)
    check("unknown reference: cannot check, does not fail", r4.ok)
    check("and says why", any("could not be checked" in n_ for n_ in r4.notes))


def test_condition_columns_manifest() -> None:
    print("\n[condition_columns in the manifest]")
    from perturbseq_report.manifest import read_manifest

    base = ("sample,prefix,h5ad_path,output_path,fixation,buffer,"
            "condition_columns\n")
    rows = ("s1,p1,a.h5ad,out,fresh,CSB,fixation|buffer\n"
            "s2,p1,a.h5ad,out,DSP-ME,SSC,fixation|buffer\n")
    m = read_manifest(write(base + rows), strict=False)
    check("declared columns parsed in order",
          m.declared_condition_columns() == ["fixation", "buffer"],
          str(m.declared_condition_columns()))
    check("condition_columns is not itself metadata",
          "condition_columns" not in m.metadata_columns(),
          str(m.metadata_columns()))

    bad = read_manifest(write(base + rows.replace("fixation|buffer", "typo_col")),
                        strict=False)
    try:
        bad.declared_condition_columns()
    except ManifestError as exc:
        check("a typo raises rather than silently changing comparisons",
              "typo_col" in str(exc))
    else:
        check("a typo raises", False, "no error")

    absent = read_manifest(write(
        "sample,prefix,h5ad_path,output_path,fixation\n"
        "s1,p1,a.h5ad,out,fresh\ns2,p1,a.h5ad,out,DSP-ME\n"), strict=False)
    check("absent column -> autodetection still applies",
          absent.declared_condition_columns() == [])


def test_condition_alias_and_priority() -> None:
    """`condition = fixation` must be honoured, and must not exclude the rest.

    v1.2.1 shipped looking only for `condition_columns`, so a manifest saying
    `condition,fixation` was ignored; autodetection then found four qualifying
    columns, the cap kept three, and the tie-break dropped `fixation` -- the
    one thing the author had named.
    """
    print("\n[condition alias and priority]")
    from perturbseq_report.config import PipelineConfig
    from perturbseq_report.manifest import read_manifest
    from perturbseq_report.pipeline import resolve_group_columns

    hdr = ("sample,h5ad_path,output_path,gRNA_method,acoh,"
           "resuspension_buffer,fixation,condition\n")
    rows = "".join(
        f"s{i},a.h5ad,out,{'CSU' if i % 2 else 'IVT'},"
        f"{'AcOH' if i < 4 else 'no AcOH'},{'SSC' if i < 2 else 'CSB'},"
        f"{'DES' if i < 6 else 'fresh'},fixation\n"
        for i in range(8)
    )
    m = read_manifest(write(hdr + rows), strict=False)
    check("alias 'condition' recognised", m.nominating_column() == "condition",
          str(m.nominating_column()))
    check("nominates fixation", m.declared_condition_columns() == ["fixation"])

    meta = m.sample_metadata_frame()
    obs = pd.DataFrame({"sample": [s for s in m.samples for _ in range(3)]})
    for c in meta.columns:
        obs[c] = obs["sample"].map(meta[c])
    axes = list(resolve_group_columns(obs, m, PipelineConfig()))
    check("nominated column comes first", axes[:1] == ["fixation"], str(axes))
    check("the others are kept, not discarded",
          {"gRNA_method", "acoh", "resuspension_buffer"} <= set(axes), str(axes))

    # A `condition` column holding real values must stay ordinary metadata.
    hdr2 = "sample,h5ad_path,output_path,fixation,condition\n"
    rows2 = ("s1,a.h5ad,out,DES,untreated\ns2,a.h5ad,out,fresh,TGFb\n")
    m2 = read_manifest(write(hdr2 + rows2), strict=False)
    check("a value-bearing 'condition' is not treated as nominating",
          m2.nominating_column() is None, str(m2.nominating_column()))
    check("and stays available as metadata",
          "condition" in m2.metadata_columns(), str(m2.metadata_columns()))


def test_retention_yield_uses_sample_grouping() -> None:
    """The end-to-end yield panel must group by sample, not by condition.

    cell_input is per sample; the retention summary is grouped by the first
    condition axis. Feeding the latter to the yield panel compared "CSU"/"IVT"
    against "MDL1898_1" and reported "no matching cell_input in manifest" on a
    manifest that plainly had one.
    """
    print("\n[retention yield grouping]")
    from perturbseq_report.config import FigureConfig, QCThresholds
    from perturbseq_report.qc import filter_cells, plot_retention
    import inspect as _inspect

    check("plot_retention accepts a separate yield_summary",
          "yield_summary" in _inspect.signature(plot_retention).parameters)

    rng = np.random.default_rng(0)
    n = 400
    qc = pd.DataFrame({
        "total_counts": rng.integers(500, 20000, n).astype(float),
        "n_genes_by_counts": rng.integers(200, 6000, n).astype(float),
        "pct_counts_mt": rng.uniform(0, 20, n),
    }, index=[f"c{i}" for i in range(n)])
    cond = pd.Series(np.where(np.arange(n) % 2 == 0, "CSU", "IVT"), index=qc.index)
    samp = pd.Series([f"MDL1898_{1 + (i % 4)}" for i in range(n)], index=qc.index)
    th = QCThresholds(min_genes=250, max_genes=5500, min_counts=600,
                      max_counts=19000, max_mito=15)
    fr_cond = filter_cells(qc, th, cond)
    fr_samp = filter_cells(qc, th, samp)
    cell_input = pd.Series({f"MDL1898_{i}": 20000 for i in range(1, 5)})

    cond_groups = set(fr_cond.summary["group"].astype(str))
    samp_groups = set(fr_samp.summary["group"].astype(str))
    check("condition groups do NOT match cell_input (the old bug)",
          not (cond_groups & set(cell_input.index)))
    check("sample groups DO match cell_input",
          samp_groups == set(cell_input.index))

    out = Path(tempfile.mktemp(suffix=".png"))
    plot_retention(fr_cond.per_reason, fr_cond.summary, FigureConfig(), out,
                   cell_input, yield_summary=fr_samp.summary)
    check("yield panel renders with sample grouping", out.exists())


def test_seq_metric_value_labels() -> None:
    """Rate metrics arrive as fractions; the panel formatted them as integers."""
    print("\n[sequencing-metric labels]")
    from perturbseq_report.pipeline import _scale_metric_for_plot

    vals, fmt, unit = _scale_metric_for_plot(
        "pct_reads_in_cells", np.array([0.607, 0.42, 0.731]))
    check("fractions are scaled to percent", bool(np.allclose(vals[0], 60.7)),
          str(vals[:1]))
    check("and formatted with a decimal", fmt(vals[0]) == "60.7", fmt(vals[0]))
    check("axis is labelled as a percentage", unit == " (%)", unit)

    vals2, fmt2, _ = _scale_metric_for_plot(
        "pct_mapped_reads", np.array([62.6, 60.4]))
    check("values already in percent are left alone",
          bool(np.allclose(vals2[0], 62.6)) and fmt2(vals2[0]) == "62.6")

    vals3, fmt3, unit3 = _scale_metric_for_plot(
        "estimated_cells", np.array([326121.0, 317963.0]))
    check("counts keep a thousands separator",
          fmt3(vals3[0]) == "326,121" and unit3 == "", fmt3(vals3[0]))


def test_dotplot_caps_exist() -> None:
    print("\n[DEG dot-plot caps]")
    c = PerturbConfig()
    check("gene cap configured", c.dotplot_max_genes == 120)
    check("target cap configured", c.dotplot_max_targets == 30)




def test_doublets_off_by_default() -> None:
    """Doublet detection is opt-in from v1.2.2, and runs on raw counts.

    It was never new code -- any input carrying obsm['X_pca'] takes the reuse
    path, which never calls scrublet. MDL1898 was the first object without a
    precomputed embedding, so it was the first time scrublet had ever executed,
    and it OOM'd the run.
    """
    print("\n[doublet detection]")
    import inspect
    from perturbseq_report import cli, gex
    from perturbseq_report.config import EmbeddingConfig

    check("off by default", EmbeddingConfig().detect_doublets is False)

    p = cli.build_parser()
    check("--doublets opts in",
          p.parse_args(["--manifest", "x.csv", "--doublets"]).doublets is True)
    check("--no-doublets still accepted (existing commands keep working)",
          p.parse_args(["--manifest", "x.csv", "--no-doublets"]).no_doublets is True)

    src = inspect.getsource(gex)
    i_dbl = src.find("sc.pp.scrublet")
    i_norm = src.find('_s("normalize_total")')
    check("scrublet runs BEFORE normalisation, i.e. on raw counts",
          0 < i_dbl < i_norm, f"scrublet@{i_dbl} normalize@{i_norm}")
    check("doublet detection has a step label",
          "_s(f\"doublet detection" in src)
    check("cell-cycle scoring has a step label",
          '_s("cell-cycle scoring")' in src)



def test_de_handles_empty_gene_columns() -> None:
    """Undetected genes must not crash the sparse DE path.

    `np.add.reduceat(data, indptr[:-1])` raises when the block's last column is
    empty, because indptr[k-1] == len(data). It killed a real run at the
    perturbation stage after everything else had completed:
    "IndexError: index 191684 out-of-bounds in add.reduceat [0, 191684)".
    My original test used uniformly dense blocks, so it never fired -- even
    though 11,504 of that object's 38,402 genes are detected in no cell.
    """
    print("\n[DE with empty gene columns]")
    rng = np.random.default_rng(7)
    n1, n2, k = 150, 400, 40
    Bg = np.log1p((rng.random((n1, k)) < 0.08) * rng.poisson(4, size=(n1, k))).astype(float)
    Br = np.log1p((rng.random((n2, k)) < 0.08) * rng.poisson(4, size=(n2, k))).astype(float)
    genes = [f"G{i}" for i in range(k)]
    cases = {
        "last column empty (the crash)": [k - 1],
        "first column empty": [0],
        "trailing run empty": [k - 3, k - 2, k - 1],
        "scattered empty": [0, 5, 11, k - 1],
        "all columns empty": list(range(k)),
    }
    for name, zero_cols in cases.items():
        G, R = Bg.copy(), Br.copy()
        for j in zero_cols:
            G[:, j] = 0.0
            R[:, j] = 0.0
        try:
            dense = differential_expression(G, R, genes, log_input=True).table
            sparse = differential_expression(MiniCSC(G), MiniCSC(R), genes,
                                             log_input=True, block=7).table
            worst = max(
                float(np.nanmax(np.abs(dense[c].to_numpy() - sparse[c].to_numpy())))
                for c in ("log2fc", "mean_group", "mean_ref",
                          "frac_detected_group")
            )
            check(f"{name}", worst < 1e-12, f"max|diff| {worst:.2e}")
        except Exception as exc:
            check(f"{name}", False, f"{type(exc).__name__}: {exc}")



def test_embedding_cache() -> None:
    """The checkpoint has to be a real round-trip AND miss when it should."""
    print("\n[embedding cache]")
    import types

    from perturbseq_report import embedcache as EC
    from perturbseq_report.config import PipelineConfig

    cfg = PipelineConfig()
    cfg.output_path = Path(tempfile.mkdtemp())
    obs_names = [f"c{i}" for i in range(400)]
    var_names = [f"G{i}" for i in range(60)]

    key = EC.embedding_key(None, obs_names, cfg)
    check("key is deterministic", key == EC.embedding_key(None, obs_names, cfg))

    other = PipelineConfig()
    other.output_path = cfg.output_path
    other.embedding.leiden_resolution = cfg.embedding.leiden_resolution + 1.0
    check("key changes when resolution changes",
          key != EC.embedding_key(None, obs_names, other))
    check("key changes when the retained cells change",
          key != EC.embedding_key(None, obs_names[:-1], cfg))

    rng = np.random.default_rng(0)
    result = types.SimpleNamespace(
        pca=rng.normal(size=(400, 30)),
        umap=rng.normal(size=(400, 2)),
        obs=pd.DataFrame(
            # 'cluster', not 'leiden' -- the name the transcriptome stage
            # actually writes. Getting this wrong made every cache hit crash
            # the report on res.obs["cluster"].
            {"cluster": pd.Categorical(rng.integers(0, 5, 400).astype(str)),
             "phase": ["G1"] * 400,
             "S_score": rng.normal(size=400),
             "predicted_doublet": rng.random(400) < 0.05},
            index=obs_names,
        ),
        hvg=[f"G{i}" for i in range(30)],
        var_names=var_names,
        backend="scanpy",
        batch_corrected="harmony on 'sample'",
    )
    check("miss before anything is written",
          EC.load(cfg, key, obs_names, var_names)[0] is None)
    check("save returns a path", EC.save(result, cfg, key) is not None)

    payload, reason = EC.load(cfg, key, obs_names, var_names)
    check("hit after save", payload is not None, reason)
    if payload is not None:
        check("pca shape survives", payload["pca"].shape == (400, 30))
        check("umap shape survives", payload["umap"].shape == (400, 2))
        check("hvg list survives", payload["hvg"] == result.hvg)
        check("batch-correction description survives",
              payload["batch_corrected"] == "harmony on 'sample'")
        restored = EC.apply(payload, pd.DataFrame(index=obs_names))
        check("cluster labels survive",
              bool((restored["cluster"].astype(str).to_numpy()
                    == result.obs["cluster"].astype(str).to_numpy()).all()))
        check("cluster comes back categorical, as the live stage produces",
              isinstance(restored["cluster"].dtype, pd.CategoricalDtype))
        check("doublet calls come back as bool",
              restored["predicted_doublet"].dtype == bool)
        check("cell-cycle score survives numerically",
              float(np.nanmax(np.abs(restored["S_score"].to_numpy()
                                     - result.obs["S_score"].to_numpy()))) < 1e-6)
        check("the column the report indexes is cached",
              "cluster" in EC.CACHED_OBS_COLUMNS)

    # A checkpoint without clusters is unusable; it must not be written, and an
    # old one without them must miss rather than be loaded and then crash.
    bad = types.SimpleNamespace(
        pca=result.pca, umap=result.umap,
        obs=pd.DataFrame({"phase": ["G1"] * 400}, index=obs_names),
        hvg=result.hvg, var_names=var_names,
        backend="scanpy", batch_corrected="none",
    )
    check("refuses to write a checkpoint with no clusters",
          EC.save(bad, cfg, "clusterless") is None)

    # The failure mode that matters: a stale embedding returned for a
    # different cell or gene set. Both must be refused, not silently reused.
    check("refuses a different cell set",
          EC.load(cfg, key, obs_names[:-1], var_names)[0] is None)
    check("refuses a different gene set",
          EC.load(cfg, key, obs_names, var_names[:-1])[0] is None)

    # A corrupt file must miss, not raise, or a bad checkpoint bricks every run.
    npz, _meta = EC._paths(cfg, key)
    npz.write_bytes(b"not an npz")
    try:
        payload2, reason2 = EC.load(cfg, key, obs_names, var_names)
        check("a corrupt checkpoint misses instead of raising",
              payload2 is None, reason2)
    except Exception as exc:
        check("a corrupt checkpoint misses instead of raising", False,
              f"{type(exc).__name__}: {exc}")

    src = (Path(__file__).resolve().parents[1]
           / "perturbseq_report" / "gex.py").read_text()
    check("gex.py reads the cache", "EMBCACHE.load(" in src)
    check("gex.py writes the cache", "EMBCACHE.save(" in src)
    check("--force-recompute overrides the cache",
          "force_recompute" in src)


def test_purity_by_group() -> None:
    """Purity panels must exist per sample and per condition axis."""
    print("\n[guide purity per sample and per condition]")
    from perturbseq_report.config import FigureConfig
    from perturbseq_report.guide import plot_purity_by_group
    from perturbseq_report.stats import GuideStats

    src = (Path(__file__).resolve().parents[1]
           / "perturbseq_report" / "guide.py").read_text()
    check("'strict secondary gate' is gone", "strict secondary gate" not in src)
    check("'purity triangle' is the new name", "purity triangle" in src)
    check("run_guide_stage takes a sample axis", "sample: pd.Series | None" in src)

    pipe = (Path(__file__).resolve().parents[1]
            / "perturbseq_report" / "pipeline.py").read_text()
    check("pipeline passes sample labels to the guide stage",
          "sample=_sample_for_guides" in pipe)

    rng = np.random.default_rng(1)
    n = 600
    stats = GuideStats(
        total=rng.integers(10, 500, n).astype(float),
        top1=rng.integers(5, 400, n).astype(float),
        top2=rng.integers(0, 50, n).astype(float),
        top1_over_top2=rng.uniform(40.0, 100.0, n),   # percent
        top1_over_total=rng.uniform(0.3, 1.0, n),
        top12_over_total=rng.uniform(0.5, 1.0, n),
        n_detected=rng.integers(1, 5, n),
        top1_index=rng.integers(0, 10, n),
    )
    group = pd.Series(np.where(np.arange(n) % 2 == 0, "CSU", "IVT"))
    out = Path(tempfile.mkdtemp()) / "purity_by_condition.png"
    try:
        p = plot_purity_by_group(stats, group, "fixation",
                                 GuideConfig(), FigureConfig(), out)
        check("a panel is produced per level", p.exists() and p.stat().st_size > 0)
    except Exception as exc:
        check("a panel is produced per level", False,
              f"{type(exc).__name__}: {exc}")

    # One level must not crash, and neither must an empty level.
    try:
        p = plot_purity_by_group(stats, pd.Series(["only"] * n), "sample",
                                 GuideConfig(), FigureConfig(),
                                 out.with_name("one.png"))
        check("a single level still renders", p.exists())
    except Exception as exc:
        check("a single level still renders", False,
              f"{type(exc).__name__}: {exc}")


def test_regress_out_covariates_are_present() -> None:
    """The report claims depth and %mito are regressed out; they must be there."""
    print("\n[regress_out actually gets its covariates]")
    pipe = (Path(__file__).resolve().parents[1]
            / "perturbseq_report" / "pipeline.py").read_text()
    for col in ("total_counts", "pct_counts_mt"):
        check(f"{col} is attached to the embedding input obs",
              f'"{col}"' in pipe.split("gex_f = split.gex[keep]")[-1][:1200])
    # v1.3.0 inverts the default: nothing is regressed out unless asked. The
    # attach-to-obs fix above still matters, because it is what makes
    # --regress-qc actually work when someone does ask.
    from perturbseq_report.config import PipelineConfig
    cfg = PipelineConfig()
    check("nothing is regressed out by default",
          tuple(cfg.embedding.regress_out) == (),
          str(tuple(cfg.embedding.regress_out)))

    from perturbseq_report.cli import _collect_overrides, build_parser
    p = build_parser()
    o = _collect_overrides(p.parse_args(["--manifest", "m.csv", "--regress-qc"]))
    check("--regress-qc opts in to depth and %mito",
          tuple(o.get("regress_out", ())) == ("total_counts", "pct_counts_mt"),
          str(o.get("regress_out")))
    check("--regress-qc does NOT offer cell-cycle regression",
          "S_score" not in tuple(o.get("regress_out", ()))
          and "G2M_score" not in tuple(o.get("regress_out", ())))
    o2 = _collect_overrides(p.parse_args(["--manifest", "m.csv"]))
    check("no flag means no covariates",
          tuple(o2.get("regress_out", ())) == ())
    check("--no-regress-out still accepted (existing commands keep working)",
          tuple(_collect_overrides(
              p.parse_args(["--manifest", "m.csv", "--no-regress-out"])
          ).get("regress_out", ())) == ())
    check("cell-cycle scores are still COMPUTED, just not removed",
          cfg.embedding.score_cell_cycle is True)


def main() -> int:
    for fn in (
        test_rank_columns_matches_scalar,
        test_mannwhitney_equivalence,
        test_de_sparse_equals_dense,
        test_de_handles_empty_gene_columns,
        test_de_is_actually_faster,
        test_single_family_labels,
        test_input_sanity_gate,
        test_condition_columns_manifest,
        test_condition_alias_and_priority,
        test_retention_yield_uses_sample_grouping,
        test_seq_metric_value_labels,
        test_dotplot_caps_exist,
        test_doublets_off_by_default,
        test_embedding_cache,
        test_purity_by_group,
        test_regress_out_covariates_are_present,
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
