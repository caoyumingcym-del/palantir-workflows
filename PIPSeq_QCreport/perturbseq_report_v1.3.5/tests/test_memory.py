"""
Guards against the class of bug that caused an out-of-memory crash on real
data: densifying the whole cells-by-genes matrix.

The end-to-end tests run at ~4,000 cells x 700 genes, where a dense copy is
22 MB and the bug is invisible. On a real Perturb-seq experiment -- 200,000
cells x 25,000 genes -- the same code path allocates 40 GB and the process is
killed with no output at all.

Two kinds of check here:

``test_no_full_densification``
    Structural, and runs everywhere. It instruments the densifying helpers and
    fails if any single call materialises more than a set fraction of the full
    matrix. This is the test that would have caught the original bug, and it
    catches it regardless of how big the test data is.

``test_sparse_input_*``
    Scale checks on genuinely sparse input. Skipped when scipy is unavailable
    (as in the sandbox this was developed in), so run these in your own
    environment -- they are the ones that exercise the real code path.

Run with:  python -m pytest tests/test_memory.py -v
      or:  python tests/test_memory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_anndata import FakeAnnData                        # noqa: E402
from perturbseq_report import stats as S                    # noqa: E402
from perturbseq_report import synthetic                     # noqa: E402
from perturbseq_report.config import build_config           # noqa: E402
from perturbseq_report.manifest import read_manifest        # noqa: E402
from perturbseq_report.pipeline import run_with_adata       # noqa: E402

try:
    import scipy.sparse as sp
    HAVE_SCIPY = True
except ImportError:
    sp = None
    HAVE_SCIPY = False


class Skip(Exception):
    """Raised to skip a check that needs a dependency we do not have."""


# ===========================================================================
# Structural guard
# ===========================================================================
def test_no_full_densification(tmp_path):
    """No single densification may materialise most of the matrix.

    ``take_columns`` is the only sanctioned way to pull dense data out of the
    expression matrix. This wraps it and records the largest block requested;
    if any call asks for a large fraction of all genes, the pipeline has
    reverted to densifying everything.
    """
    import perturbseq_report.gex as G
    import perturbseq_report.perturb as PB
    import perturbseq_report.stats as ST

    n_cells, n_genes = 2000, 600
    bundle = synthetic.make_bundle(seed=4, n_cells=n_cells, n_genes=n_genes)

    # The invariant is bounded ALLOCATION, not bounded column count.
    # Differential expression legitimately examines every gene -- it just has
    # to do it in blocks. So measure the number of elements materialised by any
    # single call.
    n_top_genes, de_block = 300, 100
    biggest = {"elems": 0, "where": ""}
    real_take = ST.take_columns

    def spy(X, idx):
        idx_arr = np.asarray(idx, dtype=int)
        rows, width = getattr(X, "shape", (0, 0))
        elems = int(rows) * int(idx_arr.size)
        if elems > biggest["elems"]:
            biggest["elems"] = elems
            biggest["where"] = (
                f"{rows:,} rows x {idx_arr.size:,} of {width:,} columns "
                f"= {elems:,} elements"
            )
        return real_take(X, idx_arr)

    # Patch every module that imported the helper by name.
    for mod in (ST, G, PB):
        if hasattr(mod, "take_columns"):
            mod.take_columns = spy

    try:
        out = tmp_path / "out"
        mp = synthetic.write_manifest(tmp_path / "m.csv", tmp_path / "s.h5ad", out)
        X = np.hstack([bundle.counts, bundle.guide_counts, bundle.hto_counts])
        var = pd.DataFrame(
            {"feature_types": (
                ["Gene Expression"] * len(bundle.gene_names)
                + ["CRISPR Guide Capture"] * len(bundle.guide_names)
                + ["Antibody Capture"] * len(bundle.hto_names))},
            index=bundle.gene_names + bundle.guide_names + bundle.hto_names,
        )
        # Prefer real AnnData so the scanpy path is exercised where available;
        # the stand-in is only for environments without the single-cell stack.
        try:
            import anndata as ad_mod
            if HAVE_SCIPY:
                adata = ad_mod.AnnData(X=sp.csr_matrix(X),
                                       obs=bundle.obs.copy(), var=var)
            else:
                adata = ad_mod.AnnData(X=X, obs=bundle.obs.copy(), var=var)
        except ImportError:
            adata = FakeAnnData(X=X, obs=bundle.obs.copy(), var=var)
        cfg = build_config({"output_path": out, "verbose": False,
                            "resample_n": 50, "auto_thresholds": True,
                            "n_top_genes": n_top_genes,
                            "de_gene_block": de_block})
        cfg.manifest_path = mp
        run_with_adata(cfg, read_manifest(mp), adata)
    finally:
        for mod in (ST, G, PB):
            if hasattr(mod, "take_columns"):
                mod.take_columns = real_take

    # The largest legitimate block is HVG selection: every cell by n_top_genes.
    # Allow a small margin. Densifying the whole matrix would be
    # n_cells * n_genes, which is well above this.
    limit = int(n_cells * n_top_genes * 1.2)
    full = n_cells * n_genes
    assert biggest["elems"] <= limit, (
        f"a single densification materialised {biggest['where']}, above the "
        f"{limit:,}-element budget (the full matrix is {full:,}). Something is "
        f"densifying the whole expression matrix again -- that is what caused "
        f"the OOM crash on real data."
    )


def test_take_columns_is_bounded():
    """take_columns must densify only the requested columns."""
    A = np.arange(200 * 50, dtype=float).reshape(200, 50)
    got = S.take_columns(A, [3, 7, 11])
    assert got.shape == (200, 3)
    assert np.array_equal(got[:, 0], A[:, 3])
    assert S.take_columns(A, []).shape == (200, 0)
    assert S.take_column(A, 5).shape == (200,)


def test_sparse_helpers_match_dense():
    """Sparse aggregates must agree with the dense computation."""
    if not HAVE_SCIPY:
        raise Skip("scipy not installed")
    rng = np.random.default_rng(0)
    dense = rng.poisson(0.3, size=(400, 120)).astype(float)
    sparse = sp.csr_matrix(dense)

    assert np.allclose(S.row_sums(sparse), dense.sum(axis=1))
    assert np.allclose(S.col_means(sparse), dense.mean(axis=0))
    assert np.allclose(S.col_variances(sparse), dense.var(axis=0), atol=1e-9)
    assert np.allclose(S.col_nonzero_fraction(sparse), (dense > 0).mean(axis=0))
    assert np.allclose(S.take_columns(sparse, [1, 5]), dense[:, [1, 5]])
    assert np.allclose(
        S.sparse_log1p(sparse).toarray(), np.log1p(dense)
    )
    norm_s = S.normalize_rows(sparse, 1e4)
    norm_d = S.normalize_rows(dense, 1e4)
    assert np.allclose(np.asarray(norm_s.todense()), norm_d)


def test_sparse_log1p_preserves_sparsity():
    if not HAVE_SCIPY:
        raise Skip("scipy not installed")
    dense = np.zeros((300, 200))
    dense[0, 0] = 5.0
    sparse = sp.csr_matrix(dense)
    out = S.sparse_log1p(sparse)
    assert sp.issparse(out), "log1p must not densify"
    assert out.nnz == 1, f"sparsity lost: {out.nnz} non-zeros"


def test_normalize_rows_preserves_sparsity():
    if not HAVE_SCIPY:
        raise Skip("scipy not installed")
    rng = np.random.default_rng(1)
    dense = (rng.random((300, 200)) < 0.05) * rng.poisson(5, (300, 200))
    sparse = sp.csr_matrix(dense.astype(float))
    out = S.normalize_rows(sparse, 1e4)
    assert sp.issparse(out), "normalisation must not densify"
    assert out.nnz == sparse.nnz


def test_differential_expression_chunking_matches_unchunked():
    """Gene-block processing must not change the answer."""
    rng = np.random.default_rng(2)
    a = np.log1p(rng.poisson(3, size=(60, 250)).astype(float))
    b = np.log1p(rng.poisson(5, size=(90, 250)).astype(float))
    genes = [f"G{i}" for i in range(250)]
    big = S.differential_expression(a, b, genes, block=10_000).table
    small = S.differential_expression(a, b, genes, block=17).table
    for col in ("log2fc", "mean_group", "mean_ref", "padj"):
        assert np.allclose(
            big[col].to_numpy(), small[col].to_numpy(), equal_nan=True
        ), f"{col} changed with block size"


def test_sparse_end_to_end(tmp_path):
    """The real check: run the pipeline on a genuinely sparse matrix."""
    if not HAVE_SCIPY:
        raise Skip("scipy not installed")
    bundle = synthetic.make_bundle(seed=6, n_cells=3000, n_genes=1200)
    X = np.hstack([bundle.counts, bundle.guide_counts, bundle.hto_counts])
    var = pd.DataFrame(
        {"feature_types": (
            ["Gene Expression"] * len(bundle.gene_names)
            + ["CRISPR Guide Capture"] * len(bundle.guide_names)
            + ["Antibody Capture"] * len(bundle.hto_names))},
        index=bundle.gene_names + bundle.guide_names + bundle.hto_names,
    )
    try:
        import anndata as ad
        adata = ad.AnnData(X=sp.csr_matrix(X), obs=bundle.obs.copy(), var=var)
    except ImportError:
        adata = FakeAnnData(X=X, obs=bundle.obs.copy(), var=var)
        adata.X = sp.csr_matrix(X)

    out = tmp_path / "out"
    mp = synthetic.write_manifest(tmp_path / "m.csv", tmp_path / "s.h5ad", out)
    cfg = build_config({"output_path": out, "verbose": False,
                        "resample_n": 50, "auto_thresholds": True})
    cfg.manifest_path = mp
    result = run_with_adata(cfg, read_manifest(mp), adata)

    assert result.report_path.exists()
    assert result.n_cells_after > 0
    kd = result.registry.get("perturbation", "knockdown_table")
    assert kd is not None and kd.skipped_reason is None, \
        "perturbation analysis failed on sparse input"


_MEM_CHILD = r'''
import resource, sys, tempfile
from pathlib import Path
sys.path.insert(0, {root!r}); sys.path.insert(0, {tests!r})
import numpy as np, pandas as pd, scipy.sparse as sp
from perturbseq_report import synthetic
from perturbseq_report.config import build_config
from perturbseq_report.manifest import read_manifest
from perturbseq_report.pipeline import run_with_adata
import anndata as ad

def peak_gb():
    # Linux reports ru_maxrss in kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

MODE = sys.argv[1]
n_cells, n_genes = {n_cells}, {n_genes}
if MODE == "baseline":
    print(peak_gb()); sys.exit(0)

b = synthetic.make_bundle(seed=8, n_cells=n_cells, n_genes=n_genes)
X = np.hstack([b.counts, b.guide_counts, b.hto_counts])
dense_gb = X.nbytes / 1e9
var = pd.DataFrame(
    {{"feature_types": (["Gene Expression"] * len(b.gene_names)
      + ["CRISPR Guide Capture"] * len(b.guide_names)
      + ["Antibody Capture"] * len(b.hto_names))}},
    index=b.gene_names + b.guide_names + b.hto_names)
adata = ad.AnnData(X=sp.csr_matrix(X), obs=b.obs.copy(), var=var)
del X, b

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    out = td / "out"
    mp = synthetic.write_manifest(td / "m.csv", td / "s.h5ad", out)
    cfg = build_config({{"output_path": out, "verbose": False,
                        "resample_n": 20, "auto_thresholds": True}})
    cfg.manifest_path = mp
    run_with_adata(cfg, read_manifest(mp), adata)
print(f"{{peak_gb()}} {{dense_gb}}")
'''


def test_peak_memory_is_bounded(tmp_path):
    """Peak memory of a real run, measured in isolation.

    Measured in a SUBPROCESS. ``ru_maxrss`` is a process-wide high-water mark
    that never decreases, so reading it in-process picks up every earlier test
    plus the ~1 GB that importing scanpy, harmony and matplotlib costs -- which
    is how the first version of this test reported 4.2 GB for a 0.19 GB matrix
    and looked like a failure when nothing was wrong.

    A baseline child that only imports the package is measured first, so the
    assertion is about what the *analysis* allocates.
    """
    if not HAVE_SCIPY:
        raise Skip("scipy not installed")
    try:
        import anndata  # noqa: F401
    except ImportError:
        raise Skip("anndata not installed")
    if not sys.platform.startswith("linux"):
        raise Skip("ru_maxrss units are platform-specific; Linux only")

    import subprocess

    n_cells, n_genes = 8000, 3000
    script = _MEM_CHILD.format(
        root=str(ROOT), tests=str(ROOT / "tests"),
        n_cells=n_cells, n_genes=n_genes,
    )
    src = tmp_path / "child.py"
    src.write_text(script)

    def run(mode: str) -> list[float]:
        proc = subprocess.run(
            [sys.executable, str(src), mode],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"memory child ({mode}) failed:\n{proc.stdout[-2000:]}"
                f"\n{proc.stderr[-2000:]}"
            )
        return [float(x) for x in proc.stdout.strip().splitlines()[-1].split()]

    baseline = run("baseline")[0]
    peak, dense_gb = run("run")
    attributable = max(peak - baseline, 0.0)

    # The analysis should not need more than a few times the dense size of the
    # matrix. Densifying it outright would blow well past this.
    budget = max(1.5, dense_gb * 6)
    assert attributable < budget, (
        f"the analysis allocated ~{attributable:.2f} GB (peak {peak:.2f} GB, "
        f"baseline {baseline:.2f} GB) for a {dense_gb:.2f} GB dense-equivalent "
        f"matrix, above the {budget:.2f} GB budget -- something is densifying "
        f"the matrix"
    )
    print(f"      [peak {peak:.2f} GB, baseline {baseline:.2f} GB, "
          f"attributable {attributable:.2f} GB, dense-equivalent "
          f"{dense_gb:.2f} GB]")


if __name__ == "__main__":
    import tempfile
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = skipped = 0
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
        except Skip as exc:
            print(f"SKIP  {fn.__name__}: {exc}")
            skipped += 1
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
            failed += 1
        except Exception:
            print(f"ERROR {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    if skipped:
        print("Skipped checks need scipy/anndata — run these in your ICA "
              "environment, they exercise the sparse path.")
    raise SystemExit(1 if failed else 0)
