#!/usr/bin/env python
"""
What is actually in this h5ad's expression matrix?

Section A of diagnose_v120_run.py established that the pipeline reads the
matrix correctly -- gex_idx == raw_idx for every probe gene, names match,
var and X shapes agree. The problem is upstream of the pipeline:

    ACTB    detected in 0.02% of cells
    EEF1A1  detected in 0.00% of cells
    RPL13A  detected in 0.00% of cells

while the object's own obs claims a median of 1,406 genes detected per cell.
Those two statements cannot both describe the same matrix. This script decides
which one is wrong, and where the real counts are.

Read-only.

    python diagnose_what_is_X.py /data/.../MDL1856_analyzed_full_depth.h5ad [--n-cells 20000]

Checks
------
1  Inventory: X dtype/sparsity/range, every layer, .raw, obsm
2  Recompute total_counts and n_genes from X and compare against obs
3  Rank genes by detection FROM the matrix -- if the top genes are not the
   usual ribosomal/mitochondrial/ACTB suspects, the columns are wrong
4  Repeat 2 and 3 for every layer and for .raw, to find a matrix that IS
   consistent with obs
5  Column-permutation test: is the observed detection profile consistent with
   var_names being shuffled relative to X?
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)

PROBES = ["ACTB", "GAPDH", "B2M", "EEF1A1", "RPL13A", "PPIA", "RPS18",
          "TPT1", "UBC", "MALAT1"]


def banner(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def describe_matrix(M, name: str, var_names, obs, n_show: int = 15) -> dict:
    """Sparsity, depth and top-detected genes for one candidate matrix."""
    import scipy.sparse as sp

    out: dict = {"name": name}
    n_cells, n_genes = M.shape
    is_sparse = sp.issparse(M)
    out["shape"] = (n_cells, n_genes)
    out["sparse"] = is_sparse

    if is_sparse:
        M = M.tocsr()
        nnz = M.nnz
        data = M.data
        row_nnz = np.diff(M.indptr)
        row_sum = np.asarray(M.sum(axis=1)).ravel()
        col_nnz = np.asarray((M > 0).sum(axis=0)).ravel()
    else:
        M = np.asarray(M)
        nz = M != 0
        nnz = int(nz.sum())
        data = M[nz]
        row_nnz = nz.sum(axis=1)
        row_sum = M.sum(axis=1)
        col_nnz = nz.sum(axis=0)

    out["pct_nonzero"] = 100.0 * nnz / (n_cells * n_genes)
    out["min"] = float(data.min()) if data.size else np.nan
    out["max"] = float(data.max()) if data.size else np.nan
    out["integral"] = bool(np.allclose(data, np.round(data))) if data.size else False
    out["median_genes_per_cell"] = float(np.median(row_nnz))
    out["median_counts_per_cell"] = float(np.median(row_sum))

    print(f"\n  --- {name} ---")
    print(f"    shape                 : {n_cells:,} x {n_genes:,}"
          f"   sparse={is_sparse}")
    print(f"    % nonzero             : {out['pct_nonzero']:.3f}")
    print(f"    value range           : {out['min']:.4g} .. {out['max']:.4g}"
          f"   integral={out['integral']}")
    print(f"    median genes/cell     : {out['median_genes_per_cell']:,.0f}")
    print(f"    median counts/cell    : {out['median_counts_per_cell']:,.1f}")

    # Compare against what obs claims.
    for col, key in (("n_genes_by_counts", "median_genes_per_cell"),
                     ("total_counts", "median_counts_per_cell")):
        if col in obs.columns:
            claimed = float(np.median(obs[col].to_numpy(dtype=float)))
            got = out[key]
            ratio = got / claimed if claimed else np.nan
            verdict = "MATCH" if 0.8 < ratio < 1.25 else "*** MISMATCH ***"
            print(f"    obs['{col}'] median : {claimed:,.1f}"
                  f"   from matrix: {got:,.1f}   ratio={ratio:.3f}  {verdict}")

    # Top genes by detection: the identity test.
    order = np.argsort(col_nnz)[::-1][:n_show]
    top = [(str(var_names[i]), 100.0 * col_nnz[i] / n_cells) for i in order]
    print(f"    top {n_show} genes by detection rate:")
    print("      " + ", ".join(f"{g} {p:.1f}%" for g, p in top))
    usual = sum(
        1 for g, _ in top
        if g.startswith(("RPL", "RPS", "MT-", "MALAT1", "ACTB", "TMSB",
                         "EEF1", "B2M", "GAPDH", "FTL", "FTH1"))
    )
    print(f"      recognisable housekeeping/ribosomal/mito in top {n_show}: "
          f"{usual}")
    if usual >= 5:
        print("      -> looks like a real expression matrix")
    else:
        print("      *** does NOT look like a real expression matrix ***")

    # Named probes.
    pos = {str(v): i for i, v in enumerate(var_names)}
    rows = []
    for p in PROBES:
        i = pos.get(p)
        if i is None:
            rows.append({"gene": p, "present": False, "pct_detected": np.nan})
        else:
            rows.append({"gene": p, "present": True,
                         "pct_detected": round(100.0 * col_nnz[i] / n_cells, 3)})
    print("    housekeeping probes:")
    print("      " + pd.DataFrame(rows).to_string(index=False).replace("\n", "\n      "))
    out["top"] = top
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("--n-cells", type=int, default=20000)
    args = ap.parse_args()

    import anndata as ad

    print(f"reading {args.h5ad} ...")
    backed = ad.read_h5ad(args.h5ad, backed="r")
    print(f"  {backed.n_obs:,} cells x {backed.n_vars:,} vars")

    banner("1. INVENTORY")
    print(f"  X dtype        : {backed.X.dtype if backed.X is not None else None}")
    print(f"  layers         : {list(backed.layers.keys()) or 'none'}")
    print(f"  raw present    : {backed.raw is not None}")
    if backed.raw is not None:
        print(f"    raw shape    : {backed.raw.shape}")
    print(f"  obsm           : {list(backed.obsm.keys())}")
    print(f"  var columns    : {list(backed.var.columns)}")
    print(f"  obs columns    : {list(backed.obs.columns)}")
    print(f"  uns keys       : {list(backed.uns.keys())}")

    n = backed.n_obs
    if n > args.n_cells:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, size=args.n_cells, replace=False))
        print(f"\n  loading {args.n_cells:,}-cell subsample ...")
        sub = backed[idx].to_memory()
    else:
        sub = backed.to_memory()

    var_names = [str(v) for v in sub.var_names]
    obs = sub.obs

    banner("2-4. EVERY CANDIDATE MATRIX, vs WHAT obs CLAIMS")
    print("  A matrix is 'the real one' if its median genes/cell matches")
    print("  obs['n_genes_by_counts'] AND its top-detected genes are the usual")
    print("  ribosomal/mitochondrial suspects.")

    describe_matrix(sub.X, "X", var_names, obs)
    for key in sub.layers.keys():
        describe_matrix(sub.layers[key], f"layers['{key}']", var_names, obs)
    if sub.raw is not None:
        raw_names = [str(v) for v in sub.raw.var_names]
        describe_matrix(sub.raw.X, "raw.X", raw_names, obs)

    banner("5. COLUMN-PERMUTATION TEST")
    import scipy.sparse as sp
    M = sub.X.tocsr() if sp.issparse(sub.X) else np.asarray(sub.X)
    col_nnz = (np.asarray((M > 0).sum(axis=0)).ravel() if sp.issparse(M)
               else (M != 0).sum(axis=0))
    det = col_nnz / M.shape[0]
    print(f"  detection rate across genes: median={np.median(det):.5f} "
          f"mean={det.mean():.5f} max={det.max():.5f}")
    print(f"  genes detected in >50% of cells: {(det > 0.5).sum():,}")
    print(f"  genes detected in >10% of cells: {(det > 0.1).sum():,}")
    print(f"  genes never detected           : {(det == 0).sum():,}")
    print()
    print("  In a real matrix a few hundred genes exceed 50% detection.")
    print("  If that count is ~0 while obs claims ~1,400 genes per cell, the")
    print("  matrix is not a standard count matrix -- and if the top-detected")
    print("  genes are unrecognisable, the columns do not match var_names.")

    if "pct_counts_mt" in obs.columns:
        mt = [i for i, v in enumerate(var_names) if str(v).upper().startswith("MT-")]
        print(f"\n  MT- genes in var: {len(mt)}")
        if mt:
            mt_sum = (np.asarray(M[:, mt].sum(axis=1)).ravel() if sp.issparse(M)
                      else M[:, mt].sum(axis=1))
            tot = (np.asarray(M.sum(axis=1)).ravel() if sp.issparse(M)
                   else M.sum(axis=1))
            pct = 100.0 * mt_sum / np.maximum(tot, 1e-9)
            print(f"    %MT from matrix : median {np.median(pct):.2f}")
            print(f"    obs pct_counts_mt: median "
                  f"{np.median(obs['pct_counts_mt'].to_numpy(float)):.2f}")
            print("    A large disagreement here is independent confirmation "
                  "that X is not the matrix obs was computed from.")

    banner("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
