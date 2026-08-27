#!/usr/bin/env python
"""
Prove -- and if possible recover -- the var_names / X column permutation.

Established so far:
  * X's nonzeros-per-cell matches obs['n_genes_by_counts'] exactly (1,417),
    so X IS the matrix obs was computed from and its structure is intact.
  * But ACTB is detected in 0.025% of cells while CD79A, NDST1-AS1 and a
    CRISPR guide feature sit at ~99%.
  * %MT computed from the MT- labelled columns is 1.30 vs obs 3.02.

That combination means the gene LABELS are permuted relative to the columns.

`var` still carries scanpy's own per-gene statistics -- n_cells_by_counts,
mean_counts, pct_dropout_by_counts -- which were computed when the labels were
still aligned. They are a fingerprint for each gene. Matching them against the
per-column statistics measured from X both proves the permutation and, if the
fingerprint is unique enough, recovers it exactly.

Read-only. Nothing is written; if a mapping is found the script prints how to
apply it.

    python diagnose_var_permutation.py /data/.../MDL1856_analyzed_full_depth.h5ad
        [--chunk 20000]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)

PROBES = ["ACTB", "GAPDH", "B2M", "EEF1A1", "RPL13A", "PPIA", "RPS18",
          "TPT1", "UBC", "MALAT1", "MT-CO1", "MT-ND3"]


def banner(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("--chunk", type=int, default=20000)
    args = ap.parse_args()

    import anndata as ad
    import scipy.sparse as sp

    print(f"reading {args.h5ad} (backed) ...")
    A = ad.read_h5ad(args.h5ad, backed="r")
    n_obs, n_var = A.n_obs, A.n_vars
    print(f"  {n_obs:,} cells x {n_var:,} vars")

    var = A.var.copy()
    var_names = np.array([str(v) for v in A.var_names])

    needed = [c for c in ("n_cells_by_counts", "mean_counts",
                          "pct_dropout_by_counts") if c in var.columns]
    print(f"  usable var statistics: {needed or 'NONE'}")
    if "n_cells_by_counts" not in var.columns:
        print("  Cannot fingerprint without var['n_cells_by_counts']. Stopping.")
        return 1

    # ---------------------------------------------------------------- measure
    banner("1. MEASURE PER-COLUMN STATISTICS FROM X")
    print(f"  streaming in chunks of {args.chunk:,} cells ...")
    col_nnz = np.zeros(n_var, dtype=np.int64)
    col_sum = np.zeros(n_var, dtype=np.float64)
    for start in range(0, n_obs, args.chunk):
        stop = min(start + args.chunk, n_obs)
        M = A[start:stop].to_memory().X
        if sp.issparse(M):
            M = M.tocsc()
            col_nnz += np.diff(M.indptr)
            col_sum += np.asarray(M.sum(axis=0)).ravel()
        else:
            M = np.asarray(M)
            col_nnz += (M != 0).sum(axis=0)
            col_sum += M.sum(axis=0)
        print(f"    {stop:,}/{n_obs:,}", end="\r", flush=True)
    print(f"    {n_obs:,}/{n_obs:,} done            ")

    claimed = var["n_cells_by_counts"].to_numpy(dtype=np.int64)

    # ------------------------------------------------------------ is it off?
    banner("2. DO THE LABELS MATCH THE COLUMNS AS-IS?")
    same = int((col_nnz == claimed).sum())
    corr = float(np.corrcoef(col_nnz, claimed)[0, 1])
    print(f"  columns whose measured n_cells == var['n_cells_by_counts'] : "
          f"{same:,} / {n_var:,}")
    print(f"  correlation between measured and claimed                   : "
          f"{corr:.4f}")
    if same > 0.95 * n_var:
        print("\n  Labels and columns AGREE. The permutation hypothesis is "
              "wrong -- stop here and re-examine.")
        return 0
    print("\n  *** They do not agree. The gene labels do not describe the "
          "columns they sit on. ***")

    # Is it even a permutation? A permutation preserves the multiset.
    a = np.sort(col_nnz)
    b = np.sort(claimed)
    identical_multiset = bool(np.array_equal(a, b))
    print(f"  sorted measured == sorted claimed (a pure permutation?)     : "
          f"{identical_multiset}")
    if not identical_multiset:
        n_close = int((np.abs(a - b) <= 1).sum())
        print(f"    values within 1 of each other after sorting: "
              f"{n_close:,}/{n_var:,}")
        print("    If this is far below n_var, the columns are not merely "
              "reordered -- some are different data entirely.")

    # ------------------------------------------------------- recover mapping
    banner("3. CAN THE MAPPING BE RECOVERED?")
    # Fingerprint each gene by (n_cells_by_counts, rounded mean) to break ties.
    use_mean = "mean_counts" in var.columns
    if use_mean:
        claimed_mean = var["mean_counts"].to_numpy(dtype=float)
        measured_mean = col_sum / max(n_obs, 1)

    # Bucket on the EXACT integer cell count -- that comparison is safe, both
    # sides being counts of nonzeros. Then pair within each bucket by mean.
    # Rounding the mean to a fixed number of digits and demanding equality
    # fails on real data, because a chunked sum and scanpy's original sum
    # accumulate floating point in a different order; ordering within a bucket
    # is stable under that, exact equality is not.
    from collections import Counter, defaultdict
    buckets: dict[int, list[int]] = defaultdict(list)
    for j, v in enumerate(col_nnz):
        buckets[int(v)].append(j)

    mapping = np.full(n_var, -1, dtype=np.int64)   # var row -> X column
    unmatched = 0
    certain = 0
    by_mean = 0
    ambiguous = 0
    residuals: list[float] = []

    claimed_by_bucket: dict[int, list[int]] = defaultdict(list)
    for i, v in enumerate(claimed):
        claimed_by_bucket[int(v)].append(i)

    for count, var_rows in claimed_by_bucket.items():
        cols = buckets.get(count, [])
        if not cols:
            unmatched += len(var_rows)
            continue
        if len(var_rows) == 1 and len(cols) == 1:
            mapping[var_rows[0]] = cols[0]
            certain += 1
            continue
        if len(var_rows) != len(cols):
            # Sizes disagree: pair what we can by mean, leave the rest.
            unmatched += abs(len(var_rows) - len(cols))
        if not use_mean:
            # Nothing to break the tie with: refuse rather than guess.
            ambiguous += len(var_rows)
            continue
        vr = sorted(var_rows, key=lambda i: claimed_mean[i])
        cc = sorted(cols, key=lambda j: measured_mean[j])
        # Genes sharing BOTH statistics are interchangeable by this evidence.
        # Pairing them by sort order produces a plausible-looking mapping with
        # a zero residual that is nonetheless wrong -- verified on synthetic
        # data, where that mistake silently mis-assigned 22% of genes. Refuse
        # those instead of guessing.
        dup_claimed = Counter(round(float(claimed_mean[i]), 12) for i in vr)
        for i, j in zip(vr, cc):
            if dup_claimed[round(float(claimed_mean[i]), 12)] > 1:
                ambiguous += 1
                continue
            mapping[i] = j
            by_mean += 1
            denom = max(abs(claimed_mean[i]), 1e-12)
            residuals.append(abs(claimed_mean[i] - measured_mean[j]) / denom)

    resolved = int((mapping >= 0).sum())
    print(f"  bucketed on exact n_cells_by_counts, paired within bucket by "
          f"{'mean_counts' if use_mean else 'index'}")
    print(f"  genes mapped                : {resolved:,} / {n_var:,} "
          f"({100*resolved/n_var:.1f}%)")
    print(f"    unique on count alone     : {certain:,}")
    print(f"    resolved by mean ordering : {by_mean:,}")
    print(f"    refused (tied on both)    : {ambiguous:,}")
    print(f"  no matching column          : {unmatched:,}")
    if residuals:
        r = np.array(residuals)
        print(f"  relative mean discrepancy after pairing: "
              f"median {np.median(r):.2e}  p99 {np.quantile(r, 0.99):.2e}  "
              f"max {r.max():.2e}")
        print("    Near-zero means the pairing is right. Large values mean "
              "the within-bucket order did not survive and those genes are "
              "not reliably recovered.")

    if resolved == 0:
        print("\n  No mapping recoverable from these statistics.")
        return 0

    # How far off is it? A constant offset is a different (easier) story than
    # a scramble.
    ok = mapping >= 0
    delta = mapping[ok] - np.flatnonzero(ok)
    uniq = np.unique(delta)
    print(f"\n  distinct (column - var_row) offsets among matched genes: "
          f"{len(uniq):,}")
    if len(uniq) == 1:
        print(f"    CONSTANT OFFSET of {uniq[0]:+,}. The columns are rotated, "
              f"not scrambled.")
    else:
        print(f"    offsets range {delta.min():+,} .. {delta.max():+,} "
              f"-- a genuine scramble, not a shift")
        # Is X in alphabetical order of var_names?
        alpha = np.argsort(var_names, kind="stable")
        frac_alpha = float(np.mean(mapping[ok] == alpha[np.flatnonzero(ok)]))
        print(f"    fraction consistent with X being var_names sorted "
              f"alphabetically: {frac_alpha:.3f}")

    # ------------------------------------------------------------- validate
    banner("4. VALIDATE THE RECOVERED MAPPING")
    pos = {n: i for i, n in enumerate(var_names)}
    rows = []
    for p in PROBES:
        i = pos.get(p)
        if i is None:
            continue
        j = mapping[i]
        rows.append({
            "gene": p,
            "var_row": i,
            "current_col_pct_detected": round(100.0 * col_nnz[i] / n_obs, 3),
            "claimed_pct_detected": round(100.0 * claimed[i] / n_obs, 3),
            "recovered_col": int(j) if j >= 0 else None,
            "recovered_pct_detected": (round(100.0 * col_nnz[j] / n_obs, 3)
                                       if j >= 0 else None),
            "name_now_at_that_col": var_names[j] if j >= 0 else None,
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("  'claimed' comes from var and is what the gene SHOULD look like.")
    print("  If 'recovered' matches 'claimed' and 'current' does not, the")
    print("  mapping is correct and the object can be repaired.")

    # MT sanity under the recovered mapping.
    if "mt" in var.columns:
        mt_rows = np.flatnonzero(var["mt"].to_numpy().astype(bool))
        mt_cols = mapping[mt_rows]
        mt_cols = mt_cols[mt_cols >= 0]
        if len(mt_cols):
            tot = col_sum.sum()
            pct_now = 100.0 * col_sum[mt_rows].sum() / tot
            pct_fix = 100.0 * col_sum[mt_cols].sum() / tot
            print(f"\n  global %MT using current labels   : {pct_now:.2f}")
            print(f"  global %MT using recovered mapping: {pct_fix:.2f}")
            if "pct_counts_mt" in A.obs.columns:
                print(f"  obs['pct_counts_mt'] median       : "
                      f"{float(np.median(A.obs['pct_counts_mt'])):.2f}")

    banner("5. IF THE MAPPING VALIDATES, THIS REPAIRS THE OBJECT")
    print("""
  import anndata as ad, numpy as np
  A = ad.read_h5ad(SRC)
  order = np.load('recovered_mapping.npy')      # var row -> X column
  assert (order >= 0).all(), 'mapping incomplete -- do not use'
  A2 = A[:, order].copy()          # reorder COLUMNS to sit under their labels
  A2.var_names = A.var_names       # labels stay put
  A2.var = A.var.copy()
  A2.write_h5ad(DST)

  Do NOT repair in place, and re-run diagnose_what_is_X.py on the output:
  ACTB should come back at >90% detection and the top-detected genes should
  be ribosomal and mitochondrial.
""")
    if resolved == n_var:
        np.save("recovered_mapping.npy", mapping)
        print("  Complete mapping saved to recovered_mapping.npy")
    else:
        print(f"  Mapping is INCOMPLETE ({n_var - resolved:,} genes "
              f"unresolved) -- not saved. Prefer regenerating the h5ad from "
              f"source over a partial repair.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
