#!/usr/bin/env python
"""
Verify the v1.2.0 fixes against the real h5ad, without running the pipeline.

The regression suite (tests/test_v120_changes.py) runs on synthetic data and
on the handful of real guide IDs recoverable from the v1.1.0 report. This
script applies the same new code to the actual object, so the numbers can be
checked against what v1.1.0 produced.

Usage:
    python verify_on_real_data.py /path/to/MDL1856_analyzed_full_depth.h5ad \\
        [--grna-whitelist whitelists/MDL1856_grna_whitelist.csv]

Read-only. Prints, with the v1.1.0 value alongside each:

  1. guide-ID parse coverage and distinct target count
  2. whether guides would still leak into the GEX matrix
  3. per-hashtag separability verdicts
  4. the labels that will appear on plot axes
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perturbseq_report.config import GuideConfig, HTOConfig, ModalityConfig
from perturbseq_report.guide import GuideParser
from perturbseq_report.hto import compute_thresholds
from perturbseq_report.modalities import Modality
from perturbseq_report.whitelists import load_guide_whitelist

V110 = {
    "unparsed": 168,
    "n_guides": 321,
    "distinct_targets": 186,
    "guides_with_ensg": 0,
    "flagged_hashtags": ["prot:hash.C"],
}


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("--grna-whitelist", default=None)
    args = ap.parse_args()

    import anndata as ad

    print(f"reading {args.h5ad} (backed) ...")
    adata = ad.read_h5ad(args.h5ad, backed="r")
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} vars")

    # ---------------------------------------------------------------- guides
    vn = [str(v) for v in adata.var_names]
    guide_ids = [
        v for v in vn
        if re.search(r"(_target_version_|_spacer_|[ACGT]{18,}$)", v)
    ]
    for key in adata.obsm.keys():
        if re.search("guide|grna|crispr|sgrna", key, re.I):
            for uk in adata.uns.keys():
                if re.search("guide|grna|crispr|sgrna", uk, re.I):
                    try:
                        names = [str(x) for x in np.asarray(adata.uns[uk]).ravel()]
                        if len(names) > len(guide_ids):
                            guide_ids = names
                    except Exception:
                        pass

    wl = load_guide_whitelist(args.grna_whitelist) if args.grna_whitelist else None

    banner("1. GUIDE-ID PARSING")
    parser = GuideParser(GuideConfig(), wl)
    m = parser.parse_all(guide_ids)
    n_ensg = int(m["target_ensg"].notna().sum())
    n_targets = int(m["target_key"].nunique())
    print(f"  guides                    : {len(guide_ids)}   "
          f"(v1.1.0: {V110['n_guides']})")
    print(f"  unparsed                  : {len(parser.unparsed):<5} "
          f"(v1.1.0: {V110['unparsed']})")
    print(f"  guides yielding an ENSG   : {n_ensg:<5} "
          f"(v1.1.0: {V110['guides_with_ensg']})")
    print(f"  distinct target groups    : {n_targets:<5} "
          f"(v1.1.0: {V110['distinct_targets']} -- inflated by the parse failure)")
    print(f"  families                  : {sorted(set(m['family']))}")
    print(f"  matched controls resolved : {int((m['role'] == 'matched_control').sum())}")
    print(f"  two-name disagreements    : {len(parser.conflicts)}")
    if parser.unparsed:
        print("  STILL UNPARSED:")
        for g in parser.unparsed[:10]:
            print(f"    {g}")
    if wl is not None and parser.unlisted:
        print(f"  not in whitelist          : {len(parser.unlisted)}")
        for g in parser.unlisted[:10]:
            print(f"    {g}")

    print("\n  role x family:")
    print(m.groupby(["family", "role"]).size().to_string())

    print("\n  guides per target group (top 12):")
    print(m["target_key"].value_counts().head(12).to_string())

    banner("2. WOULD GUIDES STILL LEAK INTO THE GEX MATRIX?")
    in_var = pd.Index(vn).isin(set(map(str, guide_ids)))
    ft_cols = [
        c for c in adata.var.columns
        if re.search(r"(feature_type|modal|assay|librar|category)", c, re.I)
    ]
    print(f"  guide IDs present in var       : {int(in_var.sum())}")
    print(f"  feature-type column present    : {ft_cols or 'NONE'}")
    print(f"  v1.1.0 GEX width               : {adata.n_vars:,} "
          f"(guides INCLUDED -- this is the bug)")
    print(f"  v1.2.0 GEX width               : {adata.n_vars - int(in_var.sum()):,} "
          f"(guides removed by name back-fill)")
    print("\n  Guide features in the GEX matrix feed HVG selection, PCA and the")
    print("  marker test. In the v1.1.0 report, clusters 0/1/2/3/8 had guide IDs")
    print("  listed as their marker genes.")

    banner("3. HASHTAG SEPARABILITY")
    hto_cols = [c for c in adata.obs.columns if re.match(r"(prot:)?hash", c, re.I)]
    hto_names = [v for v in vn if re.match(r"(prot:)?hash", v, re.I)]
    if hto_names:
        idx = [vn.index(n) for n in hto_names]
        X = adata[:, idx].to_memory().X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        names = hto_names
    elif hto_cols:
        X = adata.obs[hto_cols].to_numpy(dtype=float)
        names = hto_cols
    else:
        print("  no hashtag matrix found")
        return 0

    logx = np.log1p(np.asarray(X, dtype=float))
    clr = pd.DataFrame(logx - logx.mean(axis=0, keepdims=True), columns=names)
    th = compute_thresholds(clr, HTOConfig())
    cols = ["hashtag", "threshold", "valley_ratio", "separation_sd",
            "frac_background", "frac_positive", "separability"]
    print(th[cols].round(3).to_string(index=False))
    flagged = th.loc[th["separability"] != "clean", "hashtag"].tolist()
    print(f"\n  v1.1.0 flagged : {V110['flagged_hashtags']}")
    print(f"  v1.2.0 flags   : {flagged}")
    print("\n  Expected: prot:hash.C is NOT flagged (it has a deep trough),")
    print("  prot:hash.F IS flagged (it has no trough), and prot:hash.D is")
    print("  degenerate (98.4% zero UMIs).")
    if bool(th["raised_to_floor"].all()):
        print("\n  NOTE: every threshold is still set by the 3-SD floor rather")
        print("  than the configured background-quantile rule. v1.2.0 says so in")
        print("  the report instead of describing a rule that did not apply.")

    banner("4. PLOT LABELS")
    print("  first 25 guide labels that will appear on axes:\n")
    for _, r in m.head(25).iterrows():
        print(f"    {r['short_label']:<28} <- {r['guide_id'][:64]}")
    dupes = m["short_label"].duplicated().sum()
    print(f"\n  duplicate labels: {int(dupes)} (must be 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
