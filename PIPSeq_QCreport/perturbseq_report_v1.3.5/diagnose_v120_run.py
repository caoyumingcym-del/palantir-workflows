#!/usr/bin/env python
"""
Diagnose the four problems seen in the v1.2.0 run on MDL-1856.

Read-only. Does not run the pipeline and does not write to the h5ad. It
replicates the pipeline's own code path at reduced scale so that any
discrepancy is the pipeline's, not a reimplementation's.

Usage
-----
    python diagnose_v120_run.py /data/.../MDL1856_analyzed_full_depth.h5ad \\
        --grna-whitelist  whitelists/grna_whitelist_MDL1856.csv \\
        --hashtag-whitelist whitelists/hashtag_whitelist_MDL1856.csv \\
        [--n-cells 25000] [--full]

Sections
--------
A  GENE LOOKUP ALIGNMENT   <-- the critical one. The report shows target genes
                               detected in 0.01-0.7% of control cells and
                               mean_perturbed of exactly 0, which cannot be
                               real expression. Either the resolved column is
                               the wrong gene, or X_log and var_names are
                               shifted relative to each other by the 321 guide
                               features v1.2.0 now removes from GEX.
B  HOUSEKEEPING SANITY      If ACTB is not detected in most cells through the
                               pipeline path, the matrix is wrong and nothing
                               downstream matters.
C  HASHTAG THRESHOLD SWEEP  Per-tag positive rates against what the declared
                               design allows, swept over the SD floor.
D  FALLBACK CONTROLS        Are guide-unassigned cells in a family usable as a
                               control pool where that family has no NTCs?
E  FAMILY CONFLICT          Is the 14.7% conflict rate real, or inflated by
                               hashtag over-calling?
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perturbseq_report.config import (
    EmbeddingConfig, GuideConfig, HTOConfig, ModalityConfig,
)
from perturbseq_report.guide import call_guides
from perturbseq_report.hto import compute_thresholds, normalise
from perturbseq_report.modalities import Modality, split_modalities
from perturbseq_report.perturb import build_gene_index, resolve_gene
from perturbseq_report.stats import normalize_rows, sparse_log1p, take_column
from perturbseq_report.whitelists import (
    load_guide_whitelist, load_hashtag_whitelist,
)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

# Genes the report claims are undetectable. All are ordinary expressed genes.
SUSPECT_GENES = [
    "ABT1", "BMS1", "EIF2S2", "DCAF13", "AKT2", "CDH1", "FOXD3",
    "TSR2", "HIST1H2AM", "CDC5L", "RPS12", "XRN2",
]
HOUSEKEEPING = [
    "ACTB", "GAPDH", "B2M", "TUBB", "RPL13A", "PPIA", "EEF1A1",
    "RPS18", "TPT1", "UBC",
]


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def densify(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def col_stats(col: np.ndarray) -> tuple[float, float]:
    """(mean on the linear scale, fraction of cells detected)."""
    col = np.asarray(col, dtype=float).ravel()
    return float(np.expm1(col).clip(0).mean()), float((col > 0).mean())


# ===========================================================================
def section_a_alignment(adata_sub, cfg_mod, guide_regexes, wl) -> None:
    banner("A. GENE LOOKUP ALIGNMENT  (the critical check)")

    raw_names = [str(v) for v in adata_sub.var_names]
    print(f"  raw object: {adata_sub.n_obs:,} cells x {len(raw_names):,} vars")

    split = split_modalities(adata_sub, cfg_mod, guide_regexes)
    gex = split.gex
    gex_names = [str(v) for v in gex.var.index]

    # --- shape agreement. A mismatch here IS the bug, full stop. -----------
    n_cols = int(gex.X.shape[1])
    print(f"  gex.var length            : {len(gex_names):,}")
    print(f"  gex.X columns             : {n_cols:,}")
    if len(gex_names) != n_cols:
        print("  *** FATAL: var and X disagree. Every gene lookup is shifted. ***")
    else:
        print("  var/X shapes agree.")
    print(f"  guide features removed    : {split.guide.n_features}")
    print(f"  hashtag features removed  : {split.hto.n_features}")

    # --- replicate gex.py's transform exactly ------------------------------
    X_log = sparse_log1p(normalize_rows(gex.X, EmbeddingConfig().target_sum))
    print(f"  X_log shape               : {tuple(X_log.shape)}")

    index = build_gene_index(gex_names)
    raw_pos = {n: i for i, n in enumerate(raw_names)}

    ensg_by_symbol = {}
    if wl is not None:
        m = wl.df
        if "target_gene" in m.columns and "target_ensg" in m.columns:
            for g, e in zip(m["target_gene"], m["target_ensg"]):
                if g and e:
                    ensg_by_symbol[str(g)] = str(e)

    rows = []
    for sym in SUSPECT_GENES + HOUSEKEEPING:
        gi = resolve_gene(index, sym, ensg_by_symbol.get(sym))
        rec = {"gene": sym, "resolved": gi is not None}
        if gi is None:
            rec.update({"pipeline_name": "-", "mean": np.nan, "frac_det": np.nan,
                        "raw_idx": raw_pos.get(sym, -1), "gex_idx": -1,
                        "shift": np.nan})
            rows.append(rec)
            continue
        pipe_name = gex_names[gi]
        mean_v, frac_v = col_stats(densify(take_column(X_log, gi)))
        rec.update({
            "pipeline_name": pipe_name,
            "mean": round(mean_v, 4),
            "frac_det": round(frac_v, 4),
            "gex_idx": gi,
            "raw_idx": raw_pos.get(sym, -1),
        })
        # If the pipeline resolved to a DIFFERENT name than the gene asked
        # for, that is the smoking gun; report how far off it is.
        rec["name_match"] = (pipe_name == sym) or (
            pipe_name == ensg_by_symbol.get(sym, "\0")
        )
        rows.append(rec)

    df = pd.DataFrame(rows)
    print("\n  through the pipeline path:")
    print(df.to_string(index=False))

    bad = df[(df["resolved"]) & (~df["name_match"].fillna(True))]
    if len(bad):
        print("\n  *** resolved to the WRONG var name for "
              f"{len(bad)} gene(s). ***")
        print("  If the gex_idx -> pipeline_name offset is constant, the "
              "matrix is shifted.")

    # --- ground truth straight from the raw object -------------------------
    print("\n  same genes read directly from the raw object (ground truth):")
    truth = []
    for sym in SUSPECT_GENES + HOUSEKEEPING:
        i = raw_pos.get(sym)
        if i is None:
            truth.append({"gene": sym, "in_raw_var": False,
                          "raw_mean": np.nan, "raw_frac_det": np.nan})
            continue
        col = densify(adata_sub[:, [i]].X)
        # raw X may be counts or already normalised; report both readings
        m_lin = float(np.asarray(col, float).mean())
        f_det = float((np.asarray(col, float) > 0).mean())
        truth.append({"gene": sym, "in_raw_var": True,
                      "raw_mean": round(m_lin, 4),
                      "raw_frac_det": round(f_det, 4)})
    tdf = pd.DataFrame(truth)
    print(tdf.to_string(index=False))

    merged = df.merge(tdf, on="gene")
    disagree = merged[
        merged["in_raw_var"] & merged["frac_det"].notna()
        & (merged["raw_frac_det"] > 0.2) & (merged["frac_det"] < 0.05)
    ]
    print()
    if len(disagree):
        print(f"  *** {len(disagree)} gene(s) are well detected in the raw "
              f"object but near-absent through the pipeline path. ***")
        print(disagree[["gene", "raw_frac_det", "frac_det",
                        "pipeline_name"]].to_string(index=False))
        print("\n  That is the bug producing pct_knockdown=100 with "
              "mean_perturbed=0 in the report.")
    else:
        print("  No gene is well detected in raw but absent through the "
              "pipeline. Alignment looks OK -- if the report still shows "
              "frac_detected ~0.0006, suspect var_names being Ensembl IDs "
              "while the report's target_gene is a symbol.")

    # --- what ARE the var_names? -------------------------------------------
    print("\n  var_name style check (first 8 non-guide names):")
    print("   ", gex_names[:8])
    n_ensg = sum(1 for n in gex_names if n.upper().startswith("ENSG"))
    print(f"    names that look like Ensembl IDs: {n_ensg:,} / {len(gex_names):,}")


# ===========================================================================
def section_c_hashtags(adata, hto_wl) -> None:
    banner("C. HASHTAG THRESHOLDS vs THE DECLARED DESIGN")
    cols = [c for c in adata.obs.columns if re.match(r"(prot:)?hash\.", c, re.I)]
    if not cols:
        print("  no hashtag obs columns found")
        return
    X = adata.obs[cols].to_numpy(dtype=float)
    mod = Modality(kind="hto", X=X, names=cols, source="obs",
                   obs_names=[str(i) for i in adata.obs.index])
    clr = normalise(mod)

    # What the design says each tag should cover.
    #
    # Assumes samples are equally represented, and aliquots equally split
    # within a sample. Crude, but it only has to be right to a factor of ~2 to
    # separate "roughly as designed" from "called in three times as many cells
    # as the design has room for".
    expected = None
    if hto_wl is not None:
        df = hto_wl.df
        n_samples = max(df["demux_id"].nunique(), 1)
        exp = Counter()
        for demux, sub in df.groupby("demux_id"):
            per_row = (1.0 / n_samples) / len(sub)
            for s in sub["hashtag_set_key"]:
                for t in s:
                    exp[t] += per_row
        expected = exp
        n_tagged = int((df["hashtag_set_key"].map(len) > 0).sum())
        print(f"  declared: {n_tagged} tagged combination(s), "
              f"{n_samples} sample(s)")
        for _, r in df.iterrows():
            key = "+".join(r["hashtag_set_key"]) or "(untagged)"
            print(f"    {r['demux_id']:<8} {str(r.get('aliquot','')):<4} {key}")

    print("\n  sweep of the SD floor (min_threshold_background_sd).")
    print("  positive_quantile is forced low so the FLOOR always binds -- on")
    print("  this data every threshold is already set by the floor, and a")
    print("  sweep where the quantile sometimes wins would not move at all.")
    header = f"  {'SD':>4} {'tagged-ok%':>10} {'ambig%':>8} {'no-tag%':>7}  per-tag positive rate"
    print(header)
    for sd in (3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0):
        cfg = HTOConfig(min_threshold_background_sd=sd, positive_quantile=0.0)
        th = compute_thresholds(clr, cfg)
        thr = th.set_index("hashtag")["threshold"].reindex(clr.columns).to_numpy()
        pos = clr.to_numpy(float) > thr[None, :]
        depth = X.sum(axis=1)
        pos[depth <= cfg.min_reads, :] = False
        npos = pos.sum(axis=1)
        rates = pos.mean(axis=0)
        if hto_wl is not None:
            lookup = set(hto_wl.lookup().keys())
            keys = [tuple(sorted(np.array(cols)[r])) for r in pos]
            # Split resolved into tagged vs the untagged bucket. Without this,
            # over-thresholding looks like a triumph: every cell loses all its
            # tags, the empty set matches the declared untagged sample, and
            # "resolved" climbs to 100% while the demultiplexing has actually
            # collapsed.
            res_tagged = sum(1 for k in keys if k and k in lookup)
            res_untag = sum(1 for k in keys if not k and () in lookup)
            neg = sum(1 for k in keys if not k and () not in lookup)
            amb = len(keys) - res_tagged - res_untag - neg
            n = len(keys)
            line = (f"  {sd:>4.1f} {100*res_tagged/n:>10.1f} "
                    f"{100*amb/n:>8.1f} "
                    f"{100*(neg + res_untag)/n:>7.1f}  ")
        else:
            n = len(npos)
            line = f"  {sd:>4.1f} {'-':>10} {'-':>8} {'-':>7}  "
        line += " ".join(
            f"{c.replace('prot:hash.','')}={r:.3f}" for c, r in zip(cols, rates)
        )
        print(line)
        print("       thresholds: " + " ".join(
            f"{c.replace('prot:hash.','')}={t:.2f}" for c, t in zip(cols, thr)
        ) + f"   sum(rates)={rates.sum():.3f}")

    if expected:
        print("\n  observed vs design-allowed share, at the current 3-SD floor:")
        cfg = HTOConfig()
        th = compute_thresholds(clr, cfg)
        thr = th.set_index("hashtag")["threshold"].reindex(clr.columns).to_numpy()
        pos = clr.to_numpy(float) > thr[None, :]
        for c, r in zip(cols, pos.mean(axis=0)):
            e = expected.get(c, 0.0)
            flag = "  <-- over-called" if e and r > 2 * e else ""
            print(f"    {c:<16} observed {r:.3f}   design ~{e:.3f}{flag}")


# ===========================================================================
def section_d_fallback(adata_sub, split, ga) -> None:
    banner("D. FALLBACK CONTROLS FOR FAMILIES WITH NO NTCs")
    pc = ga.per_cell
    obs = adata_sub.obs
    mapping = ga.mapping

    ntc_by_fam = (
        mapping[mapping["is_ntc"]].groupby("family")["guide_id"].size().to_dict()
    )
    all_fams = sorted(set(mapping["family"].astype(str)))
    print(f"  NTC guides per family: {ntc_by_fam}")
    missing = [f for f in all_fams if ntc_by_fam.get(f, 0) == 0]
    print(f"  families with NO NTC guides: {missing or 'none'}")

    assigned = pc["guide_is_assigned"].to_numpy(bool)
    fam = pc["family"].astype("object")
    is_ntc = pc["is_ntc"].fillna(False).to_numpy(bool)

    # Which family do UNASSIGNED cells belong to? Their guide is unknown, so
    # family has to come from the hashtag. Report how many are recoverable.
    print(f"\n  cells with no guide assignment: {int((~assigned).sum()):,} "
          f"({100*(~assigned).mean():.1f}%)")
    hcol = next((c for c in obs.columns if "hto_family" in c), None)
    if hcol:
        print("  their hashtag family:")
        print(obs.loc[~assigned, hcol].value_counts().head(8).to_string())
    else:
        print("  (hto_family not on obs in this object -- the pipeline adds it "
              "at runtime; run this after a pipeline run to see the split)")

    qc_cols = [c for c in ("total_counts", "n_genes_by_counts", "pct_counts_mt")
               if c in obs.columns]
    if not qc_cols:
        print("  no QC columns on obs to compare populations")
        return
    print("\n  QC profile: declared NTC cells vs guide-unassigned cells")
    grp = np.where(is_ntc & assigned, "NTC (assigned)",
                   np.where(assigned, "targeting", "unassigned"))
    print(obs[qc_cols].groupby(pd.Series(grp, index=obs.index)).median().round(1).to_string())
    print("\n  If 'unassigned' sits close to 'NTC (assigned)' on these, it is a "
          "defensible fallback control pool. If it is markedly lower-depth, it "
          "is mostly capture failure and using it will bias every comparison "
          "in that family.")


# ===========================================================================
def section_e_conflict(adata_sub, ga) -> None:
    banner("E. IS THE FAMILY-CONFLICT RATE REAL?")
    obs = adata_sub.obs
    pc = ga.per_cell
    hcol = next((c for c in obs.columns if "hto_family" in c), None)
    if hcol is None:
        print("  hto_family not present on obs. Re-run this after a pipeline "
              "run, or pass the pipeline's per-cell table. Skipping.")
        return
    g = pc["family"].astype(str)
    h = obs[hcol].astype(str)
    both = (g != "None") & (h.str.len() > 0)
    conflict = both & (g != h)
    print(f"  cells with both labels : {int(both.sum()):,}")
    print(f"  conflicting            : {int(conflict.sum()):,} "
          f"({100*conflict.sum()/max(both.sum(),1):.1f}%)")
    if "total_counts" in obs.columns:
        dec = pd.qcut(obs.loc[both, "total_counts"], 10, labels=False,
                      duplicates="drop")
        print("\n  conflict rate by total-counts decile:")
        print(conflict[both].groupby(dec).mean().round(3).to_string())
        print("\n  A conflict rate that rises steeply in the LOW deciles means "
              "the conflicts are low-quality droplets, not index hops.")


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("--grna-whitelist", default=None)
    ap.add_argument("--hashtag-whitelist", default=None)
    ap.add_argument("--n-cells", type=int, default=25000,
                    help="subsample for the GEX-dependent sections")
    ap.add_argument("--full", action="store_true",
                    help="use every cell (needs a lot of RAM)")
    args = ap.parse_args()

    import anndata as ad

    print(f"reading {args.h5ad} ...")
    backed = ad.read_h5ad(args.h5ad, backed="r")
    n_obs = backed.n_obs
    print(f"  {n_obs:,} cells x {backed.n_vars:,} vars")

    guide_wl = load_guide_whitelist(args.grna_whitelist) if args.grna_whitelist else None
    hto_wl = None
    if args.hashtag_whitelist:
        hcols = [c for c in backed.obs.columns
                 if re.match(r"(prot:)?hash\.", c, re.I)]
        hto_wl = load_hashtag_whitelist(args.hashtag_whitelist,
                                        known_hashtags=hcols or None)

    # Sections C and E need only obs, which is cheap on the full object.
    section_c_hashtags(backed, hto_wl)

    # Sections A, D need the matrix. Subsample unless --full.
    if args.full or n_obs <= args.n_cells:
        sub = backed.to_memory()
    else:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n_obs, size=args.n_cells, replace=False))
        print(f"\n  loading a {args.n_cells:,}-cell subsample for the "
              f"matrix sections (use --full for all cells)")
        sub = backed[idx].to_memory()

    cfg_mod = ModalityConfig()
    greg = GuideConfig().guide_id_regexes
    section_a_alignment(sub, cfg_mod, greg, guide_wl)

    split = split_modalities(sub, cfg_mod, greg)
    if split.guide.present:
        ga = call_guides(split.guide, GuideConfig(), whitelist=guide_wl)
        section_d_fallback(sub, split, ga)
        section_e_conflict(sub, ga)
    else:
        print("\n  no guide matrix found; skipping D and E")

    banner("DONE")
    print("  Paste the whole output back. Section A decides whether the "
          "perturbation numbers in the last report mean anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
