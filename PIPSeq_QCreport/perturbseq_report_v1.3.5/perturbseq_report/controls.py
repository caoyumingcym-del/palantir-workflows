"""
Control-pool construction and diagnostics.

Three things live here, all answering "what exactly are we comparing against?".

**1. Depth-matched fallback controls (B1).** Some libraries ship without
non-targeting guides -- RPE1 in the MDL1898 experiment is the case that forced
this. A family with no NTCs currently has no control pool at all, so every
perturbation in it is unquantifiable. The agreed fallback is guide-unassigned
cells from the same family, and the reason it cannot simply be dropped in is
depth: on real data the unassigned cells run at roughly half the depth of the
assigned ones (median 2,674 vs 5,446 UMIs). Substituting them raw would put a
systematic depth difference into every fold change in the family, which reads
as a global transcriptional effect and is not one.

So the pool is depth-matched by stratified sampling before use, and the achieved
match is reported alongside every result that rests on it. Where the match
cannot be made, that is stated rather than papered over -- an honest "no control
pool" beats a fabricated one.

**2. Control-pool composition (B2).** "NTC" is a pool, not a reagent. A family
whose pool is one guide (family C: 1 guide, 974 cells) is being compared against
a single reagent's off-target profile, and the report should say so, because it
changes how every number in that family should be read.

**3. Leave-one-out consistency (B4).** The question raised directly: are these
controls interchangeable? Hold each control guide out in turn, recompute the
pool mean, and report the largest shift. A guide whose removal moves the pool is
not exchangeable with the others, whatever its name says.

Nothing here removes cells or alters counts. Everything is a selection of row
indices plus a table describing it.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


# ===========================================================================
# B1 -- depth matching
# ===========================================================================
def depth_matched_indices(
    target_depth: Sequence[float] | np.ndarray,
    pool_depth: Sequence[float] | np.ndarray,
    n_bins: int = 20,
    max_per_target: float = 3.0,
    random_state: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pick pool cells whose depth distribution matches the target's.

    Stratified on quantile bins of ``log1p(depth)`` rather than matched
    one-to-one: quantile bins put the resolution where the cells are, and
    sampling within a bin keeps the selection unbiased with respect to
    everything except depth. Matching in log space matters because sequencing
    depth is roughly log-normal, so equal-width bins on the raw scale would
    spend most of their resolution on a tail holding few cells.

    Sampling is WITHOUT replacement. A control pool containing the same cell
    several times would understate its own variance, which is worse than a
    smaller pool: the whole point of the pool is to estimate the spread of
    unperturbed expression.

    Parameters
    ----------
    target_depth
        Per-cell depth of the cells that need a control (the guide-positive
        cells of the family).
    pool_depth
        Per-cell depth of the candidate controls (guide-unassigned cells of the
        same family).
    n_bins
        Quantile bins of the target depth distribution.
    max_per_target
        Cap on pool cells per target cell. More controls than this buys almost
        nothing statistically and costs runtime in every downstream test.
    random_state
        Seed, so a re-run of the same input selects the same cells. Without
        this the control pool -- and therefore every fold change -- would move
        slightly between runs of identical data.

    Returns
    -------
    (indices, info)
        ``indices`` are positions into ``pool_depth``. ``info`` records what was
        achieved, including ``poor_match`` when the pool cannot cover the
        target's depth range.
    """
    t = np.asarray(target_depth, dtype=float)
    p = np.asarray(pool_depth, dtype=float)
    t = t[np.isfinite(t) & (t > 0)]
    empty = np.array([], dtype=np.intp)

    info: dict[str, Any] = {
        "n_target": int(t.size),
        "n_pool": int(np.isfinite(p).sum()),
        "n_selected": 0,
        "median_target": float(np.median(t)) if t.size else float("nan"),
        "median_pool_before": (
            float(np.median(p[np.isfinite(p)])) if np.isfinite(p).any()
            else float("nan")
        ),
        "median_pool_after": float("nan"),
        "median_ratio_before": float("nan"),
        "median_ratio_after": float("nan"),
        "poor_match": False,
        "reason": "",
        # Which target cells have controls at their depth. Always present, so a
        # caller never has to branch on whether matching got far enough to set
        # it.
        "target_mask": np.zeros(t.size, dtype=bool),
        "n_target_unmatched": int(t.size),
        "frac_target_unmatched": 1.0 if t.size else 0.0,
    }
    if t.size == 0 or p.size == 0 or not np.isfinite(p).any():
        info["reason"] = "no target cells or no pool cells"
        info["poor_match"] = True
        return empty, info

    finite_pool = np.flatnonzero(np.isfinite(p) & (p > 0))
    if finite_pool.size == 0:
        info["reason"] = "no pool cell had a usable depth"
        info["poor_match"] = True
        return empty, info

    if info["median_pool_before"] and np.isfinite(info["median_pool_before"]):
        info["median_ratio_before"] = (
            info["median_pool_before"] / info["median_target"]
            if info["median_target"] else float("nan")
        )

    lt = np.log1p(t)
    lp = np.log1p(p[finite_pool])

    # Quantile edges of the TARGET distribution: the target is what we are
    # matching to, so it defines the strata.
    n_bins = max(int(n_bins), 1)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(lt, qs))
    if edges.size < 2:
        edges = np.array([lt.min() - 1e-9, lt.max() + 1e-9])
    # Open the outer edges so pool cells beyond the target's observed range
    # still land in the end bins rather than being dropped.
    edges[0] = -np.inf
    edges[-1] = np.inf

    t_bin = np.digitize(lt, edges[1:-1], right=False)
    p_bin = np.digitize(lp, edges[1:-1], right=False)

    rng = np.random.default_rng(random_state)
    n_bins_effective = int(edges.size - 1)

    # --- Step 1: common support -------------------------------------------
    # A stratum the pool cannot supply at all is not a sampling shortfall, it
    # is a region of depth where no control exists. Taking "as many as
    # available" from every bin -- which is what a naive implementation does --
    # silently reshapes the pool towards wherever controls happen to be
    # plentiful, i.e. towards shallow cells, which is the exact bias this
    # function exists to remove. Those target cells are excluded from the
    # comparison instead, and the exclusion is reported.
    want_by_bin = {b: int(np.sum(t_bin == b)) for b in range(n_bins_effective)}
    avail_by_bin = {
        b: finite_pool[p_bin == b] for b in range(n_bins_effective)
    }
    supported = [
        b for b, w in want_by_bin.items()
        if w > 0 and avail_by_bin[b].size > 0
    ]
    if not supported:
        info["reason"] = (
            "no pool cell fell in any depth stratum occupied by the target "
            "cells; the two populations do not overlap in depth at all"
        )
        info["poor_match"] = True
        info["target_mask"] = np.zeros(t.size, dtype=bool)
        return empty, info

    target_mask = np.isin(t_bin, supported)
    n_unmatched = int((~target_mask).sum())

    # --- Step 2: one common ratio across strata ---------------------------
    # Sampling r * want_b from every supported stratum reproduces the target's
    # depth distribution exactly (up to rounding). r is set by the scarcest
    # stratum, ignoring strata that hold a negligible slice of the target so
    # one thin tail bin cannot collapse the whole pool.
    n_supported_target = int(target_mask.sum())
    ratios = []
    for b in supported:
        share = want_by_bin[b] / max(n_supported_target, 1)
        if share < 0.01:
            continue
        ratios.append(avail_by_bin[b].size / float(want_by_bin[b]))
    r = min(ratios) if ratios else float(max_per_target)
    r = float(min(r, float(max_per_target)))

    chosen: list[np.ndarray] = []
    for b in supported:
        take = int(np.floor(r * want_by_bin[b]))
        take = max(take, 1) if want_by_bin[b] > 0 else 0
        take = min(take, avail_by_bin[b].size)
        if take > 0:
            chosen.append(rng.choice(avail_by_bin[b], size=take, replace=False))

    if not chosen:
        info["reason"] = "no pool cell could be drawn from a supported stratum"
        info["poor_match"] = True
        info["target_mask"] = np.zeros(t.size, dtype=bool)
        return empty, info

    idx = np.unique(np.concatenate(chosen))
    sel = p[idx]
    matched_target = t[target_mask]
    info["n_selected"] = int(idx.size)
    info["median_pool_after"] = float(np.median(sel))
    info["median_target_matched"] = (
        float(np.median(matched_target)) if matched_target.size else float("nan")
    )
    info["median_ratio_after"] = (
        info["median_pool_after"] / info["median_target_matched"]
        if info["median_target_matched"] else float("nan")
    )
    info["ratio_per_target"] = r
    info["n_target_unmatched"] = n_unmatched
    info["frac_target_unmatched"] = n_unmatched / float(max(t.size, 1))
    info["target_mask"] = target_mask

    # Three independent ways the match can be bad, all worth flagging.
    ratio = info["median_ratio_after"]
    if not np.isfinite(ratio) or not (0.9 <= ratio <= 1.111):
        info["poor_match"] = True
        info["reason"] = (
            f"even on the depth range the pool covers, its median depth is "
            f"{ratio:.2f}x the target's"
        )
    elif info["frac_target_unmatched"] > 0.25:
        info["poor_match"] = True
        info["reason"] = (
            f"{100.0 * info['frac_target_unmatched']:.0f}% of the "
            f"guide-positive cells are deeper than any available control, so "
            f"they have no comparison and were excluded"
        )
    elif idx.size < max(0.5 * n_supported_target, 25):
        info["poor_match"] = True
        info["reason"] = (
            f"only {idx.size:,} pool cells could be matched to "
            f"{n_supported_target:,} target cells, which is too few to "
            f"estimate control variance"
        )
    return idx, info


def describe_depth_match(family: str, info: dict[str, Any]) -> str:
    """One-paragraph report-ready description of a fallback control pool."""
    if not info.get("n_selected"):
        return (
            f"Family {family}: no usable fallback control pool. "
            f"{info.get('reason', 'no reason recorded')}. Perturbations in this "
            f"family cannot be quantified, and are reported as skipped rather "
            f"than compared against an unmatched pool."
        )
    verdict = (
        "The match is poor and these results should be treated as indicative "
        "only: " + str(info.get("reason", ""))
        if info.get("poor_match") else
        "The match is acceptable."
    )
    unmatched = ""
    if info.get("n_target_unmatched"):
        unmatched = (
            f" {int(info['n_target_unmatched']):,} guide-positive cells "
            f"({100.0 * float(info['frac_target_unmatched']):.0f}%) were deeper "
            f"than any available control and are excluded from this family's "
            f"comparisons rather than matched to a shallower cell."
        )
    return (
        f"Family {family} has no non-targeting guides, so its control pool was "
        f"built from {info['n_selected']:,} guide-unassigned cells of the same "
        f"family, depth-matched to the {info['n_target']:,} guide-positive "
        f"cells. Unassigned cells are systematically shallower than assigned "
        f"ones -- here {info['median_ratio_before']:.2f}x the target's median "
        f"depth before matching, {info['median_ratio_after']:.2f}x after, on "
        f"the depth range both populations share -- and using them unmatched "
        f"would put that depth difference into every fold change in the "
        f"family.{unmatched} {verdict} One caveat that matching cannot fix: "
        f"cells that failed guide assignment are not verified unperturbed, "
        f"since some carry a guide that was simply not called. Treat this "
        f"family's effect sizes as conservative."
    )


# ===========================================================================
# B2 -- pool composition
# ===========================================================================
def summarise_control_pool(
    guides_by_family: dict[str, pd.Series],
) -> pd.DataFrame:
    """Per-family control-pool composition, one row per family.

    ``single_guide`` is the flag that matters: a pool resting on one guide is a
    comparison against one reagent, and its off-target effects are then
    indistinguishable from the biology of every perturbation in that family.
    ``top_guide_frac`` catches the softer version of the same problem -- three
    guides where one supplies 90% of the cells is closer to a single-guide pool
    than the count suggests.
    """
    rows: list[dict[str, Any]] = []
    for fam, guides in guides_by_family.items():
        g = pd.Series(guides).astype(str)
        g = g[g.notna() & (g != "")]
        vc = g.value_counts()
        n_guides = int(vc.size)
        top_frac = float(vc.iloc[0] / vc.sum()) if n_guides else float("nan")
        rows.append({
            "family": str(fam),
            "n_control_cells": int(g.size),
            "n_guides": n_guides,
            "top_guide": str(vc.index[0]) if n_guides else "",
            "top_guide_frac": top_frac,
            "single_guide": bool(n_guides == 1),
            "concentrated": bool(n_guides > 1 and top_frac >= 0.80),
            "min_cells_per_guide": int(vc.min()) if n_guides else 0,
        })
    return pd.DataFrame(rows)


def control_pool_warnings(summary: pd.DataFrame) -> list[str]:
    """Report-ready warnings from ``summarise_control_pool``."""
    out: list[str] = []
    if summary.empty:
        return out
    single = summary[summary["single_guide"]]
    for _, r in single.iterrows():
        out.append(
            f"Family {r['family']}'s control pool is a SINGLE guide "
            f"({r['top_guide']}, {int(r['n_control_cells']):,} cells). Every "
            f"fold change in this family is therefore measured against one "
            f"reagent, so that guide's own off-target profile is "
            f"indistinguishable from the biology of the perturbations being "
            f"compared to it. Treat effect sizes here as less reliable than in "
            f"families with several controls, and do not compare them directly "
            f"across families."
        )
    conc = summary[summary["concentrated"]]
    for _, r in conc.iterrows():
        out.append(
            f"Family {r['family']}'s control pool nominally has "
            f"{int(r['n_guides'])} guides, but {r['top_guide']} supplies "
            f"{100.0 * float(r['top_guide_frac']):.0f}% of its cells, so it "
            f"behaves closer to a single-guide pool than the count suggests."
        )
    return out


# ===========================================================================
# B4 -- leave-one-out consistency
# ===========================================================================
def leave_one_out_consistency(
    X: Any,
    guides: pd.Series,
    max_genes: int = 2000,
) -> pd.DataFrame:
    """Hold each control guide out and measure how far the pool mean moves.

    Directly addresses "they are still individual guides, why are they being
    treated as interchangeable?". If the control pool is genuinely exchangeable,
    dropping any one guide should barely move its mean expression profile. A
    guide whose removal shifts the pool is contributing something of its own,
    and pooling it flattens that into the baseline every perturbation is
    measured against.

    ``max_abs_shift`` is the largest absolute change in any gene's pool mean,
    in the units of ``X`` (log-normalised, so roughly log2-fold). Also reported
    is ``mean_abs_shift``, which is less sensitive to one aberrant gene.

    Genes are capped at ``max_genes`` (highest-variance first) because this is a
    diagnostic, not a test: the ranking of guides is stable long before every
    gene is included, and the full matrix would be the expensive part.
    """
    g = pd.Series(guides).astype(str).to_numpy()
    A = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=float)
    if A.ndim != 2 or A.shape[0] != g.size or A.shape[0] == 0:
        return pd.DataFrame(columns=[
            "guide", "n_cells", "max_abs_shift", "mean_abs_shift",
        ])

    if A.shape[1] > max_genes:
        keep = np.argsort(A.var(axis=0))[::-1][:max_genes]
        A = A[:, np.sort(keep)]

    levels = [str(x) for x in pd.unique(g)]
    if len(levels) < 2:
        return pd.DataFrame([{
            "guide": levels[0] if levels else "",
            "n_cells": int(g.size),
            "max_abs_shift": float("nan"),
            "mean_abs_shift": float("nan"),
            "note": "only one control guide -- nothing to leave out",
        }])

    n = A.shape[0]
    total = A.sum(axis=0)
    full_mean = total / n

    rows: list[dict[str, Any]] = []
    for lvl in levels:
        m = g == lvl
        k = int(m.sum())
        if k == 0 or k >= n:
            continue
        # Pool mean without this guide, from the running total: no need to
        # re-sum the retained rows for every level.
        loo_mean = (total - A[m].sum(axis=0)) / float(n - k)
        d = np.abs(loo_mean - full_mean)
        rows.append({
            "guide": lvl,
            "n_cells": k,
            "frac_of_pool": k / float(n),
            "max_abs_shift": float(np.max(d)),
            "mean_abs_shift": float(np.mean(d)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("max_abs_shift", ascending=False).reset_index(
            drop=True
        )
    return out


def consistency_warnings(
    tab: pd.DataFrame, ratio_warn: float = 2.0
) -> list[str]:
    """Flag control guides that move the pool far more than their peers."""
    if tab.empty or "max_abs_shift" not in tab.columns or len(tab) < 3:
        return []
    s = tab["max_abs_shift"].astype(float)
    med = float(np.nanmedian(s))
    if not np.isfinite(med) or med <= 0:
        return []
    bad = tab[s > ratio_warn * med]
    out: list[str] = []
    for _, r in bad.iterrows():
        out.append(
            f"Control guide {r['guide']} shifts the pooled control mean "
            f"{float(r['max_abs_shift']) / med:.1f}x more than the median "
            f"control guide does when it is held out "
            f"({int(r['n_cells']):,} cells). It is not exchangeable with the "
            f"others, so pooling it moves the baseline that every perturbation "
            f"in this family is measured against. Worth checking whether it is "
            f"a mislabelled targeting guide before trusting the family's "
            f"effect sizes."
        )
    return out
