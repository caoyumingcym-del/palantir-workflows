"""
Perturbation-effect quantification.

This module is where the collaborator's analyses that the original pipeline
lacked have been implemented:

============================  ==============================================
Collaborator's section        Implemented here as
============================  ==============================================
Pseudobulk %KD                ``knockdown_table`` (+ NTC baseline expression)
SCEPTRE target-gene log2FC    ``stats.resampling_test`` calibrated null
Mixscale target-gene expr     ``perturbation_scores`` + score-ranked heatmap
Transcriptome-wide log2FC     ``plot_transcriptome_log2fc``
Top-10 DEG dotplot            ``deg_dotplot`` (own gene first, shared flagged)
Perturbation similarity       ``similarity_matrices`` (Jaccard / Spearman)
Guide-pair analysis           ``guide_pair_knockdown``
============================  ==============================================

What the original had that is kept: target-gene knockdown against NTC on the
linear scale (it correctly inverted log1p before averaging) and E-distance in
PCA space.  Both are extended -- knockdown gains a calibrated p-value and an
explicit "target undetectable in control" state, and E-distance gains a
permutation p-value, without which values from perturbations with very
different cell counts are not comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import controls as CTRL
from . import plotting as P
from . import text as T
from .artifacts import Registry
from .config import FigureConfig, PerturbConfig, PipelineConfig
from .stats import (
    benjamini_hochberg, differential_expression, edistance_permutation_pvalue,
    jaccard_matrix, percent_knockdown, perturbation_score, rank_degs,
    resampling_test, select_degs, significance_stars, spearman_matrix,
    row_sums, take_column, take_columns, take_rows, to_csc, to_csr,
)


@dataclass
class PerturbResults:
    knockdown: pd.DataFrame          # per target: %KD, log2FC, p, n cells
    edistance: pd.DataFrame          # per target: E-distance, permutation p
    de: dict[str, pd.DataFrame]      # target -> full DE table
    degs: dict[str, pd.DataFrame]    # target -> top-N DEG table
    similarity: dict[str, pd.DataFrame] = field(default_factory=dict)
    scores: pd.DataFrame | None = None
    guide_pairs: pd.DataFrame | None = None
    excluded: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)


# ===========================================================================
# Gene lookup
# ===========================================================================
def build_gene_index(var_names: Sequence[str]) -> dict[str, int]:
    """Map gene symbol and version-stripped Ensembl ID to column index.

    Ensembl IDs in h5ads frequently carry a version suffix
    (``ENSG00000141510.17``), and the guide's declared target does not.  The
    original stripped versions in one helper but not everywhere it mattered.
    """
    index: dict[str, int] = {}
    for i, raw in enumerate(var_names):
        name = str(raw)
        index.setdefault(name, i)
        index.setdefault(name.upper(), i)
        if "." in name and name.upper().startswith("ENS"):
            index.setdefault(name.split(".")[0], i)
            index.setdefault(name.split(".")[0].upper(), i)
    return index


def resolve_gene(
    index: dict[str, int], symbol: str | None, ensg: str | None = None
) -> int | None:
    for key in (ensg, symbol):
        if not key:
            continue
        k = str(key)
        for candidate in (k, k.upper(), k.split(".")[0], k.split(".")[0].upper()):
            if candidate in index:
                return index[candidate]
    return None


# ===========================================================================
# Core quantification
# ===========================================================================
@dataclass
class TargetAnnotations:
    """Lookups from ``target_key`` to everything else about that group.

    ``target_key`` is ``"{gene}_{family}"`` -- e.g. ``ABT1_A``, ``NTC_A``,
    ``NTC_B``. Grouping on it rather than on the gene symbol is what keeps four
    guide populations from sharing one control pool, and what stops two
    libraries that happen to use the same gene symbol from being merged into a
    single perturbation.
    """

    gene_by_key: dict[str, str]
    ensg_by_key: dict[str, str | None]
    family_by_key: dict[str, str | None]
    ntc_key_by_family: dict[str | None, str]
    ntc_keys: set[str]

    @property
    def n_families(self) -> int:
        return len({f for f in self.family_by_key.values() if f is not None})


def target_annotations(
    guide_mapping: pd.DataFrame | None,
    ntc_label: str,
    pool_across_families: bool = False,
) -> TargetAnnotations:
    """Build the target_key lookups from the guide mapping.

    Falls back to treating target keys as bare gene names when no mapping is
    supplied, so callers that predate families keep working.

    ``pool_across_families`` mirrors ``GuideConfig.pool_ntc_across_families``
    and must be passed explicitly -- see the comment on the branch below for why
    inferring it was a bug.
    """
    if guide_mapping is None or "target_key" not in guide_mapping.columns:
        return TargetAnnotations(
            gene_by_key={}, ensg_by_key={}, family_by_key={},
            ntc_key_by_family={None: ntc_label}, ntc_keys={ntc_label},
        )

    m = guide_mapping.drop_duplicates(subset=["target_key"])
    gene_by_key = dict(zip(m["target_key"], m["target_gene"]))
    family_by_key = dict(zip(m["target_key"], m.get("family", pd.Series(dtype=object))))

    # An ENSG is only useful if every guide for the target agrees on it.
    ensg_by_key: dict[str, str | None] = {}
    if "target_ensg" in guide_mapping.columns:
        for key, sub in guide_mapping.groupby("target_key"):
            vals = {v for v in sub["target_ensg"].dropna() if str(v)}
            ensg_by_key[str(key)] = vals.pop() if len(vals) == 1 else None

    ntc_keys = set(
        guide_mapping.loc[guide_mapping["is_ntc"], "target_key"].astype(str)
    )
    ntc_key_by_family: dict[str | None, str] = {}
    for key in ntc_keys:
        fam = family_by_key.get(key)
        ntc_key_by_family[fam] = key

    # Cross-family control borrowing happens ONLY when it was explicitly asked
    # for.
    #
    # Up to v1.3.0 this branch was `if len(ntc_keys) == 1:` and inferred the
    # intent instead of being told it. That inference is lossy in the one case
    # that matters: a single NTC key means "pooling was requested" when
    # pool_ntc_across_families=True (guide.target_key collapses every control to
    # cfg.ntc_label), but it is ALSO true when pooling was not requested and
    # only one family happens to carry controls. In that second case every
    # family without its own NTCs silently borrowed the one family that had
    # them -- which is precisely the mixed experiment the B1 fallback exists
    # for, so the fallback could never fire and RPE1 guides would have been
    # compared against another cell line's controls with nothing saying so.
    if pool_across_families and len(ntc_keys) == 1:
        only = next(iter(ntc_keys))
        for fam in set(family_by_key.values()):
            ntc_key_by_family.setdefault(fam, only)

    return TargetAnnotations(
        gene_by_key=gene_by_key, ensg_by_key=ensg_by_key,
        family_by_key=family_by_key, ntc_key_by_family=ntc_key_by_family,
        ntc_keys=ntc_keys,
    )


def knockdown_table(
    X_log: np.ndarray,
    var_names: Sequence[str],
    target_by_cell: pd.Series,
    ntc_label: str,
    cfg: PerturbConfig,
    guide_mapping: pd.DataFrame | None = None,
    fallback_pool: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    fallback_out: dict[str, Any] | None = None,
    pool_across_families: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-target knockdown of the target gene versus NTC cells.

    Returns ``(knockdown, excluded)``.  Targets are excluded -- and listed, not
    silently dropped -- when they have too few cells or when the target gene is
    absent from the matrix.  A target that is present but undetectable in
    control cells gets a row with NaN knockdown and an explicit reason, because
    reporting 0% or 100% there would be an invention.

    ``fallback_pool`` is a boolean mask of guide-UNASSIGNED cells, and ``depth``
    their per-cell total counts. Where a family has no non-targeting guides, a
    depth-matched subset of the same family's unassigned cells is used as its
    control pool (B1). Rows built this way are marked ``control_is_fallback``,
    and ``fallback_out`` (if given) receives the per-family matching diagnostics
    so the caller can report them. Without both arguments the behaviour is
    exactly as before: such families are excluded with a reason.
    """
    gene_index = build_gene_index(var_names)
    ann = target_annotations(guide_mapping, ntc_label, pool_across_families)

    # Controls are FAMILY-SCOPED. An NTC guide from one library is not a
    # control for a guide from another: different genetic background, different
    # library prep, often a different cell line. v1.1.0 pooled every NTC in the
    # object into one baseline, which on MDL-1856 meant 60 NTC guides spanning
    # multiple populations backing every knockdown call.
    ntc_masks, n_ntc_by_family = {}, {}
    for fam, key in ann.ntc_key_by_family.items():
        m = (target_by_cell == key).to_numpy()
        ntc_masks[fam] = m
        n_ntc_by_family[fam] = int(m.sum())

    # ------------------------------------------- fallback controls (B1)
    # A family with no NTC guides has no control pool, so every perturbation in
    # it is unquantifiable. Fall back to guide-unassigned cells OF THE SAME
    # FAMILY, depth-matched -- unassigned cells run systematically shallower
    # (median 2,674 vs 5,446 UMIs on real data), and substituting them raw puts
    # that depth difference into every fold change in the family.
    fallback_info: dict[str, dict[str, Any]] = {}
    if fallback_pool is not None and depth is not None:
        d = np.asarray(depth, dtype=float)
        for fam in sorted(set(ann.family_by_key.values()) - {None}):
            if n_ntc_by_family.get(fam, 0) >= cfg.min_cells_per_group:
                continue
            pool_mask = np.asarray(fallback_pool, dtype=bool).copy()
            fam_mask = np.array([
                ann.family_by_key.get(k) == fam
                for k in target_by_cell.to_numpy()
            ])
            # Target = this family's guide-positive, non-control cells.
            tgt_mask = fam_mask & ~pool_mask
            if not tgt_mask.any() or not pool_mask.any():
                continue
            pool_idx_all = np.flatnonzero(pool_mask)
            sel_local, info = CTRL.depth_matched_indices(
                d[tgt_mask], d[pool_idx_all],
                random_state=cfg.resample_random_state,
            )
            info.pop("target_mask", None)
            fallback_info[str(fam)] = info
            if sel_local.size == 0:
                continue
            m = np.zeros(len(target_by_cell), dtype=bool)
            m[pool_idx_all[sel_local]] = True
            ntc_masks[fam] = m
            n_ntc_by_family[fam] = int(m.sum())

    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    if not n_ntc_by_family or max(n_ntc_by_family.values(), default=0) < cfg.min_cells_per_group:
        best = max(n_ntc_by_family.values(), default=0)
        return (
            pd.DataFrame(),
            pd.DataFrame(
                [
                    {
                        "target_key": "(all)",
                        "target_gene": "(all)",
                        "reason": (
                            f"no family has at least {cfg.min_cells_per_group} "
                            f"control ({ntc_label}) cells (largest control group "
                            f"has {best}); no perturbation can be quantified "
                            f"without a control group"
                        ),
                        "n_cells": best,
                    }
                ]
            ),
        )

    for target in sorted(set(target_by_cell.dropna()) - set(ann.ntc_keys)):
        mask = (target_by_cell == target).to_numpy()
        n = int(mask.sum())
        gene = ann.gene_by_key.get(target, target)
        family = ann.family_by_key.get(target)
        if n < cfg.min_cells_per_group:
            excluded.append(
                {"target_key": target, "target_gene": gene, "family": family,
                 "reason": "too few cells", "n_cells": n}
            )
            continue

        ntc_mask = ntc_masks.get(family)
        n_ntc = n_ntc_by_family.get(family, 0)
        if ntc_mask is None or n_ntc < cfg.min_cells_per_group:
            excluded.append(
                {
                    "target_key": target, "target_gene": gene, "family": family,
                    "reason": (
                        f"family {family!r} has only {n_ntc} control cells "
                        f"(minimum {cfg.min_cells_per_group}). Controls are "
                        f"family-scoped, so this target cannot borrow another "
                        f"family's NTCs"
                    ),
                    "n_cells": n,
                }
            )
            continue

        gi = resolve_gene(gene_index, gene, ann.ensg_by_key.get(target))
        if gi is None:
            excluded.append(
                {
                    "target_key": target, "target_gene": gene, "family": family,
                    "reason": "target gene not present in the expression matrix",
                    "n_cells": n,
                }
            )
            continue

        # One gene column, densified once and indexed -- never the matrix.
        gene_col = take_column(X_log, gi)
        pert_expr = gene_col[mask]
        ntc_expr = gene_col[ntc_mask]
        kd = percent_knockdown(pert_expr, ntc_expr, log_input=True)
        rs = resampling_test(
            pert_expr, ntc_expr, cfg.resample_n, cfg.resample_random_state,
            log_input=True,
        )
        frac_detected_ntc = float((ntc_expr > 0).mean())
        rows.append(
            {
                "target_key": target,
                "target_gene": gene,
                "family": family,
                "n_cells": n,
                "n_control_cells": n_ntc,
                "pct_knockdown": kd["pct_knockdown"],
                "log2fc": kd["log2fc"],
                "mean_perturbed": kd["mean_perturbed"],
                "mean_control": kd["mean_control"],
                "frac_detected_control": frac_detected_ntc,
                "pvalue_rank": kd["pvalue"],
                "pvalue_resample": rs["p_resample"],
                "z_resample": rs["z_resample"],
                "target_detectable": frac_detected_ntc
                >= cfg.de_min_frac_detected_in_ntc,
                # True where the control pool is depth-matched unassigned cells
                # rather than real non-targeting guides. Carried on every row so
                # a reader cannot pick a number out of the table without it.
                "control_is_fallback": str(family) in fallback_info,
            }
        )

    if fallback_out is not None:
        fallback_out.update(fallback_info)

    kdf = pd.DataFrame(rows)
    if not kdf.empty:
        kdf["padj_resample"] = benjamini_hochberg(kdf["pvalue_resample"].to_numpy())
        kdf["significance"] = kdf["padj_resample"].map(significance_stars)
        kdf = kdf.sort_values("pct_knockdown", ascending=False)
    return kdf, pd.DataFrame(excluded)


def per_condition_knockdown(
    X_log: np.ndarray,
    var_names: Sequence[str],
    target_by_cell: pd.Series,
    condition: pd.Series,
    ntc_label: str,
    cfg: PerturbConfig,
    guide_mapping: pd.DataFrame | None = None,
    max_levels: int = 6,
    pool_across_families: bool = False,
) -> pd.DataFrame:
    """Target-gene knockdown computed WITHIN each condition level (B5).

    The pooled table answers "did this perturbation work?". This answers "did it
    work in every arm?", which is a different question and the one that matters
    when the experiment is a comparison. A knockdown that is 80% in one fixation
    arm and 5% in the other reads as ~45% when pooled -- a number that describes
    neither arm and looks unremarkable.

    Crucially the CONTROLS are taken from the same condition level too, not just
    the perturbed cells. Comparing a condition's perturbed cells against pooled
    controls from every condition would fold the condition effect into every
    knockdown estimate, which is the mistake this function exists to avoid.
    Controls remain family-scoped as well, so a comparison is always
    within-family and within-condition.

    Levels are capped at ``max_levels`` (largest first) to bound the work: each
    level is a full pass of ``knockdown_table``.
    """
    cond = pd.Series(condition).astype(str)
    tgt = pd.Series(target_by_cell)
    if len(cond) != len(tgt):
        return pd.DataFrame()

    cond.index = tgt.index
    counts = cond.value_counts()
    levels = [str(x) for x in counts.index[:max_levels]]

    frames: list[pd.DataFrame] = []
    for lvl in levels:
        m = (cond == lvl).to_numpy()
        if int(m.sum()) < cfg.min_cells_per_group:
            continue
        sub_targets = tgt[m]
        sub_X = take_rows(X_log, m)
        kd, _excluded = knockdown_table(
            sub_X, var_names, sub_targets, ntc_label, cfg, guide_mapping,
            pool_across_families=pool_across_families,
        )
        if kd.empty:
            continue
        kd = kd.copy()
        kd.insert(0, "condition", lvl)
        kd["n_cells_in_condition"] = int(m.sum())
        frames.append(kd)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    label = _label_col(out)
    return out.sort_values([label, "condition"]).reset_index(drop=True)


def condition_effect_spread(per_cond: pd.DataFrame) -> pd.DataFrame:
    """Per-target spread of knockdown across condition levels.

    One row per target seen in two or more levels. ``range_pp`` is the gap in
    percentage points between its best and worst arm -- the number that says
    whether pooling the conditions was hiding something.
    """
    if per_cond.empty or "condition" not in per_cond.columns:
        return pd.DataFrame()
    label = _label_col(per_cond)
    rows: list[dict[str, Any]] = []
    for key, grp in per_cond.groupby(label):
        v = grp["pct_knockdown"].astype(float)
        if v.notna().sum() < 2:
            continue
        best = grp.loc[v.idxmax()]
        worst = grp.loc[v.idxmin()]
        rows.append({
            label: key,
            "n_levels": int(v.notna().sum()),
            "best_condition": str(best["condition"]),
            "best_pct_knockdown": float(best["pct_knockdown"]),
            "worst_condition": str(worst["condition"]),
            "worst_pct_knockdown": float(worst["pct_knockdown"]),
            "range_pp": float(v.max() - v.min()),
            "mean_pct_knockdown": float(v.mean()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("range_pp", ascending=False).reset_index(drop=True)
    return out


def condition_spread_notes(
    spread: pd.DataFrame, axis_name: str, warn_pp: float = 25.0
) -> list[str]:
    """Report-ready findings about condition-dependent perturbation effects."""
    if spread.empty:
        return []
    label = _label_col(spread)
    bad = spread[spread["range_pp"] >= warn_pp]
    if bad.empty:
        w = spread.iloc[0]
        return [
            f"No perturbation's knockdown differs by more than "
            f"{warn_pp:.0f} percentage points across {axis_name}. The largest "
            f"gap is {w[label]} at {float(w['range_pp']):.0f} pp "
            f"({w['best_condition']} {float(w['best_pct_knockdown']):.0f}% vs "
            f"{w['worst_condition']} {float(w['worst_pct_knockdown']):.0f}%), so "
            f"pooling the conditions for the main analysis is defensible."
        ]
    w = bad.iloc[0]
    return [
        f"{len(bad)} perturbation(s) show knockdown differing by at least "
        f"{warn_pp:.0f} percentage points across {axis_name}. The largest is "
        f"{w[label]}: {float(w['best_pct_knockdown']):.0f}% in "
        f"{w['best_condition']} against {float(w['worst_pct_knockdown']):.0f}% in "
        f"{w['worst_condition']}, a gap of {float(w['range_pp']):.0f} pp. Pooled "
        f"across conditions that averages to "
        f"{float(w['mean_pct_knockdown']):.0f}%, a figure that describes neither "
        f"arm. Either the perturbation genuinely behaves differently between "
        f"these conditions, or one arm's guide calling or capture is worse -- the "
        f"per-condition guide purity panels and the pseudobulk comparability "
        f"section distinguish those two."
    ]


def edistance_table(
    pca: np.ndarray,
    target_by_cell: pd.Series,
    ntc_label: str,
    cfg: PerturbConfig,
    n_perm: int = 200,
    guide_mapping: pd.DataFrame | None = None,
    pool_across_families: bool = False,
) -> pd.DataFrame:
    """E-distance from control, with a label-permutation p-value per target.

    Distances are measured against the target's OWN family's control cells.
    Measuring a family-A perturbation against family-B NTCs mostly recovers the
    distance between the two cell populations, which can dwarf any perturbation
    effect and is not what the number is supposed to mean.
    """
    if pca is None or np.asarray(pca).size == 0:
        return pd.DataFrame()
    A_all = np.asarray(pca, dtype=float)[:, : cfg.edistance_n_pcs]
    ann = target_annotations(guide_mapping, ntc_label, pool_across_families)
    ntc_masks = {
        fam: (target_by_cell == key).to_numpy()
        for fam, key in ann.ntc_key_by_family.items()
    }
    if not any(int(m.sum()) >= cfg.min_cells_per_group for m in ntc_masks.values()):
        return pd.DataFrame()
    rows = []
    for target in sorted(set(target_by_cell.dropna()) - set(ann.ntc_keys)):
        mask = (target_by_cell == target).to_numpy()
        if int(mask.sum()) < cfg.min_cells_per_group:
            continue
        family = ann.family_by_key.get(target)
        ntc_mask = ntc_masks.get(family)
        if ntc_mask is None or int(ntc_mask.sum()) < cfg.min_cells_per_group:
            continue
        ctrl = A_all[ntc_mask]
        e, p = edistance_permutation_pvalue(
            A_all[mask], ctrl, n_perm=n_perm,
            random_state=cfg.resample_random_state,
        )
        rows.append(
            {
                "target_key": target,
                "target_gene": ann.gene_by_key.get(target, target),
                "family": family,
                "n_cells": int(mask.sum()),
                "n_control_cells": int(ntc_mask.sum()),
                "edistance": e, "pvalue_perm": p,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["padj_perm"] = benjamini_hochberg(df["pvalue_perm"].to_numpy())
        df["significance"] = df["padj_perm"].map(significance_stars)
        df = df.sort_values("edistance", ascending=False)
    return df


def differential_expression_per_target(
    X_log: np.ndarray,
    var_names: Sequence[str],
    target_by_cell: pd.Series,
    ntc_label: str,
    cfg: PerturbConfig,
    max_targets: int = 40,
    guide_mapping: pd.DataFrame | None = None,
    pool_across_families: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[str]]:
    """Full DE table and top-N DEG table per target, against family controls."""
    notes: list[str] = []
    ann = target_annotations(guide_mapping, ntc_label, pool_across_families)
    ntc_masks = {
        fam: (target_by_cell == key).to_numpy()
        for fam, key in ann.ntc_key_by_family.items()
    }
    usable = {
        f: m for f, m in ntc_masks.items()
        if int(m.sum()) >= cfg.min_cells_per_group
    }
    if not usable:
        best = max((int(m.sum()) for m in ntc_masks.values()), default=0)
        return {}, {}, [
            f"Differential expression skipped: no family has at least "
            f"{cfg.min_cells_per_group} control cells (largest has {best})."
        ]

    targets = sorted(set(target_by_cell.dropna()) - set(ann.ntc_keys))
    eligible = [
        t for t in targets
        if int((target_by_cell == t).sum()) >= cfg.min_cells_per_group
        and ann.family_by_key.get(t) in usable
    ]
    skipped_no_ctrl = [
        t for t in targets
        if ann.family_by_key.get(t) not in usable
    ]
    if skipped_no_ctrl:
        notes.append(
            f"{len(skipped_no_ctrl)} target(s) were skipped because their "
            f"family has too few control cells. Controls are family-scoped, so "
            f"a target cannot borrow another family's NTCs."
        )
    if len(eligible) > max_targets:
        sizes = {t: int((target_by_cell == t).sum()) for t in eligible}
        eligible = sorted(sizes, key=lambda t: -sizes[t])[:max_targets]
        notes.append(
            f"Differential expression was run for the {max_targets} targets with "
            f"the most cells, out of {len(targets)} present. Raise "
            f"`--max-de-targets` to include more (runtime grows linearly)."
        )

    # One reference matrix per family, built once rather than per target.
    refs = {f: take_rows(X_log, m) for f, m in usable.items()}
    de: dict[str, pd.DataFrame] = {}
    degs: dict[str, pd.DataFrame] = {}
    for target in eligible:
        mask = (target_by_cell == target).to_numpy()
        X_ref = refs[ann.family_by_key.get(target)]
        res = differential_expression(
            take_rows(X_log, mask), X_ref, var_names, log_input=True,
            min_frac_detected_ref=cfg.de_min_frac_detected_in_ntc,
            block=cfg.de_gene_block,
        )
        de[target] = res.table
        sel = select_degs(
            res.table, cfg.de_padj_max, cfg.de_abs_log2fc_min,
            exclude_low_expression=True,
        )
        degs[target] = rank_degs(
            sel, cfg.de_top_n_per_perturbation,
            always_first=ann.gene_by_key.get(target, target),
        )
    return de, degs, notes


def similarity_matrices(
    de: dict[str, pd.DataFrame], degs: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Jaccard of DEG sets and Spearman of log2FC over the union DEG gene set.

    The Spearman correlation uses a *fixed* gene set -- every gene that is a
    DEG in any perturbation -- so values are comparable across all pairs. That
    detail is from the collaborator's method note and matters: correlating over
    each pair's own union would make every value incomparable to every other.
    """
    targets = [t for t in degs if not degs[t].empty]
    if len(targets) < 2:
        return {}
    deg_sets = [set(degs[t]["gene"]) for t in targets]
    union = sorted(set().union(*deg_sets))
    if not union:
        return {}

    J = pd.DataFrame(jaccard_matrix(deg_sets), index=targets, columns=targets)

    profile = np.full((len(union), len(targets)), np.nan)
    for j, t in enumerate(targets):
        tab = de[t].set_index("gene")["log2fc"]
        profile[:, j] = tab.reindex(union).to_numpy(dtype=float)
    profile = np.nan_to_num(profile, nan=0.0)
    S = pd.DataFrame(spearman_matrix(profile), index=targets, columns=targets)

    counts = pd.Series({t: len(degs[t]) for t in targets}, name="n_degs")
    return {"jaccard": J, "spearman": S, "n_degs": counts.to_frame()}


def perturbation_scores(
    X_log: np.ndarray,
    var_names: Sequence[str],
    target_by_cell: pd.Series,
    de: dict[str, pd.DataFrame],
    degs: dict[str, pd.DataFrame],
    ntc_label: str,
    cfg: PerturbConfig,
    guide_mapping: pd.DataFrame | None = None,
    pool_across_families: bool = False,
) -> pd.DataFrame:
    """Continuous per-cell perturbation score plus that cell's target expression.

    This is the Mixscale-style ranking the collaborator's target-expression
    heatmap needs: cells ordered from least to most perturbed, with target
    expression alongside, so escapers are visible.

    Scores are computed against the target's own family's control cells, for
    the same reason every other comparison here is: a score measured against
    another population's controls mostly reports the difference between the two
    populations.
    """
    gene_index = build_gene_index(var_names)
    ann = target_annotations(guide_mapping, ntc_label, pool_across_families)
    ctrl_by_family: dict[Any, Any] = {}
    for fam, key in ann.ntc_key_by_family.items():
        m = (target_by_cell == key).to_numpy()
        if m.sum():
            ctrl_by_family[fam] = take_rows(X_log, m)
    if not ctrl_by_family:
        return pd.DataFrame()
    rows = []
    for target, deg in degs.items():
        if deg.empty:
            continue
        family = ann.family_by_key.get(target)
        X_ctrl = ctrl_by_family.get(family)
        if X_ctrl is None:
            continue
        mask = (target_by_cell == target).to_numpy()
        idx = [gene_index[g] for g in deg["gene"] if g in gene_index]
        w = deg.loc[deg["gene"].isin(gene_index.keys()), "log2fc"].to_numpy()
        if not idx:
            continue
        score = perturbation_score(
            take_rows(X_log, mask), X_ctrl, np.array(idx), w
        )
        gi = resolve_gene(
            gene_index, ann.gene_by_key.get(target, target),
            ann.ensg_by_key.get(target),
        )
        target_expr = (
            take_column(X_log, gi)[mask] if gi is not None
            else np.full(int(mask.sum()), np.nan)
        )
        cells = np.flatnonzero(mask)
        rows.append(
            pd.DataFrame(
                {
                    "cell": cells,
                    "target_key": target,
                    "target_gene": ann.gene_by_key.get(target, target),
                    "family": family,
                    "perturbation_score": score,
                    "target_expression": target_expr,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def guide_pair_knockdown(
    X_log: np.ndarray,
    var_names: Sequence[str],
    per_cell: pd.DataFrame,
    ntc_label: str,
    cfg: PerturbConfig,
    min_pairs: int = 2,
) -> pd.DataFrame:
    """Knockdown broken out by guide (or guide pair) within each target.

    Answers "is this target's knockdown driven by one good guide?", which a
    per-target average hides.  The collaborator's report has this as the
    guide-pair dotplot; the original pipeline had nothing equivalent.
    """
    if "assigned_guide" not in per_cell.columns:
        return pd.DataFrame()
    gene_index = build_gene_index(var_names)

    group_col = "target_key" if "target_key" in per_cell.columns else "target_gene"
    is_ntc = (
        per_cell["is_ntc"].fillna(False).to_numpy(bool)
        if "is_ntc" in per_cell.columns
        else (per_cell["target_gene"] == ntc_label).to_numpy()
    )
    fam_col = "family" if "family" in per_cell.columns else None
    # Family-scoped control masks, so a guide's knockdown is measured against
    # its own population's controls.
    ctrl_masks: dict[Any, np.ndarray] = {}
    if fam_col:
        for fam in per_cell.loc[is_ntc, fam_col].dropna().unique():
            ctrl_masks[fam] = is_ntc & (per_cell[fam_col] == fam).to_numpy()
    else:
        ctrl_masks[None] = is_ntc
    if not any(int(m.sum()) >= cfg.min_cells_per_group for m in ctrl_masks.values()):
        return pd.DataFrame()

    key = "construct" if per_cell.get("construct") is not None and \
        per_cell["construct"].notna().any() else "assigned_guide"
    label_col = "short_label" if "short_label" in per_cell.columns else key

    rows = []
    for target, sub in per_cell.groupby(group_col, dropna=True):
        if bool(sub["is_ntc"].iloc[0]) if "is_ntc" in sub.columns else target == ntc_label:
            continue
        gene = str(sub["target_gene"].iloc[0]) if "target_gene" in sub.columns else str(target)
        family = sub[fam_col].iloc[0] if fam_col else None
        ntc_mask = ctrl_masks.get(family)
        if ntc_mask is None or int(ntc_mask.sum()) < cfg.min_cells_per_group:
            continue
        gi = resolve_gene(gene_index, gene)
        if gi is None:
            continue
        units = sub[key].dropna().unique()
        if len(units) < min_pairs:
            continue
        gene_col = take_column(X_log, gi)
        ntc_expr = gene_col[ntc_mask]
        for unit in units:
            m = (per_cell[key] == unit).to_numpy() & (
                per_cell[group_col] == target
            ).to_numpy()
            n = int(m.sum())
            if n < cfg.min_cells_per_group:
                continue
            labels = per_cell.loc[m, label_col].dropna()
            kd = percent_knockdown(gene_col[m], ntc_expr, log_input=True)
            rows.append(
                {
                    "target_key": target, "target_gene": gene, "family": family,
                    "unit": str(unit),
                    "unit_label": str(labels.iloc[0]) if len(labels) else str(unit),
                    "unit_type": key,
                    "n_cells": n, "pct_knockdown": kd["pct_knockdown"],
                    "log2fc": kd["log2fc"], "pvalue": kd["pvalue"],
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["padj"] = benjamini_hochberg(df["pvalue"].to_numpy())
        df["significance"] = df["padj"].map(significance_stars)
    return df


# ===========================================================================
# Figures
# ===========================================================================

def _label_col(df: "pd.DataFrame") -> str:
    """Which column to put on an axis.

    ``target_key`` ("ABT1_A") is preferred over ``target_gene`` ("ABT1"): with
    more than one guide family the bare symbol is ambiguous, and the four
    separate control pools all collapse onto a single "NTC" tick otherwise.
    """
    return "target_key" if "target_key" in df.columns else "target_gene"



def _barh_geometry(n_rows: int, base_h: float = 4.2) -> tuple[float, float]:
    """(figure height, tick fontsize) for a horizontal bar chart of n_rows.

    A fixed figure height with a fixed font is readable at 20 targets and
    illegible at 80 -- the labels simply overprint. Height grows with the row
    count and the font shrinks within bounds, which is what makes the
    knockdown panel legible and is now applied to every per-target bar chart.
    """
    height = float(np.clip(0.26 * max(n_rows, 1) + 1.6, base_h, 26.0))
    fontsize = float(np.clip(9.0 - 0.045 * max(n_rows - 20, 0), 5.0, 9.0))
    return height, fontsize


def plot_knockdown(
    kd: pd.DataFrame, ntc_label: str, fcfg: FigureConfig, path: Path
) -> Path:
    """%KD per target, control-cell expression, and effect vs cell count."""
    d = kd.sort_values("pct_knockdown", ascending=True)
    _h, _fs = _barh_geometry(len(d))
    fig, axes = plt.subplots(1, 3, figsize=(15, _h))

    colors = [
        "#8C8C8C" if not ok else ("#55A868" if v > 50 else "#DD8452" if v > 20
                                  else "#C44E52")
        for v, ok in zip(d["pct_knockdown"].fillna(0), d["target_detectable"])
    ]
    axes[0].barh(d[_label_col(d)], d["pct_knockdown"], color=colors)
    for i, (v, s) in enumerate(zip(d["pct_knockdown"], d["significance"])):
        if np.isfinite(v):
            axes[0].text(v, i, f" {v:.0f}% {s}", va="center", fontsize=6.5)
    axes[0].axvline(0, color="#222", lw=0.8)
    axes[0].set_xlabel(f"% knockdown vs {ntc_label}")
    axes[0].set_title("target-gene knockdown")
    for _ax in (axes[0], axes[1]):
        _ax.tick_params(axis="y", labelsize=_fs)

    axes[1].barh(d[_label_col(d)], d["mean_control"], color="#4C72B0")
    axes[1].set_xlabel(f"mean expression in {ntc_label} cells (linear)")
    axes[1].set_title("is the target even expressed?")

    ok = d["target_detectable"].to_numpy(bool)
    axes[2].scatter(d.loc[ok, "n_cells"], d.loc[ok, "pct_knockdown"], s=22,
                    color="#4C72B0", label="target detectable")
    if (~ok).any():
        axes[2].scatter(d.loc[~ok, "n_cells"], d.loc[~ok, "pct_knockdown"], s=22,
                        color="#C44E52", marker="x", label="target barely detected")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("cells with this perturbation")
    axes[2].set_ylabel("% knockdown")
    axes[2].axhline(0, color="#222", lw=0.8)
    axes[2].legend(fontsize=7)
    axes[2].set_title("effect vs power")
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_edistance(ed: pd.DataFrame, fcfg: FigureConfig, path: Path) -> Path:
    d = ed.sort_values("edistance", ascending=True)
    _h, _fs = _barh_geometry(len(d), base_h=4.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, _h))
    sig = d["padj_perm"] < 0.05
    axes[0].barh(d[_label_col(d)], d["edistance"],
                 color=np.where(sig, "#55A868", "#8C8C8C"))
    for i, (v, s) in enumerate(zip(d["edistance"], d["significance"])):
        if np.isfinite(v):
            axes[0].text(v, i, f" {s}", va="center", fontsize=6.5)
    axes[0].set_xlabel("E-distance from control (PCA space)")
    axes[0].set_title("transcriptome-wide effect size")
    axes[0].tick_params(axis="y", labelsize=_fs)

    axes[1].scatter(d["n_cells"], d["edistance"], s=22,
                    c=np.where(sig, "#55A868", "#8C8C8C"))
    axes[1].set_xscale("log")
    axes[1].set_xlabel("cells with this perturbation")
    axes[1].set_ylabel("E-distance")
    axes[1].set_title("E-distance is biased upward at small n")
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_deg_dotplot(
    degs: dict[str, pd.DataFrame], de: dict[str, pd.DataFrame],
    order: Sequence[str], cfg: PerturbConfig, fcfg: FigureConfig, path: Path,
) -> Path:
    """Top-N DEGs per perturbation, blocked by perturbation.

    Reproduces the collaborator's flagship dotplot: one block of genes per
    perturbation, the perturbation's own gene first in its block, and genes
    that are a top DEG for more than one perturbation labelled in red.
    """
    targets = [t for t in order if t in degs and not degs[t].empty]
    if not targets:
        fig, ax = plt.subplots(figsize=(7, 3))
        P.annotate_empty(ax, "no perturbation reached the DEG thresholds")
        return P.save_figure(fig, path, fcfg)

    # Cap the plot before building it. With 40 perturbations x 10 DEGs the
    # x-axis reaches ~400 gene columns, at which point the figure is metres
    # wide and every label overprints -- the panel stops being a figure and
    # becomes a data dump. The full tables are written to CSV regardless, so
    # nothing is lost by drawing a legible subset.
    n_targets_all = len(targets)
    if cfg.dotplot_max_targets and n_targets_all > cfg.dotplot_max_targets:
        targets = targets[:cfg.dotplot_max_targets]

    columns: list[str] = []
    blocks: list[tuple[str, int, int]] = []
    truncated_genes = False
    for t in targets:
        genes = [g for g in degs[t]["gene"] if g not in columns]
        if cfg.dotplot_max_genes:
            room = cfg.dotplot_max_genes - len(columns)
            if room <= 0:
                truncated_genes = True
                break
            if len(genes) > room:
                genes = genes[:room]
                truncated_genes = True
        start = len(columns)
        columns.extend(genes)
        blocks.append((t, start, len(columns)))
    targets = [t for t, _s, _e in blocks]
    if not columns:
        fig, ax = plt.subplots(figsize=(7, 3))
        P.annotate_empty(ax, "no DEGs to plot after capping")
        return P.save_figure(fig, path, fcfg)

    counts: dict[str, int] = {}
    for t in targets:
        for g in degs[t]["gene"]:
            counts[g] = counts.get(g, 0) + 1
    shared = [g for g, c in counts.items() if c > 1]

    size = pd.DataFrame(np.nan, index=targets, columns=columns)
    color = pd.DataFrame(np.nan, index=targets, columns=columns)
    for t in targets:
        tab = de[t].set_index("gene")
        sub = tab.reindex(columns)
        with np.errstate(divide="ignore"):
            nlp = -np.log10(sub["padj"].to_numpy(dtype=float))
        size.loc[t] = np.clip(np.nan_to_num(nlp, nan=0.0, posinf=cfg.dot_neglog10p_cap),
                              0, cfg.dot_neglog10p_cap)
        color.loc[t] = np.clip(sub["log2fc"].to_numpy(dtype=float),
                               -cfg.dot_log2fc_clip, cfg.dot_log2fc_clip)

    width = max(8.0, 0.20 * len(columns) + 3.0)
    height = max(3.0, 0.32 * len(targets) + 2.2)
    fig, ax = plt.subplots(figsize=(width, height))
    P.dotplot(
        ax, size, color, fcfg,
        size_label="-log10 adj. p", color_label="log2 fold change",
        vmin=-cfg.dot_log2fc_clip, vmax=cfg.dot_log2fc_clip,
        highlight_columns=shared, column_blocks=blocks,
    )
    _cap = ""
    if len(targets) < n_targets_all or truncated_genes:
        _cap = (f" -- showing {len(targets)} of {n_targets_all} perturbations "
                f"and {len(columns)} genes; full tables in the CSV")
    ax.set_title(
        f"top {cfg.de_top_n_per_perturbation} DEGs per perturbation "
        f"({len(shared)} gene(s) shared across perturbations, in red)" + _cap,
        fontsize=9,
    )
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_similarity(
    sim: dict[str, pd.DataFrame], cfg: PerturbConfig, fcfg: FigureConfig, path: Path
) -> Path:
    J, S = sim["jaccard"], sim["spearman"]
    n_degs = sim["n_degs"]["n_degs"]
    order_idx = P.hierarchical_order(S.to_numpy())
    labels = [S.index[i] for i in order_idx]
    Jo = J.loc[labels, labels]
    So = S.loc[labels, labels]

    side = max(5.0, 0.34 * len(labels) + 3.2)
    fig = plt.figure(figsize=(side + 1.2, side + 1.0))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1, 5], hspace=0.06,
        left=0.16, right=0.84, top=0.92, bottom=0.16,
    )
    ax_bar = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=ax_bar)

    vals = n_degs.reindex(labels).to_numpy(dtype=float)
    ax_bar.bar(range(len(labels)), vals, color="#4C72B0", width=0.8)
    # symlog only when the range actually needs it; on a small range it turns a
    # readable bar chart into an uninformative block.
    if np.nanmax(vals, initial=0) > 100:
        ax_bar.set_yscale("symlog")
    ax_bar.set_xlim(-0.5, len(labels) - 0.5)
    ax_bar.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_bar.tick_params(axis="y", labelsize=6)
    ax_bar.set_ylabel("DEGs", fontsize=7)
    ax_bar.set_title("perturbation similarity\n(upper: Jaccard of DEG sets; "
                     "lower: Spearman of log2FC profiles)", fontsize=9)

    P.triangular_similarity(ax, Jo, So, labels, fcfg,
                            upper_sqrt=cfg.jaccard_sqrt_colour_scale)
    # No tight_layout: the colorbar axes are positioned from ax.get_position(),
    # so re-laying-out the figure afterwards would move the parent out from
    # under them.
    return P.save_figure(fig, path, fcfg)


def plot_transcriptome_log2fc(
    de: dict[str, pd.DataFrame], kd: pd.DataFrame, fcfg: FigureConfig, path: Path
) -> Path:
    """Per-perturbation transcriptome-wide log2FC violin, target highlighted."""
    targets = [t for t in de if not de[t].empty]
    if not targets:
        fig, ax = plt.subplots(figsize=(7, 3))
        P.annotate_empty(ax, "no differential expression results")
        return P.save_figure(fig, path, fcfg)

    n_cells = kd.set_index("target_gene")["n_cells"] if not kd.empty else pd.Series()
    targets = sorted(targets, key=lambda t: -float(n_cells.get(t, 0)))

    fig = plt.figure(figsize=(max(7.0, 0.42 * len(targets) + 3.0), 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.08)
    ax_bar = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])

    ax_bar.bar(range(len(targets)), [float(n_cells.get(t, np.nan)) for t in targets],
               color="#8C8C8C")
    ax_bar.set_yscale("symlog")
    ax_bar.set_xlim(-0.5, len(targets) - 0.5)
    ax_bar.set_xticks([])
    ax_bar.set_ylabel("cells", fontsize=7)
    ax_bar.set_title("transcriptome-wide log2 fold change per perturbation",
                     fontsize=9)

    data, positions = [], []
    for i, t in enumerate(targets):
        v = de[t].loc[~de[t]["low_expression"], "log2fc"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        data.append(v)
        positions.append(i)
    if data:
        parts = ax.violinplot(data, positions=positions, showextrema=False,
                              widths=0.85)
        for b in parts["bodies"]:
            b.set_facecolor("#4C72B0")
            b.set_alpha(0.55)
            b.set_edgecolor("none")
    for i, t in enumerate(targets):
        own = de[t].loc[de[t]["gene"] == t, "log2fc"]
        if not own.empty and np.isfinite(own.iloc[0]):
            ax.plot([i], [own.iloc[0]], marker="D", ms=5, color="#C44E52",
                    zorder=8)
    ax.axhline(0, color="#222", lw=0.8)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=90, fontsize=7)
    ax.set_ylabel("log2 fold change vs control")
    ax.plot([], [], marker="D", ls="", color="#C44E52",
            label="the perturbation's own target gene")
    ax.legend(fontsize=7, loc="upper right")
    return P.save_figure(fig, path, fcfg)


def plot_perturbation_scores(
    scores: pd.DataFrame, fcfg: FigureConfig, path: Path, max_targets: int = 12
) -> Path:
    """Target expression against perturbation score, per perturbation."""
    if scores.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        P.annotate_empty(ax, "no perturbation scores computed")
        return P.save_figure(fig, path, fcfg)
    # Select and filter on the SAME column. This used to group on the bare
    # "target_gene" to pick the top targets, then filter each panel with
    # _label_col(scores) -- which returns "target_key" (gene+family, e.g.
    # "TGFBR2_B") whenever more than one guide family is present, which is
    # every MDL-1856-shaped run. A gene symbol never equals a target_key
    # string, so every panel's filter matched nothing: the grid rendered with
    # real targets selected and "(n=0)" in every title, regardless of how much
    # data `scores` actually held for them.
    label_col = _label_col(scores)
    targets = (
        scores.groupby(label_col).size().sort_values(ascending=False)
        .head(max_targets).index.tolist()
    )
    nrows, ncols = P.grid_dims(len(targets), max_cols=4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.5 * nrows),
                             squeeze=False)
    for ax, t in zip(axes.ravel(), targets):
        sub = scores[scores[label_col] == t].sort_values("perturbation_score")
        x = np.arange(len(sub))
        ax.scatter(x, sub["target_expression"], s=4, c=sub["perturbation_score"],
                   cmap=fcfg.continuous_cmap, linewidths=0)
        if len(sub) > 20:
            w = max(5, len(sub) // 20)
            smooth = (
                pd.Series(sub["target_expression"].to_numpy())
                .rolling(w, min_periods=1, center=True).mean()
            )
            ax.plot(x, smooth, color="#C44E52", lw=1.2)
        ax.set_title(f"{t} (n={len(sub):,})", fontsize=8)
        ax.set_xlabel("cells, ranked by perturbation score", fontsize=7)
        ax.set_ylabel("target expr (log1p)", fontsize=7)
    P.blank_unused_axes(axes, len(targets))
    fig.suptitle("target expression vs continuous perturbation score", fontsize=10)
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_guide_pairs(gp: pd.DataFrame, fcfg: FigureConfig, path: Path,
                     max_targets: int = 16) -> Path:
    if gp.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        P.annotate_empty(ax, "no target had enough guides for a per-guide breakdown")
        return P.save_figure(fig, path, fcfg)
    keep = (
        gp.groupby("target_gene")["pct_knockdown"].mean()
        .sort_values(ascending=False).head(max_targets).index
    )
    d = gp[gp[_label_col(gp)].isin(keep)]
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(keep) + 3), 4.2))
    targets = list(keep)
    for i, t in enumerate(targets):
        sub = d[d[_label_col(d)] == t]
        jitter = np.linspace(-0.22, 0.22, len(sub))
        ax.scatter(np.full(len(sub), i) + jitter, sub["pct_knockdown"],
                   s=np.clip(sub["n_cells"] / 5, 8, 90), color="#4C72B0",
                   alpha=0.8, edgecolors="#222", linewidths=0.3)
        ax.hlines(sub["pct_knockdown"].mean(), i - 0.3, i + 0.3, color="#C44E52",
                  lw=1.5)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=90, fontsize=7)
    ax.set_ylabel("% knockdown")
    ax.axhline(0, color="#222", lw=0.8)
    ax.set_title("knockdown per individual guide/construct (red bar = target mean; "
                 "dot size = cells)", fontsize=9)
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


# ===========================================================================
# Stage driver
# ===========================================================================
def run_perturbation_stage(
    X_log: np.ndarray,
    var_names: Sequence[str],
    pca: np.ndarray | None,
    guide_per_cell: pd.DataFrame | None,
    guide_mapping: pd.DataFrame | None,
    cfg: PipelineConfig,
    reg: Registry,
    max_de_targets: int = 40,
    batch_corrected: str = "none",
    group_columns: dict[str, pd.Series] | None = None,
) -> PerturbResults | None:
    if guide_per_cell is None or "target_gene" not in guide_per_cell.columns:
        reg.skipped(
            "perturbation", "all", "Perturbation analysis",
            "Needs guide assignments; no guide matrix was found in this experiment.",
        )
        return None

    pcfg, fcfg = cfg.perturb, cfg.figures
    fig_dir, table_dir = cfg.fig_dir, cfg.table_dir
    ntc = cfg.guide.ntc_label

    # Only cells with an unambiguous single perturbation are used. For
    # dual-guide libraries that means excluding recombinant pairs, which have
    # no single well-defined perturbation.
    # Not done in place (`&=`): pandas >= 2.x can hand back a read-only view
    # from `.to_numpy()` (always true under pandas 3.x, where copy-on-write is
    # no longer optional), and an in-place `&=` on that raises "output array
    # is read-only". `a & b` always allocates a fresh, writable array.
    usable = guide_per_cell["guide_is_assigned"].to_numpy(bool)
    if "guide_pair_status" in guide_per_cell.columns:
        usable = usable & (
            guide_per_cell["guide_pair_status"] != "recombinant_pair"
        ).to_numpy()

    # Cells whose guide family disagrees with their hashtag family are doublets
    # or index hops. They are counted and reported in the cross-check panel, but
    # excluded here: whichever family they were assigned to, they would
    # contaminate that family's comparison with cells from another population.
    n_conflict = 0
    if "family_conflict" in guide_per_cell.columns:
        conflict = guide_per_cell["family_conflict"].fillna(False).to_numpy(bool)
        n_conflict = int((conflict & usable).sum())
        usable = usable & ~conflict

    # Group on target_key ("ABT1_A"), not the bare gene symbol. Two libraries
    # can use the same symbol with different constructs, and every comparison
    # below is scoped to the target's own family.
    key_col = "target_key" if "target_key" in guide_per_cell.columns else "target_gene"
    target_by_cell = guide_per_cell[key_col].where(
        pd.Series(usable, index=guide_per_cell.index)
    )

    ann = target_annotations(guide_mapping, ntc,
                             cfg.guide.pool_ntc_across_families)
    n_ntc = int(target_by_cell.isin(ann.ntc_keys).sum())

    # What space E-distance is measured in, stated positively either way.
    _bc = str(batch_corrected or "none").lower()
    if _bc.startswith("harmony"):
        batch_note = (
            f"The PCA WAS batch-corrected ({batch_corrected}), so these "
            f"E-distances are measured in corrected space and any real "
            f"transcriptional difference that tracks the batch key has been "
            f"attenuated along with the technical one. Re-run with "
            f"--batch-correct none for uncorrected effect sizes."
        )
    elif _bc in ("none", ""):
        batch_note = (
            "the PCA was NOT batch-corrected, so these effect sizes are "
            "measured in uncorrected space and nothing has been integrated "
            "away."
        )
    else:
        batch_note = (
            f"the embedding arrived with correction already applied "
            f"({batch_corrected}), which was not done by this pipeline and "
            f"whose parameters are therefore unknown here."
        )

    if n_conflict:
        notes_early = (
            f"{n_conflict:,} cells were excluded from perturbation analysis "
            f"because their guide's family disagrees with the family implied by "
            f"their hashtag call. These are doublets or index hops; see the "
            f"guide-family vs hashtag-family cross-check."
        )
        reg.note("perturbation", "family_conflict",
                 "Cells excluded for family conflict", notes_early,
                 level="warn", order=5)

    if ann.n_families > 1:
        per_fam = {
            fam: int((target_by_cell == key).sum())
            for fam, key in ann.ntc_key_by_family.items()
        }
        reg.note(
            "perturbation", "family_scoped_controls",
            "Controls are scoped per family",
            (
                f"This experiment contains {ann.n_families} guide families, so "
                f"every knockdown, E-distance and DE comparison uses only its "
                f"own family's control cells. Control cells per family: "
                f"{per_fam}. Pooling them would compare a perturbation in one "
                f"cell line or library against controls from another."
            ),
            order=6,
        )

    if n_ntc == 0:
        reg.skipped(
            "perturbation", "all", "Perturbation analysis",
            (
                f"No control cells were identified. Perturbation effects are all "
                f"measured relative to non-targeting controls, so nothing can be "
                f"quantified without them. Guides are labelled controls when their "
                f"ID matches GuideConfig.ntc_regex; if this library names its "
                f"controls differently, set that pattern. "
                f"Guides seen: "
                f"{', '.join(map(str, sorted(set(guide_per_cell['target_gene'].dropna()))[:8]))}."
            ),
        )
        return None

    reg.metric("summary", "n_control_cells", f"Control ({ntc}) cells", n_ntc,
               order=40)

    # ----------------------------------------- control-pool diagnostics (B2/B4)
    # "NTC" is a pool, not a reagent. Before any effect size is reported, say
    # what the pool is actually made of.
    # The per-cell guide column is 'assigned_guide' (guide.py). Getting this
    # name wrong makes the whole control-pool section silently vanish rather
    # than fail, which is exactly the failure mode this release is about.
    _guide_col = next(
        (c for c in ("assigned_guide", "guide_id", "top_guide", "guide")
         if c in guide_per_cell.columns), None,
    )
    if _guide_col is not None:
        guides_by_family: dict[str, pd.Series] = {}
        for fam, key in ann.ntc_key_by_family.items():
            m = (target_by_cell == key).to_numpy()
            if m.any():
                guides_by_family[str(fam)] = guide_per_cell.loc[m, _guide_col]
        if guides_by_family:
            pool = CTRL.summarise_control_pool(guides_by_family)
            pool.to_csv(table_dir / "control_pool_composition.csv", index=False)
            reg.table(
                "perturbation", "control_pool", "What the control pool is made of",
                path=table_dir / "control_pool_composition.csv",
                inline=pool.round(3).to_dict("records"),
                columns=list(pool.columns),
                caption=(
                    "Every effect size in this section is measured against these "
                    "cells, so their composition sets the floor on what can be "
                    "believed. A pool resting on one guide means that guide's "
                    "own off-target profile is indistinguishable from the "
                    "biology of every perturbation compared against it."
                ),
                order=6,
            )
            for j, w in enumerate(CTRL.control_pool_warnings(pool)):
                reg.note("perturbation", f"control_pool_warn_{j}",
                         "Control-pool composition", w, level="warn",
                         order=7 + j)

            # Leave-one-out: are these controls actually interchangeable?
            for fam, guides in guides_by_family.items():
                g = pd.Series(guides).astype(str)
                if g.nunique() < 3:
                    continue
                m = (target_by_cell == ann.ntc_key_by_family[fam]).to_numpy()
                try:
                    loo = CTRL.leave_one_out_consistency(take_rows(X_log, m), g)
                except Exception:
                    continue
                if loo.empty:
                    continue
                safe = "".join(ch if ch.isalnum() or ch in "._-" else "_"
                               for ch in str(fam))
                loo.to_csv(table_dir / f"control_loo_{safe}.csv", index=False)
                reg.table(
                    "perturbation", f"control_loo_{safe}",
                    f"Control consistency, family {fam}",
                    path=table_dir / f"control_loo_{safe}.csv",
                    inline=loo.round(4).to_dict("records"),
                    columns=list(loo.columns),
                    caption=(
                        "Each control guide is held out in turn and the pooled "
                        "control mean recomputed. If the pool is genuinely "
                        "exchangeable, dropping any one guide barely moves it. A "
                        "guide that does move it is contributing something of "
                        "its own, and pooling flattens that into the baseline "
                        "every perturbation in this family is measured against."
                    ),
                    order=8,
                )
                for j, w in enumerate(CTRL.consistency_warnings(loo)):
                    reg.note("perturbation", f"control_loo_warn_{safe}_{j}",
                             f"Control consistency, family {fam}", w,
                             level="warn", order=9 + j)

    # Guide-unassigned cells, and per-cell depth, for the B1 fallback.
    _unassigned = target_by_cell.isna().to_numpy()
    _depth = None
    try:
        _depth = np.asarray(row_sums(X_log), dtype=float).ravel()
    except Exception:
        _depth = None
    _fallback: dict[str, Any] = {}

    kd, excluded = knockdown_table(
        X_log, var_names, target_by_cell, ntc, pcfg, guide_mapping,
        fallback_pool=_unassigned if _unassigned.any() else None,
        depth=_depth,
        fallback_out=_fallback,
        pool_across_families=cfg.guide.pool_ntc_across_families,
    )
    for j, (fam, info) in enumerate(sorted(_fallback.items())):
        reg.note(
            "perturbation", f"fallback_controls_{fam}",
            f"Fallback control pool, family {fam}",
            CTRL.describe_depth_match(fam, info),
            level="warn", order=10 + j,
        )
    notes: list[str] = []

    if kd.empty:
        reg.skipped(
            "perturbation", "knockdown", "Target-gene knockdown",
            "No target had both enough cells and its target gene present in the "
            "expression matrix.",
        )
    else:
        kd.to_csv(table_dir / "perturbation_knockdown.csv", index=False)
        reg.figure(
            "perturbation", "knockdown", "Target-gene knockdown",
            plot_knockdown(kd, ntc, fcfg, fig_dir / "perturbation_knockdown.png"),
            caption=T.kd_desc(ntc), order=10, width="full",
        )
        reg.note("perturbation", "kd_note", "Reading the knockdown panel",
                 T.KD_NOTE + " " + T.RESAMPLE_NOTE, order=15)
        reg.table(
            "perturbation", "knockdown_table", "Knockdown per target",
            path=table_dir / "perturbation_knockdown.csv",
            inline=kd.head(cfg.report.max_table_rows).round(4).to_dict("records"),
            columns=list(kd.columns), order=20,
        )
        n_strong = int((kd["pct_knockdown"] > 50).sum())
        reg.metric(
            "summary", "n_targets_kd50", "Targets with >50% knockdown",
            n_strong,
            level=("good" if n_strong >= max(1, 0.5 * len(kd))
                   else "warn" if n_strong else "poor"),
            order=41,
        )
        reg.metric(
            "summary", "median_knockdown", "Median knockdown",
            round(float(kd["pct_knockdown"].median()), 1), unit="%", order=42,
        )

    if not excluded.empty:
        excluded.to_csv(table_dir / "perturbation_excluded.csv", index=False)
        reg.table(
            "perturbation", "excluded", "Targets excluded from quantification",
            path=table_dir / "perturbation_excluded.csv",
            inline=excluded.to_dict("records"), columns=list(excluded.columns),
            caption=(
                "Listed rather than silently dropped: a target missing from this "
                "report because it had too few cells is a different problem from "
                "one that had cells and showed no effect."
            ),
            order=25,
        )

    ed = edistance_table(pca, target_by_cell, ntc, pcfg,
                         guide_mapping=guide_mapping,
                         pool_across_families=cfg.guide.pool_ntc_across_families) if pca is not None \
        else pd.DataFrame()
    if ed.empty:
        reg.skipped(
            "perturbation", "edistance", "E-distance from control",
            "Needs a PCA embedding and at least "
            f"{pcfg.min_cells_per_group} cells per group.",
        )
    else:
        ed.to_csv(table_dir / "perturbation_edistance.csv", index=False)
        reg.figure(
            "perturbation", "edistance", "Transcriptome-wide effect (E-distance)",
            plot_edistance(ed, fcfg, fig_dir / "perturbation_edistance.png"),
            caption=T.EDISTANCE_DESC, order=30, width="full",
        )
        reg.note("perturbation", "edist_note", "About E-distance",
                 T.EDISTANCE_NOTE, order=35)
        reg.note(
            "perturbation", "edist_space", "What space E-distance is measured in",
            (
                f"E-distance is computed on the first {pcfg.edistance_n_pcs} "
                f"principal components of the embedding &mdash; the same array "
                f"the UMAP and clusters are built from, not on the "
                f"log-normalised counts. That matters for one reason: "
                f"<strong>{batch_note}</strong> Everything else in this section "
                f"&mdash; fold changes, DEG counts, target knockdown and the "
                f"resampling test &mdash; is computed directly on the "
                f"log-normalised matrix and is unaffected by any embedding "
                f"choice."
            ),
            level=("warn" if batch_note.startswith("The PCA WAS") else "info"),
            order=36,
        )

    # ------------------------------------------------- per-condition (B5)
    # Pooling conditions answers "did this perturbation work?" but cannot
    # answer "did it work in every arm?", and the second question is the one
    # the experiment is asking.
    for ci, (axis_name, series) in enumerate((group_columns or {}).items()):
        s = pd.Series(series).astype(str)
        if s.nunique() < 2 or len(s) != len(target_by_cell):
            continue
        try:
            per_cond = per_condition_knockdown(
                X_log, var_names, target_by_cell, s, ntc, pcfg, guide_mapping,
                pool_across_families=cfg.guide.pool_ntc_across_families,
            )
        except Exception as exc:
            reg.skipped("perturbation", f"per_condition_{axis_name}",
                        f"Knockdown by {axis_name}",
                        f"Could not be computed ({exc}).")
            continue
        if per_cond.empty:
            reg.skipped(
                "perturbation", f"per_condition_{axis_name}",
                f"Knockdown by {axis_name}",
                f"No target had enough cells AND enough family-matched "
                f"controls within a single {axis_name} level. The pooled "
                f"knockdown table above is still valid; it just cannot be "
                f"broken down this way.",
            )
            continue
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_"
                       for ch in str(axis_name))
        per_cond.to_csv(table_dir / f"knockdown_by_{safe}.csv", index=False)
        spread = condition_effect_spread(per_cond)
        if not spread.empty:
            spread.to_csv(table_dir / f"knockdown_spread_{safe}.csv", index=False)
        reg.table(
            "perturbation", f"per_condition_{safe}",
            f"Knockdown by {axis_name}",
            path=table_dir / f"knockdown_by_{safe}.csv",
            inline=(spread if not spread.empty else per_cond)
            .head(25).round(3).to_dict("records"),
            columns=list((spread if not spread.empty else per_cond).columns),
            caption=(
                f"Target-gene knockdown recomputed within each {axis_name} "
                f"level. Controls are drawn from the same level as well as the "
                f"same family &mdash; comparing one condition's perturbed cells "
                f"against controls pooled across all conditions would fold the "
                f"condition effect into every knockdown estimate. "
                f"<code>range_pp</code> is the gap between a target's best and "
                f"worst arm, and it is the number that says whether pooling was "
                f"hiding something: a perturbation at 80% in one arm and 5% in "
                f"the other pools to ~45%, which describes neither."
            ),
            order=26 + ci,
        )
        for j, note in enumerate(condition_spread_notes(spread, str(axis_name))):
            reg.note("perturbation", f"cond_spread_{safe}_{j}",
                     f"Consistency across {axis_name}", note,
                     level=("warn" if "at least" in note else "info"),
                     order=27 + ci)

    de, degs, de_notes = differential_expression_per_target(
        X_log, var_names, target_by_cell, ntc, pcfg, max_de_targets,
        guide_mapping=guide_mapping,
        pool_across_families=cfg.guide.pool_ntc_across_families,
    )
    notes.extend(de_notes)

    sim: dict[str, pd.DataFrame] = {}
    if de:
        n_pert = len(de)
        order = (
            kd.sort_values("pct_knockdown", ascending=False)[_label_col(kd)].tolist()
            if not kd.empty else sorted(de)
        )
        sim = similarity_matrices(de, degs)
        if sim:
            order = [
                sim["spearman"].index[i]
                for i in P.hierarchical_order(sim["spearman"].to_numpy())
            ]

        reg.figure(
            "perturbation", "deg_dotplot", "Differential expression per perturbation",
            plot_deg_dotplot(degs, de, order, pcfg, fcfg,
                             fig_dir / "perturbation_deg_dotplot.png"),
            caption=T.deg_howto(
                n_pert, pcfg.de_padj_max, pcfg.de_abs_log2fc_min,
                pcfg.de_min_frac_detected_in_ntc, pcfg.de_top_n_per_perturbation,
                ntc,
            ),
            order=40, width="full",
        )
        reg.figure(
            "perturbation", "log2fc", "Transcriptome-wide fold changes",
            plot_transcriptome_log2fc(de, kd, fcfg,
                                      fig_dir / "perturbation_log2fc.png"),
            caption=T.TRANSCRIPTOME_LOG2FC_DESC, order=50, width="full",
        )
        if sim:
            reg.figure(
                "perturbation", "similarity", "Perturbation similarity",
                plot_similarity(sim, pcfg, fcfg,
                                fig_dir / "perturbation_similarity.png"),
                caption=T.similarity_desc(len(sim["jaccard"])), order=60,
                width="full",
            )
            sim["jaccard"].to_csv(table_dir / "perturbation_jaccard.csv")
            sim["spearman"].to_csv(table_dir / "perturbation_spearman.csv")

        all_degs = pd.concat(
            [d.assign(target_gene=t) for t, d in degs.items() if not d.empty],
            ignore_index=True,
        ) if any(not d.empty for d in degs.values()) else pd.DataFrame()
        if not all_degs.empty:
            all_degs.to_csv(table_dir / "perturbation_degs.csv", index=False)
            reg.metric(
                "summary", "n_perturbations_with_degs",
                "Perturbations with >=1 DEG",
                int(all_degs["target_gene"].nunique()), order=43,
            )

        scores = perturbation_scores(
            X_log, var_names, target_by_cell, de, degs, ntc, pcfg,
            guide_mapping=guide_mapping,
            pool_across_families=cfg.guide.pool_ntc_across_families,
        )
        if not scores.empty:
            reg.figure(
                "perturbation", "scores", "Continuous perturbation score",
                plot_perturbation_scores(
                    scores, fcfg, fig_dir / "perturbation_scores.png"
                ),
                caption=T.PERT_SCORE_DESC, order=70, width="full",
            )
    else:
        reg.skipped(
            "perturbation", "deg_dotplot", "Differential expression per perturbation",
            "Not enough control or perturbed cells to run differential expression.",
        )
        scores = pd.DataFrame()

    gp = guide_pair_knockdown(X_log, var_names, guide_per_cell.assign(
        target_gene=target_by_cell), ntc, pcfg)
    if gp.empty:
        reg.skipped(
            "perturbation", "guide_pairs", "Knockdown per guide",
            "No target had at least two guides with enough cells for a per-guide "
            "breakdown.",
        )
    else:
        gp.to_csv(table_dir / "perturbation_guide_pairs.csv", index=False)
        reg.figure(
            "perturbation", "guide_pairs", "Knockdown per individual guide",
            plot_guide_pairs(gp, fcfg, fig_dir / "perturbation_guide_pairs.png"),
            caption=(
                "Knockdown broken out by the individual guide or construct within "
                "each target. A target whose average knockdown is carried by one "
                "guide while the others do nothing is a guide-design result, not a "
                "biological one, and the per-target average hides it."
            ),
            order=80, width="full",
        )

    for i, note in enumerate(notes):
        reg.note("perturbation", f"note_{i}", "Note", note, level="info",
                 order=100 + i)

    return PerturbResults(
        knockdown=kd, edistance=ed, de=de, degs=degs, similarity=sim,
        scores=scores if isinstance(scores, pd.DataFrame) else None,
        guide_pairs=gp, excluded=excluded, notes=notes,
    )
