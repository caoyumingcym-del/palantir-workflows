"""
Pseudobulk comparability across conditions (B3).

The cheapest way to catch a condition that went wrong upstream. Every other
panel in this report looks at cells; this one collapses each condition level to
a single profile and asks whether the levels are comparable at all before any
per-cell comparison between them is believed.

Two profiles per condition level:

**Transcriptome.** Mean log-normalised expression per gene. Correlated across
levels at r > 0.95 is the normal picture for the same cells processed two ways;
a level sitting at 0.7 while the others sit at 0.98 is a processing failure, not
a biological finding, and no amount of per-cell testing will tell you that as
directly.

**gRNA composition.** The percentage of cells carrying each guide. The library
was pooled once, so composition should be near-identical across conditions
regardless of what the conditions did to the transcriptome. A condition whose
guide composition has drifted has lost cells non-randomly -- differential
dropout, a failed sort, uneven recovery -- and every per-guide effect size in
that condition inherits the bias.

The distinction matters: a transcriptome difference between conditions may be
real biology, but a *guide composition* difference between conditions almost
never is.

Deliberately simple: means and correlations, no model. This is a screening
panel, and a panel whose failure mode needs explaining is not a screening panel.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotting as P
from .artifacts import Registry
from .config import FigureConfig, PipelineConfig

# Values that mean "no condition recorded".  ``Series.astype(str)`` turns a
# real NaN into the literal string "nan", after which ``notna()``, ``dropna()``
# and ``nunique()`` all treat it as a genuine condition level -- so a phantom
# level appears in the figure, built from the cells that have no condition at
# all.  Every entry point here filters against the *original* series.
_BLANK_TOKENS = frozenset(
    {"", "nan", "none", "na", "n/a", "null", "<na>", "-", "."}
)


def pseudobulk_by_group(
    X: Any,
    var_names: Sequence[str],
    group: pd.Series,
    min_cells: int = 10,
) -> pd.DataFrame:
    """Mean expression per gene per group level. Genes x levels.

    Mean rather than sum: a sum is proportional to the number of cells in the
    level, so levels of different size would differ by construction and the
    correlation would be dominated by that.

    Levels below ``min_cells`` are dropped -- a two-cell "condition" produces a
    profile that is noise and a correlation that means nothing.
    """
    gs = pd.Series(group).reset_index(drop=True)
    txt = gs.astype(str).str.strip()
    blank = gs.isna() | txt.str.lower().isin(_BLANK_TOKENS)
    g = txt.mask(blank).to_numpy(dtype=object)
    n = len(g)
    A = X
    is_sparse = hasattr(A, "toarray") and hasattr(A, "shape")
    if getattr(A, "shape", (0, 0))[0] != n:
        raise ValueError(
            f"group has {n} entries but the matrix has "
            f"{getattr(A, 'shape', ('?',))[0]} rows"
        )

    out: dict[str, np.ndarray] = {}
    # Sorted, not first-appearance: this order has to match the one
    # ``gRNA_composition`` gets from ``pd.crosstab`` (which sorts), or the two
    # heatmaps in the same figure list the same conditions differently.
    levels = sorted({x for x in g if isinstance(x, str)})
    for lvl in levels:
        m = g == lvl
        k = int(m.sum())
        if k < min_cells:
            continue
        block = A[m]
        if is_sparse:
            mean = np.asarray(block.mean(axis=0)).ravel()
        else:
            mean = np.asarray(block, dtype=float).mean(axis=0)
        out[lvl] = mean
    if not out:
        return pd.DataFrame(index=pd.Index(list(map(str, var_names)), name="gene"))
    return pd.DataFrame(out, index=pd.Index(list(map(str, var_names)),
                                            name="gene"))


def correlation_matrix(pb: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    """Level-by-level correlation of pseudobulk profiles.

    Spearman by default: it answers "do these rank genes the same way?", which
    is the comparability question, and it is not dragged around by the handful
    of very high-expression genes that dominate a Pearson correlation on
    expression data.
    """
    if pb.empty or pb.shape[1] < 2:
        return pd.DataFrame()
    return pb.corr(method=method)


def gRNA_composition(
    guides: pd.Series,
    group: pd.Series,
    min_cells: int = 10,
) -> pd.DataFrame:
    """Percentage of cells carrying each guide, per group level. Guides x levels.

    Columns sum to 100. Expressed as a percentage rather than a count precisely
    so that unequal cell numbers between conditions cannot masquerade as a
    composition difference.
    """
    graw = pd.Series(group).reset_index(drop=True)
    vraw = pd.Series(guides).reset_index(drop=True)
    if len(graw) != len(vraw):
        raise ValueError(
            f"guides ({len(vraw)}) and group ({len(graw)}) differ in length"
        )
    g = graw.astype(str).str.strip()
    v = vraw.astype(str).str.strip()
    # The original wrote ``v.notna()`` here, which is always True: ``v`` had
    # already been through ``astype(str)``.  Test the original series instead,
    # and drop unlabelled conditions as well as unassigned guides.
    keep = (
        ~graw.isna() & ~vraw.isna()
        & ~g.str.lower().isin(_BLANK_TOKENS)
        & ~v.str.lower().isin(_BLANK_TOKENS)
    )
    g, v = g[keep], v[keep]

    tab = pd.crosstab(v, g)
    sizes = tab.sum(axis=0)
    tab = tab.loc[:, sizes >= min_cells]
    if tab.empty:
        return tab.astype(float)
    return 100.0 * tab / tab.sum(axis=0)


def composition_drift(comp: pd.DataFrame) -> pd.DataFrame:
    """Per-guide spread in representation across levels.

    ``max_abs_dev_pp`` is the largest deviation from that guide's mean
    representation, in percentage points. The library was pooled once, so this
    should be small for every guide; a guide that is 4% of one condition and
    0.5% of another has been lost non-randomly somewhere upstream.
    """
    if comp.empty or comp.shape[1] < 2:
        return pd.DataFrame()
    mean = comp.mean(axis=1)
    dev = comp.sub(mean, axis=0)
    out = pd.DataFrame({
        "guide": comp.index.astype(str),
        "mean_pct": mean.to_numpy(),
        "min_pct": comp.min(axis=1).to_numpy(),
        "max_pct": comp.max(axis=1).to_numpy(),
        "max_abs_dev_pp": dev.abs().max(axis=1).to_numpy(),
    })
    # Relative, because a 1 pp swing on a guide that is 1% of the library is a
    # different problem from 1 pp on a guide that is 20%.
    with np.errstate(divide="ignore", invalid="ignore"):
        out["max_rel_dev"] = np.where(
            out["mean_pct"] > 0, out["max_abs_dev_pp"] / out["mean_pct"], np.nan
        )
    return out.sort_values("max_abs_dev_pp", ascending=False).reset_index(
        drop=True
    )


def _shared_level_order(corr: pd.DataFrame, comp: pd.DataFrame) -> list[str]:
    """One level order shared by both heatmaps in the figure.

    ``pseudobulk_by_group`` used to build its columns in first-appearance order
    while ``pd.crosstab`` sorts them, so the transcriptome panel and the guide
    composition panel listed the same conditions in *different* orders, one
    directly above the other.  Reading one against the other meant matching
    labels by eye.  Both are now reindexed onto this order.
    """
    seen: set[str] = set()
    if not corr.empty:
        seen.update(str(c) for c in corr.columns)
    if not comp.empty:
        seen.update(str(c) for c in comp.columns)
    return sorted(seen)


def plot_comparability(
    pb: pd.DataFrame,
    corr: pd.DataFrame,
    comp: pd.DataFrame,
    drift: pd.DataFrame,
    axis_name: str,
    fcfg: FigureConfig,
    path: Path,
    max_guides_shown: int = 20,
) -> Path:
    """Four panels: transcriptome correlation, the worst pair as a scatter,
    guide composition, and per-guide drift.

    Formatting notes, because each one was a way the panel misled a reader:

    * The correlation heatmap masks its diagonal.  Self-correlation is 1.0 by
      construction, so including it pinned ``vmax`` at 1.0 and pushed a healthy
      0.977-0.993 spread into the black end of viridis -- the panel looked like
      a failed experiment.  The scale now spans the observed off-diagonal
      range, and the title says so, because a rescaled colour axis that does
      not announce itself is its own kind of lie.
    * Cell annotations pick black or white per cell from the luminance
      underneath, instead of a fixed dark grey that vanished on dark cells.
    * Condition names are set at 40 degrees rather than 90, and both heatmaps
      use the same level order.
    * The composition panel shows 20 guides, not 40, and states how many it
      omitted.  40 rows of labels in a half-height panel overlapped into a
      solid grey band.
    """
    order = _shared_level_order(corr, comp)
    if not corr.empty:
        keep = [c for c in order if c in corr.columns]
        corr = corr.loc[keep, keep]
    if not comp.empty:
        comp = comp[[c for c in order if c in comp.columns]]

    n_lvl = max(len(order), 2)
    label_len = max((len(str(x)) for x in order), default=6)
    # Long condition names need room under the axes or they are simply clipped.
    fig_w = min(24.0, max(14.0, 7.0 + 1.1 * n_lvl + 0.05 * label_len * n_lvl))
    fig_h = max(11.0, 8.0 + 0.10 * label_len)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))

    # 1. Level-by-level transcriptome correlation.
    ax = axes[0][0]
    if corr.empty:
        P.annotate_empty(ax, f"fewer than two {axis_name} levels", fcfg)
    else:
        off = corr.to_numpy(dtype=float).copy()
        np.fill_diagonal(off, np.nan)
        lo = float(np.nanmin(off))
        hi = float(np.nanmax(off))
        pad = max((hi - lo) * 0.08, 5e-4)
        P.heatmap(
            ax, corr, fcfg, cmap=fcfg.continuous_cmap,
            vmin=lo - pad, vmax=min(hi + pad, 1.0),
            cbar_label="Spearman rho", annotate=True, fmt="{:.3f}",
            mask_diagonal=True, cbar_fmt="{x:.3f}",
        )
        ax.set_title(
            f"Pseudobulk transcriptome correlation across {axis_name}\n"
            f"colour spans the observed off-diagonal range "
            f"{lo:.3f}\u2013{hi:.3f}; the diagonal is 1 by construction "
            f"and is masked",
            fontsize=P.fs(fcfg, "title"), fontweight="bold",
        )

    # 2. The weakest pair, plotted. A correlation is a summary; the scatter
    #    shows whether a low value is global compression or a few genes.
    ax = axes[0][1]
    if corr.empty:
        P.annotate_empty(ax, "needs two or more levels", fcfg)
    else:
        v = corr.to_numpy(dtype=float).copy()
        np.fill_diagonal(v, np.nan)
        iy, ix = np.unravel_index(int(np.nanargmin(v)), v.shape)
        a, b = str(corr.index[iy]), str(corr.columns[ix])
        ax.scatter(pb[a], pb[b], s=4, alpha=0.25,
                   color=P.palette(fcfg, 1)[0], edgecolors="none",
                   rasterized=len(pb) > 20000)
        lo = float(min(pb[a].min(), pb[b].min()))
        hi = float(max(pb[a].max(), pb[b].max()))
        ax.plot([lo, hi], [lo, hi], lw=1.6, color="#C44E52", ls="--",
                zorder=5, label="y = x")
        # Equal limits and equal aspect, so "off the diagonal" is judged
        # against a line that is actually at 45 degrees.
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"mean log expression\n{P.shorten(a, 34)}",
                      fontsize=P.fs(fcfg, "label"), fontweight="bold")
        ax.set_ylabel(f"mean log expression\n{P.shorten(b, 34)}",
                      fontsize=P.fs(fcfg, "label"), fontweight="bold")
        ax.set_title(
            f"Weakest pair, gene by gene\nrho = {float(v[iy, ix]):.3f}",
            fontsize=P.fs(fcfg, "title"), fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=P.fs(fcfg, "legend"))

    # 3. Guide composition per level.
    ax = axes[1][0]
    if comp.empty:
        P.annotate_empty(ax, "no guide assignments to compare", fcfg)
    else:
        top_idx = comp.mean(axis=1).sort_values(
            ascending=False).index[:max_guides_shown]
        show = comp.loc[top_idx]
        P.heatmap(ax, show, fcfg, cbar_label="% of cells", robust=True,
                  annotate=show.size <= 120, fmt="{:.2f}")
        ax.set_title(
            f"gRNA composition by {axis_name}\n"
            f"{len(show)} most-represented guides of {comp.shape[0]}",
            fontsize=P.fs(fcfg, "title"), fontweight="bold",
        )

    # 4. Per-guide drift.
    ax = axes[1][1]
    if drift.empty:
        P.annotate_empty(ax, "needs two or more levels", fcfg)
    else:
        top = drift.head(max_guides_shown)
        y = np.arange(len(top))[::-1]
        vals = top["max_abs_dev_pp"].to_numpy(dtype=float)
        ax.barh(y, vals, color=P.palette(fcfg, 1)[0], height=0.72)
        span = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
        span = span if span > 0 else 1.0
        # Label every bar: these bars are often near-identical in length, so
        # the ranking carries no information without the numbers.
        for yy, val in zip(y, vals):
            if np.isfinite(val):
                ax.text(val + span * 0.015, yy, f"{val:.2f}", va="center",
                        ha="left", fontsize=P.fs(fcfg, "annot"),
                        fontweight="bold", color="#333")
        ax.set_xlim(0, span * 1.20)
        ax.set_yticks(y)
        ax.set_yticklabels([P.shorten(g, 26) for g in top["guide"].astype(str)],
                           fontsize=P.fs(fcfg, "tick"), fontweight="bold")
        ax.set_xlabel(
            "largest deviation from that guide's mean\n"
            "representation (percentage points)",
            fontsize=P.fs(fcfg, "label"), fontweight="bold",
        )
        ax.set_title(
            f"Guides whose representation moves most across {axis_name}\n"
            f"worst {len(top)} of {len(drift)}",
            fontsize=P.fs(fcfg, "title"), fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", alpha=0.0)

    fig.suptitle(f"Pseudobulk comparability across {axis_name}",
                 fontsize=P.fs(fcfg, "suptitle"), fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return P.save_figure(fig, path, fcfg)


def _safe_key(axis_name: str, used: set[str]) -> str:
    """A filesystem- and registry-safe key that cannot collide with another axis.

    The old inline sanitiser mapped every non-alphanumeric character to ``_``,
    so ``"acoh 1"`` and ``"acoh/1"`` produced the same key.  The second
    registration then raised a duplicate-key ``ValueError``, which the caller
    in ``pipeline.py`` caught and reported as "could not be computed" -- an
    upstream naming coincidence presented to the reader as an analysis failure.
    """
    base = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in str(axis_name)
    ).strip("_") or "axis"
    key, n = base, 2
    while key in used:
        key = f"{base}_{n}"
        n += 1
    used.add(key)
    return key


def run_comparability_stage(
    X: Any,
    var_names: Sequence[str],
    group_columns: dict[str, pd.Series],
    guide_per_cell: pd.DataFrame | None,
    cfg: PipelineConfig,
    reg: Registry,
) -> dict[str, pd.DataFrame]:
    """Register a comparability panel per condition axis."""
    if not group_columns:
        reg.skipped(
            "comparability", "all", "Pseudobulk comparability",
            "No condition axes were resolved, so there is nothing to compare "
            "across.",
        )
        return {}

    guides = None
    if guide_per_cell is not None:
        for c in ("assigned_guide", "guide_id", "top_guide", "guide",
                  "target_key", "target_gene"):
            if c in guide_per_cell.columns:
                guides = guide_per_cell[c]
                break

    out: dict[str, pd.DataFrame] = {}
    used_keys: set[str] = set()
    # Honour the pipeline's own minimum-group-size setting rather than the
    # function default, so this panel and the perturbation panels agree on what
    # counts as too few cells to profile.
    min_cells = 10
    for holder in ("perturbation", "qc"):
        val = getattr(getattr(cfg, holder, None), "min_cells_per_group", None)
        if isinstance(val, (int, float)) and val > 0:
            min_cells = int(val)
            break

    for i, (axis_name, series) in enumerate(group_columns.items()):
        # The key is derived before anything can fail, so the skipped entry and
        # the produced entry for one axis always carry the same key.  They used
        # to disagree: the failure path used the raw axis name, the success path
        # the sanitised one.
        safe = _safe_key(axis_name, used_keys)
        title = f"Pseudobulk comparability by {axis_name}"

        raw = pd.Series(series).reset_index(drop=True)
        txt = raw.astype(str).str.strip()
        s = txt.mask(raw.isna() | txt.str.lower().isin(_BLANK_TOKENS))
        n_levels = int(s.dropna().nunique())
        if n_levels < 2:
            # Previously a bare ``continue``: the axis vanished from the report
            # with nothing said about it, which is the failure mode this
            # pipeline's own docstrings claim to have eliminated.
            reg.skipped(
                "comparability", f"axis_{safe}", title,
                f"Only {n_levels} distinct non-blank level(s) of "
                f"{axis_name} were present among these cells, so there is "
                f"nothing to compare across.",
                order=10 + i,
            )
            continue

        try:
            pb = pseudobulk_by_group(X, var_names, s, min_cells=min_cells)
            corr = correlation_matrix(pb)
            comp = pd.DataFrame()
            if guides is not None:
                g = pd.Series(guides).reset_index(drop=True)
                if len(g) == len(s):
                    comp = gRNA_composition(g, s)
                else:
                    # A length mismatch is a wiring bug, not an absence of
                    # data. Say so instead of silently dropping the bottom
                    # half of the figure.
                    reg.note(
                        "comparability", f"comp_misaligned_{safe}",
                        f"gRNA composition by {axis_name}",
                        f"Guide calls ({len(g):,} cells) and the "
                        f"{axis_name} annotation ({len(s):,} cells) are "
                        f"different lengths, so guide composition could not "
                        f"be compared across this axis. The transcriptome "
                        f"panels below are unaffected.",
                        level="warn", order=45 + i,
                    )
            drift = composition_drift(comp)
        except Exception as exc:
            reg.skipped("comparability", f"axis_{safe}", title,
                        f"Could not be computed ({exc}).", order=10 + i)
            continue

        if not corr.empty:
            corr.to_csv(cfg.table_dir / f"pseudobulk_corr_{safe}.csv")
        if not comp.empty:
            comp.to_csv(cfg.table_dir / f"grna_composition_{safe}.csv")
        if not drift.empty:
            # Panel 4's numbers were the only ones in this figure with no CSV
            # beside them, against a report that promises every number is
            # written out.
            drift.to_csv(cfg.table_dir / f"grna_drift_{safe}.csv", index=False)
        out[str(axis_name)] = corr

        reg.figure(
            "comparability", f"axis_{safe}",
            f"Pseudobulk comparability by {axis_name}",
            plot_comparability(pb, corr, comp, drift, str(axis_name),
                               cfg.figures,
                               cfg.fig_dir / f"pseudobulk_{safe}.png"),
            caption=(
                "Collapses each condition level to one profile and asks whether "
                "the levels are comparable at all, before any per-cell "
                "difference between them is believed. Top row: transcriptome. "
                "Bottom row: gRNA composition. The distinction matters &mdash; a "
                "transcriptome difference between conditions may be real "
                "biology, but the library was pooled once, so a <em>guide "
                "composition</em> difference almost never is. It means cells "
                "were lost non-randomly, and every per-guide effect size in "
                "that condition inherits the bias."
            ),
            order=10 + i, width="full",
        )
        # The old code recovered the severity by searching the rendered
        # sentence for substrings, one of which ("lost\\ncells") contained a
        # newline that the message never had and so could never match.
        # ``comparability_findings`` returns the level alongside the text.
        for j, (level, note) in enumerate(
            comparability_findings(corr, drift, str(axis_name))
        ):
            reg.note("comparability", f"note_{safe}_{j}",
                     f"Comparability across {axis_name}", note,
                     level=level, order=50 + i * 10 + j)
    return out


def comparability_notes(
    corr: pd.DataFrame,
    drift: pd.DataFrame,
    axis_name: str,
    corr_warn: float = 0.95,
    rel_drift_warn: float = 0.50,
) -> list[str]:
    """Report-ready findings from the two comparability views, text only."""
    return [
        text for _, text in comparability_findings(
            corr, drift, axis_name, corr_warn, rel_drift_warn
        )
    ]


def comparability_findings(
    corr: pd.DataFrame,
    drift: pd.DataFrame,
    axis_name: str,
    corr_warn: float = 0.95,
    rel_drift_warn: float = 0.50,
) -> list[tuple[str, str]]:
    """``(level, text)`` findings from the two comparability views.

    The level is decided where the finding is made, not recovered afterwards by
    grepping the rendered sentence.
    """
    notes: list[tuple[str, str]] = []

    if not corr.empty and corr.shape[0] >= 2:
        vals = corr.to_numpy(dtype=float).copy()
        np.fill_diagonal(vals, np.nan)
        worst = float(np.nanmin(vals))
        iy, ix = np.unravel_index(int(np.nanargmin(vals)), vals.shape)
        pair = f"{corr.index[iy]} vs {corr.columns[ix]}"
        if worst < corr_warn:
            notes.append(("warn",
                f"Pseudobulk transcriptomes are NOT well correlated across "
                f"{axis_name}: the weakest pair is {pair} at rho={worst:.3f} "
                f"(below {corr_warn:.2f}). Levels of the same cells processed "
                f"differently normally sit above 0.95, so check that level for "
                f"an upstream processing problem before interpreting any "
                f"per-cell difference between these conditions as biology."
            ))
        else:
            notes.append(("info",
                f"Pseudobulk transcriptomes are comparable across "
                f"{axis_name}: the weakest pair is {pair} at rho={worst:.3f}. "
                f"Per-condition comparisons rest on levels that are globally "
                f"similar, which is the precondition for reading a difference "
                f"between them as biology."
            ))

    if not drift.empty:
        bad = drift[drift["max_rel_dev"] > rel_drift_warn]
        worst = drift.iloc[0]
        if not bad.empty:
            notes.append(("warn",
                f"{len(bad)} guide(s) differ in representation across "
                f"{axis_name} by more than {100 * rel_drift_warn:.0f}% of their "
                f"own mean. The worst is {worst['guide']}, ranging "
                f"{worst['min_pct']:.2f}%-{worst['max_pct']:.2f}% of cells. The "
                f"library was pooled once, so guide composition should be "
                f"near-identical across conditions whatever the conditions did "
                f"to the transcriptome. A drift like this means cells were lost "
                f"non-randomly -- differential dropout, a failed sort, uneven "
                f"recovery -- and every per-guide effect size in the affected "
                f"condition inherits that bias."
            ))
        else:
            notes.append(("info",
                f"gRNA composition is stable across {axis_name}: the largest "
                f"deviation is {worst['guide']} at "
                f"{worst['max_abs_dev_pp']:.2f} percentage points from its mean "
                f"representation. Guide-level comparisons between these "
                f"conditions are not confounded by differential cell recovery."
            ))
    return notes
