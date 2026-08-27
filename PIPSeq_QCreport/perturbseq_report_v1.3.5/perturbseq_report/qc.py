"""
Per-cell QC: metrics, automatic thresholds, filtering, and the panels that
justify the thresholds.

Ordering is the thing that matters here and it is easy to get wrong:

    raw counts -> QC metrics -> filter cells -> normalise -> log1p -> HVG

Filtering must happen on **raw counts**, before normalisation, because
normalising to a fixed depth destroys exactly the library-size information the
count thresholds are testing.  The original got this order right; it is
restated explicitly and asserted here so a future edit cannot quietly break it.

The automatic thresholds are new.  The original hardcoded MIN_GENES=1000,
MAX_GENES=4000, MIN_COUNTS=500, MAX_COUNTS=12500, MAX_MITO=15 -- five numbers
tuned to one experiment, applied to every subsequent one, with a two-step
"explore then finalise" workflow whose only purpose was to make a human supply
better ones.  Here they are derived from the data by default, their provenance
is recorded, and the explore workflow remains available for when it matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotting as P
from .artifacts import Registry
from .config import FigureConfig, ModalityConfig, PipelineConfig, QCThresholds
from .modalities import Modality
from .stats import mad_bounds, to_dense
from . import text as T

QC_METRICS = ("total_counts", "n_genes_by_counts", "pct_counts_mt")


# ===========================================================================
# Metrics
# ===========================================================================
def compute_qc_metrics(
    gex: Any, cfg: ModalityConfig, guide: Modality | None = None,
    hto: Modality | None = None, prior: Any = None, counts=None,
) -> pd.DataFrame:
    """Per-cell QC metrics from the raw count matrix.

    Computed directly rather than via ``scanpy.pp.calculate_qc_metrics`` so
    that (a) the definitions are visible and testable here, and (b) the guide
    and hashtag depth metrics land in the same table, which is what makes the
    joint panels ("are guide-poor cells also RNA-poor?") possible.
    """
    var_names = pd.Index([str(v) for v in gex.var.index])
    # Counts may live in a layer rather than X (see provenance.detect): an
    # already-analysed object has normalised values in X, and computing UMI
    # totals from those would be meaningless.
    X = gex.X if counts is None else counts

    # Reuse QC metrics the upstream analysis already computed. Recomputing them
    # is wasteful, and on an object whose X is normalised it would also be
    # wrong -- so prefer what is there.
    if prior is not None and getattr(prior, "has_qc_metrics", False):
        reused = pd.DataFrame(index=pd.Index(
            [str(x) for x in gex.obs.index], name="barcode"))
        for canonical, actual in prior.qc_columns.items():
            reused[canonical] = pd.to_numeric(
                gex.obs[actual], errors="coerce").to_numpy()
        missing = [c for c in QC_METRICS if c not in reused.columns]
        if not missing:
            if "pct_counts_ribo" not in reused.columns:
                reused["pct_counts_ribo"] = np.nan
            with np.errstate(invalid="ignore", divide="ignore"):
                tot = reused["total_counts"].to_numpy(dtype=float)
                ng = reused["n_genes_by_counts"].to_numpy(dtype=float)
                reused["log10_genes_per_umi"] = np.where(
                    (tot > 0) & (ng > 0),
                    np.log10(np.maximum(ng, 1)) / np.log10(np.maximum(tot, 2)),
                    np.nan,
                )
            if guide is not None and guide.present:
                reused["guide_total_umis"] = guide.X.sum(axis=1)
                reused["n_guides_detected"] = (guide.X > 0).sum(axis=1).astype(float)
            if hto is not None and hto.present:
                reused["hto_total_umis"] = hto.X.sum(axis=1)
                reused["n_htos_detected"] = (hto.X > 0).sum(axis=1).astype(float)
            reused.attrs["reused_from_obs"] = dict(prior.qc_columns)
            return reused

    def prefix_mask(prefixes: Sequence[str]) -> np.ndarray:
        return np.array(
            [any(str(n).startswith(p) for p in prefixes) for n in var_names]
        )

    mt = prefix_mask(cfg.mito_prefixes)
    ribo = prefix_mask(cfg.ribo_prefixes)

    # Row sums and non-zero counts, sparse-aware, without densifying.
    if hasattr(X, "tocsr"):
        Xc = X.tocsr()
        total = np.asarray(Xc.sum(axis=1)).ravel()
        # getnnz counts stored entries per row directly from indptr. The obvious
        # `(Xc > 0).sum(axis=1)` instead materialises a whole second sparse
        # matrix with the same non-zero count -- several GB on a real
        # experiment, for a number already implicit in the structure.
        n_genes = Xc.getnnz(axis=1).astype(np.float64)

        def _subset_sum(mask: np.ndarray) -> np.ndarray:
            """Row sums over selected columns, without slicing the matrix.

            Column-slicing a CSR matrix is expensive and copies. Multiplying by
            an indicator vector touches the existing data in place instead.
            """
            if not mask.any():
                return np.zeros_like(total)
            indicator = mask.astype(np.float64)
            return np.asarray(Xc @ indicator).ravel()

        mt_sum = _subset_sum(mt)
        ribo_sum = _subset_sum(ribo)
    else:
        Xd = np.asarray(X, dtype=np.float64)
        total = Xd.sum(axis=1)
        n_genes = (Xd > 0).sum(axis=1)
        mt_sum = Xd[:, mt].sum(axis=1) if mt.any() else np.zeros_like(total)
        ribo_sum = Xd[:, ribo].sum(axis=1) if ribo.any() else np.zeros_like(total)

    with np.errstate(invalid="ignore", divide="ignore"):
        pct_mt = np.where(total > 0, 100.0 * mt_sum / total, np.nan)
        pct_ribo = np.where(total > 0, 100.0 * ribo_sum / total, np.nan)

    df = pd.DataFrame(
        {
            "total_counts": total,
            "n_genes_by_counts": n_genes.astype(float),
            "pct_counts_mt": pct_mt,
            "pct_counts_ribo": pct_ribo,
        },
        index=pd.Index([str(x) for x in gex.obs.index], name="barcode"),
    )
    # Complexity: genes per UMI. A low value at high depth is the signature of
    # an ambient-RNA droplet, and it separates them better than either metric
    # alone -- which is why it is worth having as its own column.
    with np.errstate(invalid="ignore", divide="ignore"):
        df["log10_genes_per_umi"] = np.where(
            (total > 0) & (n_genes > 0),
            np.log10(np.maximum(n_genes, 1)) / np.log10(np.maximum(total, 2)),
            np.nan,
        )

    if mt.sum() == 0:
        df.attrs["mito_warning"] = (
            f"No mitochondrial genes matched prefixes {list(cfg.mito_prefixes)}; "
            f"pct_counts_mt is 0 for every cell and the %mito gate will do "
            f"nothing. If this is a non-human/mouse reference, set "
            f"ModalityConfig.mito_prefixes accordingly."
        )

    if guide is not None and guide.present:
        df["guide_total_umis"] = guide.X.sum(axis=1)
        df["n_guides_detected"] = (guide.X > 0).sum(axis=1).astype(float)
    if hto is not None and hto.present:
        df["hto_total_umis"] = hto.X.sum(axis=1)
        df["n_htos_detected"] = (hto.X > 0).sum(axis=1).astype(float)

    return df


# ===========================================================================
# Automatic thresholds
# ===========================================================================
def auto_thresholds(
    qc: pd.DataFrame, base: QCThresholds, group: pd.Series | None = None
) -> QCThresholds:
    """Fill in any unset threshold from the data.

    Count and gene gates use asymmetric MAD bounds on log-transformed values
    (see ``stats.mad_bounds`` for why).  The %mito ceiling uses the MAD rule
    clamped into ``[auto_max_mito_floor, auto_max_mito_cap]``, because a MAD
    rule alone gives an absurdly tight ceiling on a very clean run and would
    discard good cells.

    ``group`` is accepted but deliberately *not* used to produce per-group
    thresholds. Filtering is applied globally, so a per-group threshold would
    be reported and not applied -- the confusing situation the original's
    docstring had to apologise for. Per-group distributions are instead shown
    as a diagnostic so the reader can judge whether one global gate is fair.
    """
    out = QCThresholds(
        **{k: v for k, v in base.as_dict().items()},
        source=dict(base.source),
        mad_lower=base.mad_lower,
        mad_upper=base.mad_upper,
        auto_min_genes_floor=base.auto_min_genes_floor,
        auto_min_counts_floor=base.auto_min_counts_floor,
        auto_max_mito_floor=base.auto_max_mito_floor,
        auto_max_mito_cap=base.auto_max_mito_cap,
    )

    counts = qc["total_counts"].to_numpy(dtype=float)
    genes = qc["n_genes_by_counts"].to_numpy(dtype=float)
    mito = qc["pct_counts_mt"].to_numpy(dtype=float)

    c_lo, c_hi = mad_bounds(counts, base.mad_lower, base.mad_upper, log=True)
    g_lo, g_hi = mad_bounds(genes, base.mad_lower, base.mad_upper, log=True)

    def _floor(mad_value: float, floor: float, observed: np.ndarray) -> float:
        """Raise a MAD bound to the safety floor -- but never past the median.

        The floor exists to stop a pathological run producing a near-zero lower
        gate that lets empty droplets through. It must not be able to reject
        most of the data: a genuinely shallow (or downsampled) experiment would
        otherwise have every cell filtered, which is worse than a permissive
        gate because it looks like a pipeline failure rather than a threshold
        choice. Capping the floor at the median guarantees the lower gate can
        never remove more than half the cells.
        """
        finite = observed[np.isfinite(observed)]
        if finite.size == 0:
            return float(mad_value)
        ceiling = float(np.median(finite))
        return float(min(max(mad_value, floor), max(ceiling, mad_value)))

    if out.min_counts is None:
        out.min_counts = _floor(c_lo, base.auto_min_counts_floor, counts)
        out.source["min_counts"] = "auto"
    if out.max_counts is None:
        out.max_counts = float(max(c_hi, out.min_counts * 2))
        out.source["max_counts"] = "auto"
    if out.min_genes is None:
        out.min_genes = _floor(g_lo, base.auto_min_genes_floor, genes)
        out.source["min_genes"] = "auto"
    if out.max_genes is None:
        out.max_genes = float(max(g_hi, out.min_genes * 2))
        out.source["max_genes"] = "auto"
    if out.max_mito is None:
        finite = mito[np.isfinite(mito)]
        if finite.size == 0 or np.allclose(finite, 0):
            out.max_mito = 100.0
            out.source["max_mito"] = "auto (no mito genes detected; gate disabled)"
        else:
            _, m_hi = mad_bounds(finite, base.mad_lower, base.mad_upper, log=False)
            out.max_mito = float(
                np.clip(m_hi, base.auto_max_mito_floor, base.auto_max_mito_cap)
            )
            out.source["max_mito"] = "auto"

    for k in out.as_dict():
        out.source.setdefault(k, "config")
    return out


# ===========================================================================
# Filtering
# ===========================================================================
@dataclass
class FilterResult:
    mask: np.ndarray
    thresholds: QCThresholds
    per_reason: pd.DataFrame          # cells failing each individual gate
    summary: pd.DataFrame             # per-group retention
    n_before: int
    n_after: int

    @property
    def frac_retained(self) -> float:
        return self.n_after / self.n_before if self.n_before else float("nan")


def filter_cells(
    qc: pd.DataFrame, th: QCThresholds, group: pd.Series | None = None
) -> FilterResult:
    """Apply the QC gates and report *why* cells were lost.

    Boundary convention is uniform and inclusive on both sides
    (``min <= x <= max``).  The original mixed conventions -- genes used
    ``>=``/``<`` while counts used ``>``/``<`` -- which is harmless but means
    two panels drawn with the same dashed line disagreed about whether a cell
    exactly on the line was kept.

    ``per_reason`` counts each gate independently (cells can fail several), so
    the report can say "we lost 12,000 cells, 9,000 of them for high %mito"
    rather than only reporting the total.  The original reported only the total.
    """
    n = len(qc)
    counts = qc["total_counts"].to_numpy(dtype=float)
    genes = qc["n_genes_by_counts"].to_numpy(dtype=float)
    mito = pd.to_numeric(qc["pct_counts_mt"], errors="coerce").fillna(0.0).to_numpy()

    gates = {
        "low_counts": counts < th.min_counts,
        "high_counts": counts > th.max_counts,
        "low_genes": genes < th.min_genes,
        "high_genes": genes > th.max_genes,
        "high_mito": mito > th.max_mito,
    }
    fail_any = np.zeros(n, dtype=bool)
    for m in gates.values():
        fail_any |= m
    mask = ~fail_any

    per_reason = pd.DataFrame(
        {
            "gate": list(gates),
            "threshold": [
                th.min_counts, th.max_counts, th.min_genes, th.max_genes, th.max_mito,
            ],
            "n_cells_failing": [int(m.sum()) for m in gates.values()],
            "pct_cells_failing": [100.0 * m.sum() / n if n else np.nan
                                  for m in gates.values()],
        }
    )
    # Cells lost only because of this gate -- the marginal cost of the gate.
    for i, (name, m) in enumerate(gates.items()):
        others = np.zeros(n, dtype=bool)
        for other_name, om in gates.items():
            if other_name != name:
                others |= om
        per_reason.loc[i, "n_cells_lost_only_here"] = int((m & ~others).sum())

    if group is not None:
        g = group.astype(str).to_numpy()
        rows = []
        for gv in sorted(pd.unique(g)):
            sel = g == gv
            rows.append(
                {
                    "group": gv,
                    "n_before": int(sel.sum()),
                    "n_after": int((sel & mask).sum()),
                    "frac_retained": (
                        float((sel & mask).sum() / sel.sum()) if sel.sum() else np.nan
                    ),
                }
            )
        summary = pd.DataFrame(rows)
    else:
        summary = pd.DataFrame(
            [{"group": "all", "n_before": n, "n_after": int(mask.sum()),
              "frac_retained": float(mask.sum() / n) if n else np.nan}]
        )

    return FilterResult(
        mask=mask, thresholds=th, per_reason=per_reason, summary=summary,
        n_before=n, n_after=int(mask.sum()),
    )


# ===========================================================================
# Figures
# ===========================================================================
def plot_threshold_hexbins(
    qc: pd.DataFrame, th: QCThresholds, fig_cfg: FigureConfig, path: Path,
    title_suffix: str = "",
) -> Path:
    """The headline QC panel: genes vs counts and genes vs %mito, as hexbin.

    Directly modelled on the collaborator's ``qc_thresholds`` figure, which is
    the single most useful QC plot in their report.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    P.hexbin_panel(
        axes[0], qc["total_counts"], qc["n_genes_by_counts"], fig_cfg,
        xlabel="total UMI counts", ylabel="genes detected",
        vlines=[th.min_counts, th.max_counts],
        hlines=[th.min_genes, th.max_genes],
        log_x=True, log_y=True,
        title=f"genes vs counts{title_suffix}",
    )
    P.hexbin_panel(
        axes[1], qc["total_counts"], qc["pct_counts_mt"], fig_cfg,
        xlabel="total UMI counts", ylabel="% mitochondrial",
        vlines=[th.min_counts, th.max_counts], hlines=[th.max_mito],
        log_x=True,
        title=f"%mito vs counts{title_suffix}",
    )
    fig.tight_layout()
    return P.save_figure(fig, path, fig_cfg)


def plot_qc_histograms(
    qc: pd.DataFrame, th: QCThresholds, fig_cfg: FigureConfig, path: Path,
    group: pd.Series | None = None,
) -> Path:
    """Marginal distributions of each gated metric, with the cut-offs drawn on."""
    specs = [
        ("total_counts", "total UMI counts", True,
         [th.min_counts, th.max_counts]),
        ("n_genes_by_counts", "genes detected", True,
         [th.min_genes, th.max_genes]),
        ("pct_counts_mt", "% mitochondrial", False, [th.max_mito]),
        ("log10_genes_per_umi", "complexity (log10 genes / log10 UMIs)", False, []),
    ]
    extra = [c for c in ("guide_total_umis", "hto_total_umis") if c in qc.columns]
    for c in extra:
        specs.append((c, c.replace("_", " "), True, []))

    nrows, ncols = P.grid_dims(len(specs), max_cols=3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.0 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    for ax, (col, label, logx, lines) in zip(flat, specs):
        if col not in qc.columns:
            P.annotate_empty(ax, f"{col} not available")
            continue
        if group is not None:
            by = {
                str(g): qc.loc[group.astype(str).to_numpy() == str(g), col].to_numpy()
                for g in sorted(pd.unique(group.astype(str)))
            }
        else:
            by = {"all cells": qc[col].to_numpy()}
        P.histogram_by_group(ax, by, fig_cfg, xlabel=label, log_x=logx,
                             vlines=lines)
    P.blank_unused_axes(axes, len(specs))
    fig.tight_layout()
    return P.save_figure(fig, path, fig_cfg)


def plot_qc_by_group(
    qc: pd.DataFrame, group: pd.Series, group_name: str, th: QCThresholds,
    fig_cfg: FigureConfig, path: Path,
) -> Path:
    """Violins of each gated metric split by a condition, with global cut-offs."""
    specs = [
        ("total_counts", "total UMI counts", True, [th.min_counts, th.max_counts]),
        ("n_genes_by_counts", "genes detected", True, [th.min_genes, th.max_genes]),
        ("pct_counts_mt", "% mitochondrial", False, [th.max_mito]),
    ]
    for c in ("guide_total_umis", "hto_total_umis", "n_guides_detected"):
        if c in qc.columns:
            specs.append((c, c.replace("_", " "), c.endswith("umis"), []))

    nrows, ncols = P.grid_dims(len(specs), max_cols=3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows),
                             squeeze=False)
    for ax, (col, label, logy, lines) in zip(axes.ravel(), specs):
        P.violin_by_group(ax, qc[col], group, fig_cfg, ylabel=label,
                          hlines=lines, log_y=logy)
    P.blank_unused_axes(axes, len(specs))
    fig.suptitle(f"per-cell QC by {group_name}", fontsize=10)
    fig.tight_layout()
    return P.save_figure(fig, path, fig_cfg)


def plot_retention(
    per_reason: pd.DataFrame, summary: pd.DataFrame, fig_cfg: FigureConfig,
    path: Path, cell_input: pd.Series | None = None,
    yield_summary: pd.DataFrame | None = None,
) -> Path:
    """Why cells were lost, and end-to-end yield per sample."""
    ncols = 3 if cell_input is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 3.6))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    pr = per_reason.sort_values("n_cells_failing", ascending=True)
    ax.barh(pr["gate"], pr["n_cells_failing"], color="#C44E52", alpha=0.75,
            label="failing this gate")
    ax.barh(pr["gate"], pr["n_cells_lost_only_here"], color="#4C72B0",
            label="lost only to this gate", height=0.45)
    ax.set_xlabel("cells")
    ax.set_title("cells failing each QC gate")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.bar(summary["group"].astype(str), summary["frac_retained"] * 100,
           color=P.palette(fig_cfg, len(summary)))
    for i, v in enumerate(summary["frac_retained"] * 100):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("% of cells retained")
    ax.set_ylim(0, 105)
    ax.set_title("QC retention")
    ax.tick_params(axis="x", rotation=45)

    if cell_input is not None and ncols == 3:
        ax = axes[2]
        # End-to-end yield needs SAMPLE grouping, because cell_input is per
        # sample. The retention bar to the left is grouped by the first
        # condition axis, which is a different thing entirely -- grouping the
        # yield panel the same way compared "CSU"/"IVT" against sample names
        # like "MDL1898_1" and matched nothing, so the panel rendered as "no
        # matching cell_input in manifest" on a manifest that had it.
        src = yield_summary if yield_summary is not None else summary
        merged = src.set_index(src["group"].astype(str))
        common = [g for g in merged.index if g in cell_input.index]
        if common:
            loaded = cell_input.loc[common].to_numpy(dtype=float)
            recovered = merged.loc[common, "n_after"].to_numpy(dtype=float)
            pct = 100.0 * recovered / np.where(loaded > 0, loaded, np.nan)
            ax.bar(common, pct, color="#55A868")
            for i, v in enumerate(pct):
                if np.isfinite(v):
                    ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=7)
            ax.set_ylabel("% of loaded cells recovered & retained")
            ax.set_title("end-to-end yield")
            ax.tick_params(axis="x", rotation=45)
        else:
            have = ", ".join(map(str, list(merged.index)[:3]))
            want = ", ".join(map(str, list(cell_input.index)[:3]))
            P.annotate_empty(
                ax,
                "cell_input present but no group matched\n"
                f"retention groups: {have}...\n"
                f"manifest samples: {want}...",
            )

    fig.tight_layout()
    return P.save_figure(fig, path, fig_cfg)


# ===========================================================================
# Stage driver
# ===========================================================================
def run_qc_stage(
    gex: Any,
    guide: Modality,
    hto: Modality,
    cfg: PipelineConfig,
    reg: Registry,
    group_columns: dict[str, pd.Series],
    cell_input: pd.Series | None = None,
    prior: Any = None,
    counts=None,
) -> tuple[pd.DataFrame, FilterResult]:
    """Compute QC metrics, resolve thresholds, plot, filter, and register.

    Returns ``(qc_table, filter_result)``.  The caller owns applying the mask,
    so this function has no side effects on ``gex``.
    """
    fig_dir, table_dir = cfg.fig_dir, cfg.table_dir
    fcfg = cfg.figures

    qc = compute_qc_metrics(gex, cfg.modality, guide, hto, prior=prior,
                            counts=counts)
    if qc.attrs.get("reused_from_obs"):
        reg.note(
            "cell_qc", "qc_reused", "QC metrics reused from the input",
            (
                "The per-cell QC metrics below were already present in the input "
                "object and have been reused rather than recomputed: "
                + ", ".join(f"<code>{a}</code> &rarr; {b}"
                            for b, a in qc.attrs['reused_from_obs'].items())
                + ". Pass <code>--force-recompute</code> to compute them from the "
                "count matrix instead."
            ),
            order=3,
        )
    if "mito_warning" in qc.attrs:
        reg.note("cell_qc", "mito_warning", "Mitochondrial genes not found",
                 qc.attrs["mito_warning"], level="warn", order=5)

    th = auto_thresholds(qc, cfg.qc)

    # --- provenance table -------------------------------------------------
    prov = pd.DataFrame(
        [
            {"threshold": k, "value": v, "source": th.source.get(k, "?")}
            for k, v in th.as_dict().items()
        ]
    )
    prov.to_csv(table_dir / "qc_thresholds.csv", index=False)
    reg.table(
        "cell_qc", "thresholds", "QC thresholds applied",
        path=table_dir / "qc_thresholds.csv",
        caption=T.THRESHOLD_PROVENANCE_NOTE,
        inline=prov.to_dict("records"),
        columns=list(prov.columns), order=10,
    )

    # --- pre-filter panels ------------------------------------------------
    reg.figure(
        "cell_qc", "hexbin_prefilter", "QC metrics and thresholds (before filtering)",
        plot_threshold_hexbins(qc, th, fcfg, fig_dir / "qc_hexbin_prefilter.png",
                               " -- all cells"),
        caption=T.CELL_QC_DESC, order=20, width="full",
    )
    reg.figure(
        "cell_qc", "hist_prefilter", "Marginal QC distributions (before filtering)",
        plot_qc_histograms(qc, th, fcfg, fig_dir / "qc_histograms_prefilter.png"),
        caption=(
            "Each gated metric on its own, with the applied cut-offs: "
            "<strong>total UMI counts</strong>, <strong>genes detected</strong>, "
            "<strong>% mitochondrial</strong>, and complexity (genes per UMI), plus "
            "guide and hashtag depth where present. Look for a cut-off sitting on a "
            "peak rather than in a valley &mdash; that is the sign of a threshold "
            "splitting a real population rather than trimming a tail."
        ),
        order=30, width="full",
    )

    for axis_name, series in group_columns.items():
        reg.figure(
            "cell_qc", f"by_{axis_name}", f"QC metrics by {axis_name}",
            plot_qc_by_group(qc, series, axis_name, th, fcfg,
                             fig_dir / f"qc_by_{axis_name}.png"),
            caption=T.QC_COMPARISON_DESC, order=40, width="full",
        )

    # --- filter ------------------------------------------------------------
    primary_group = next(iter(group_columns.values()), None)
    fr = filter_cells(qc, th, primary_group)

    # A second, sample-grouped retention summary, used only by the end-to-end
    # yield panel. Cheap -- it is mask arithmetic over the same QC table.
    # The sample label lives on obs, not in the QC table (which holds metrics
    # only), and must be realigned to the QC index before use.
    sample_series = None
    obs = getattr(gex, "obs", None)
    if obs is not None:
        for cand in cfg.modality.sample_col_candidates:
            if cand in obs.columns:
                s = obs[cand].astype(str)
                s.index = pd.Index([str(x) for x in obs.index])
                sample_series = s.reindex(qc.index)
                break
    fr_by_sample = (
        filter_cells(qc, th, sample_series) if sample_series is not None else None
    )

    fr.per_reason.to_csv(table_dir / "qc_gate_failures.csv", index=False)
    fr.summary.to_csv(table_dir / "qc_retention_by_group.csv", index=False)

    reg.figure(
        "cell_qc", "retention", "Cell retention",
        plot_retention(fr.per_reason, fr.summary, fcfg,
                       fig_dir / "qc_retention.png", cell_input,
                       yield_summary=(fr_by_sample.summary
                                      if fr_by_sample is not None else None)),
        caption=(
            "Left: how many cells each gate rejected, and how many were rejected "
            "by that gate alone -- a gate with a large total but a small "
            "alone-count is mostly re-rejecting cells another gate already caught. "
            + (T.RETENTION_DESC if cell_input is not None else "")
        ),
        order=50, width="full",
    )
    reg.table(
        "cell_qc", "gate_failures", "Cells rejected per QC gate",
        path=table_dir / "qc_gate_failures.csv",
        inline=fr.per_reason.to_dict("records"),
        columns=list(fr.per_reason.columns), order=60,
    )

    # A QC step that rejects most of the experiment is far more likely to be a
    # threshold problem than a sample problem, and it should be stated at the
    # top of the report rather than left for the reader to infer from a bar
    # chart. The original reported only the retained count, with no comment.
    if fr.frac_retained < 0.5:
        worst = fr.per_reason.sort_values("n_cells_failing", ascending=False).iloc[0]
        reg.note(
            "cell_qc", "low_retention", "Most cells were filtered out",
            (
                f"Only {fr.frac_retained * 100:.1f}% of cells passed QC "
                f"({fr.n_after:,} of {fr.n_before:,}). The largest single cause is "
                f"<code>{worst['gate']}</code> at a threshold of "
                f"{worst['threshold']:,.1f}, which rejected "
                f"{int(worst['n_cells_failing']):,} cells. Before interpreting "
                f"anything downstream, check the QC panels below: a loss this large "
                f"usually means a threshold is cutting through a real population "
                f"rather than removing debris."
            ),
            level="poor", order=1,
        )

    reg.metric("summary", "n_cells_before_qc", "Cells before QC", fr.n_before,
               order=10)
    reg.metric("summary", "n_cells_after_qc", "Cells after QC", fr.n_after,
               order=11)
    reg.metric(
        "summary", "pct_cells_retained", "Cells retained",
        round(100.0 * fr.frac_retained, 1), unit="%",
        level=("good" if fr.frac_retained > 0.7
               else "warn" if fr.frac_retained > 0.4 else "poor"),
        order=12,
    )

    # --- post-filter panel, for comparison ---------------------------------
    qc_post = qc.loc[fr.mask]
    reg.figure(
        "cell_qc", "hexbin_postfilter", "QC metrics (after filtering)",
        plot_threshold_hexbins(qc_post, th, fcfg,
                               fig_dir / "qc_hexbin_postfilter.png",
                               " -- retained cells"),
        caption=(
            "The same panel restricted to retained cells, as a check that the "
            "gates did what was intended and no population was cut in half."
        ),
        order=70, width="full",
    )

    qc.to_csv(table_dir / "qc_metrics_per_cell.csv.gz", compression="gzip")
    return qc, fr
