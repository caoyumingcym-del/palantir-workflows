"""
Shared plotting primitives.

Every figure in the pipeline goes through ``save_figure``, which closes the
figure afterwards.  The original leaked figures in several places (creating
figures inside loops without closing them), which on a large experiment
produces a stream of matplotlib warnings and eventually exhausts memory.

Plot helpers here take explicit arguments and return the figure.  None of them
read a module-level ``FIG_DIR``.  In the original, nearly every plotting
function ended with ``fig_dir if fig_dir is not None else FIG_DIR`` and several
used ``TABLE_DIR`` with no parameter at all -- and near the end of the file
``FIG_DIR``/``TABLE_DIR`` were silently reassigned back to their defaults by a
stray copy-paste block, so the last figure of the run could land somewhere
different from all the others.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")   # no display in a batch pipeline; set before pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.ticker import StrMethodFormatter
from matplotlib.figure import Figure

from .config import FigureConfig


# ===========================================================================
# Typography
# ===========================================================================
# Every label size in this module used to be an absolute number (5.5, 6, 6.5,
# 7, 8) written inline at the call site.  Two consequences: raising
# ``FigureConfig.font_size`` did nothing to the labels that were actually too
# small to read, and there was no single place to go to fix legibility.  Sizes
# now derive from one base through named roles.
#
# The floor exists because these figures are viewed as base64 PNGs embedded in
# an HTML report and scaled to the column width.  Below roughly 11 pt at the
# authored size nothing survives that scaling, whatever the config says.
_MIN_BASE_FONT = 11.0

_ROLE_SCALE = {
    "suptitle": 1.45,
    "title": 1.15,
    "label": 1.05,
    "tick": 1.00,
    "annot": 0.95,
    "legend": 0.95,
    "cbar": 1.00,
    "note": 0.95,
}


def base_font(cfg: FigureConfig | None) -> float:
    raw = getattr(cfg, "font_size", None) or _MIN_BASE_FONT
    return max(float(raw), _MIN_BASE_FONT)


def fs(cfg: FigureConfig | None, role: str = "label") -> float:
    """Point size for a named text role, derived from one base size."""
    return round(base_font(cfg) * _ROLE_SCALE.get(role, 1.0), 1)


def tick_layout(labels: Sequence[Any]) -> tuple[float, str]:
    """Rotation and horizontal alignment for categorical tick labels.

    Long condition names rotated a flat 90 degrees are technically present and
    practically unreadable, and they eat a tall strip of the panel that the
    data could have used.  40 degrees with right alignment reads as text.
    """
    longest = max((len(str(x)) for x in labels), default=0)
    if longest <= 3:
        return 0.0, "center"
    if longest <= 24:
        return 40.0, "right"
    return 60.0, "right"


def shorten(label: Any, limit: int = 28) -> str:
    """Middle-elide an over-long label instead of letting it run off the axes."""
    s = str(label)
    if len(s) <= limit:
        return s
    keep = max((limit - 1) // 2, 1)
    return f"{s[:keep]}\u2026{s[-keep:]}"


def text_on(cmap: Any, norm: Any, value: float) -> str:
    """Black or white, whichever is readable on the cell this value maps to.

    A fixed dark annotation colour is invisible on the dark end of any
    sequential colormap -- which is exactly where the interesting cells sit.

    Uses WCAG relative luminance rather than a naive Rec. 601 luma cut: the
    two agree at the extremes but disagree across the mid-greens of viridis,
    which is where most heatmap cells actually land.  White wins below a
    relative luminance of ``sqrt(0.0525) - 0.05``, the exact crossover of the
    two contrast ratios.
    """
    try:
        r, g, b = cmap(norm(float(value)))[:3]
    except Exception:
        return "#111111"

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    return "#111111" if lum >= 0.1791 else "#FFFFFF"


# ===========================================================================
# Style
# ===========================================================================
def apply_style(cfg: FigureConfig) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": cfg.dpi,
            "savefig.bbox": "tight",
            "font.size": base_font(cfg),
            "axes.titlesize": fs(cfg, "title"),
            "axes.titleweight": "bold",
            "axes.labelsize": fs(cfg, "label"),
            "axes.labelweight": "bold",
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "figure.titlesize": fs(cfg, "suptitle"),
            "figure.titleweight": "bold",
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": fs(cfg, "legend"),
            "legend.title_fontsize": fs(cfg, "legend"),
            "xtick.labelsize": fs(cfg, "tick"),
            "ytick.labelsize": fs(cfg, "tick"),
            "figure.autolayout": False,
        }
    )


def palette(cfg: FigureConfig, n: int) -> list[str]:
    """``n`` distinguishable colours, cycling the base palette then extending it.

    The original returned the base palette cycled, so with 15 groups colours 1
    and 11 were identical and the legend was ambiguous.  Beyond the base
    palette this interpolates through HSV to keep colours distinct.
    """
    base = list(cfg.palette)
    if n <= len(base):
        return base[:n]
    extra = n - len(base)
    hsv = plt.get_cmap("hsv")
    extended = [
        matplotlib.colors.to_hex(hsv((i + 0.5) / extra * 0.85))
        for i in range(extra)
    ]
    return base + extended


def density_cmap(cfg: FigureConfig):
    try:
        return plt.get_cmap(cfg.density_cmap)
    except ValueError:
        return LinearSegmentedColormap.from_list(
            "density", ["#f7fbff", "#4C72B0", "#1a2f4b"]
        )


# ===========================================================================
# Saving
# ===========================================================================
def save_figure(
    fig: Figure,
    path: Path | str,
    cfg: FigureConfig | None = None,
    close: bool = True,
) -> Path:
    """Write a figure and (by default) close it.

    Returns the path so the caller can register it in one expression:

        reg.figure("cell_qc", "hexbin", "QC thresholds",
                   save_figure(fig, fig_dir / "qc_hexbin.png", cfg))
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    dpi = cfg.dpi if cfg else 200
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    return p


def grid_dims(n: int, max_cols: int = 4) -> tuple[int, int]:
    """Rows and columns for ``n`` panels, at most ``max_cols`` wide."""
    if n <= 0:
        return 1, 1
    ncols = min(n, max_cols)
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def blank_unused_axes(axes: np.ndarray, n_used: int) -> None:
    flat = np.atleast_1d(axes).ravel()
    for ax in flat[n_used:]:
        ax.set_visible(False)


def annotate_empty(ax: plt.Axes, message: str,
                   cfg: FigureConfig | None = None) -> None:
    """Draw an explanatory message on an axis that has no data.

    Better than an empty white box: the reader learns *why* there is nothing
    there.  The original silently produced blank panels in this situation.
    """
    ax.text(
        0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
        fontsize=fs(cfg, "note"), fontweight="bold", color="#555", wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


# ===========================================================================
# Reusable panels
# ===========================================================================
def hexbin_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    cfg: FigureConfig,
    xlabel: str = "",
    ylabel: str = "",
    vlines: Sequence[float] = (),
    hlines: Sequence[float] = (),
    gridsize: int = 60,
    log_x: bool = False,
    log_y: bool = False,
    title: str = "",
    colorbar: bool = True,
) -> Any:
    """Hexbin density with threshold lines.

    Hexbin rather than a scatter is taken from the collaborator's QC panel and
    is the right call: with 100k+ cells a scatter is a solid black blob that
    hides exactly the structure you need to see when choosing thresholds.
    Counts are on a log colour scale for the same reason.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if log_x:
        ok &= x > 0
    if log_y:
        ok &= y > 0
    if ok.sum() < 3:
        annotate_empty(ax, "too few cells with finite values to plot", cfg)
        return None

    hb = ax.hexbin(
        x[ok], y[ok], gridsize=gridsize, cmap=density_cmap(cfg),
        bins="log", mincnt=1, linewidths=0,
        xscale="log" if log_x else "linear",
        yscale="log" if log_y else "linear",
    )
    for v in vlines:
        if v is not None and np.isfinite(v):
            ax.axvline(v, color="#C44E52", ls="--", lw=1.1, zorder=5)
    for h in hlines:
        if h is not None and np.isfinite(h):
            ax.axhline(h, color="#C44E52", ls="--", lw=1.1, zorder=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if colorbar:
        cb = ax.figure.colorbar(hb, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("cells per bin", fontsize=fs(cfg, "cbar"),
                     fontweight="bold")
    return hb


def violin_by_group(
    ax: plt.Axes,
    values: pd.Series,
    groups: pd.Series,
    cfg: FigureConfig,
    ylabel: str = "",
    hlines: Sequence[float] = (),
    log_y: bool = False,
    max_groups: int = 30,
    show_n: bool = True,
) -> None:
    """Violin plot of one metric split by group, with medians and group sizes."""
    df = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"),
                       "g": groups.astype(str)}).dropna()
    if df.empty:
        annotate_empty(ax, "no finite values", cfg)
        return
    order = sorted(df["g"].unique())
    if len(order) > max_groups:
        # Keep the largest groups; naming the omission beats silently
        # truncating, which is what a bare head() would do.
        counts = df["g"].value_counts()
        order = sorted(counts.head(max_groups).index)
        df = df[df["g"].isin(order)]
        ax.set_title(f"showing {max_groups} largest groups of {counts.size}",
                     fontsize=fs(cfg, "note"), color="#666",
                     fontweight="bold")

    data = [df.loc[df["g"] == g, "v"].to_numpy() for g in order]
    data = [d if d.size else np.array([np.nan]) for d in data]
    colors = palette(cfg, len(order))

    parts = ax.violinplot(
        [d[np.isfinite(d)] if np.isfinite(d).any() else np.array([0.0]) for d in data],
        positions=np.arange(len(order)), showextrema=False, widths=0.85,
    )
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.65)
        body.set_edgecolor("none")

    for i, d in enumerate(data):
        dd = d[np.isfinite(d)]
        if dd.size:
            ax.hlines(np.median(dd), i - 0.32, i + 0.32, color="#222", lw=1.4,
                      zorder=6)
    for h in hlines:
        if h is not None and np.isfinite(h):
            ax.axhline(h, color="#C44E52", ls="--", lw=1.0)

    labels = list(order)
    if show_n:
        labels = [
            f"{g}\nn={int(np.isfinite(d).sum()):,}" for g, d in zip(order, data)
        ]
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, rotation=45 if len(order) > 4 else 0,
                       ha="right" if len(order) > 4 else "center")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")


def stacked_fraction_bars(
    ax: plt.Axes,
    frac: pd.DataFrame,
    cfg: FigureConfig,
    ylabel: str = "fraction of cells",
    legend_title: str | None = None,
    annotate_min: float = 0.05,
) -> None:
    """Stacked proportion bars (rows = groups, columns = categories).

    Segments above ``annotate_min`` are labelled with their percentage, so the
    reader gets the numbers without having to eyeball a stacked bar -- a common
    complaint about the original's composition plots.
    """
    if frac.empty:
        annotate_empty(ax, "no data", cfg)
        return
    cats = list(frac.columns)
    colors = palette(cfg, len(cats))
    bottom = np.zeros(len(frac))
    xs = np.arange(len(frac))
    for cat, c in zip(cats, colors):
        vals = frac[cat].to_numpy(dtype=float)
        ax.bar(xs, vals, bottom=bottom, color=c, label=str(cat), width=0.75,
               edgecolor="white", linewidth=0.4)
        for x, v, b in zip(xs, vals, bottom):
            if np.isfinite(v) and v >= annotate_min:
                ax.text(x, b + v / 2, f"{v * 100:.0f}%", ha="center", va="center",
                        fontsize=fs(cfg, "annot"), color="white",
                        fontweight="bold")
        bottom += np.nan_to_num(vals)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in frac.index],
                       rotation=45 if len(frac) > 4 else 0,
                       ha="right" if len(frac) > 4 else "center")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.legend(title=legend_title, bbox_to_anchor=(1.01, 1), loc="upper left")


def grouped_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    category_col: str,
    cfg: FigureConfig,
    ylabel: str = "",
    err_col: str | None = None,
    annotate: bool = False,
) -> None:
    """Side-by-side bars: categories on x, one bar per group."""
    if df.empty:
        annotate_empty(ax, "no data", cfg)
        return
    cats = list(pd.unique(df[category_col]))
    groups = list(pd.unique(df[group_col]))
    colors = palette(cfg, len(groups))
    width = 0.8 / max(len(groups), 1)
    xs = np.arange(len(cats))
    for i, (g, c) in enumerate(zip(groups, colors)):
        sub = df[df[group_col] == g].set_index(category_col)
        vals = [sub[value_col].get(cat, np.nan) for cat in cats]
        errs = (
            [sub[err_col].get(cat, np.nan) for cat in cats]
            if err_col and err_col in sub.columns else None
        )
        pos = xs - 0.4 + width * (i + 0.5)
        ax.bar(pos, vals, width=width * 0.92, color=c, label=str(g),
               yerr=errs, capsize=2 if errs else 0, error_kw={"lw": 0.7})
        if annotate:
            for x, v in zip(pos, vals):
                if v is not None and np.isfinite(v):
                    ax.text(x, v, f"{v:,.0f}", ha="center", va="bottom",
                            fontsize=fs(cfg, "annot"), fontweight="bold",
                            rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(c) for c in cats],
                       rotation=45 if len(cats) > 4 else 0,
                       ha="right" if len(cats) > 4 else "center")
    ax.set_ylabel(ylabel)
    if len(groups) > 1:
        ax.legend(title=group_col, bbox_to_anchor=(1.01, 1), loc="upper left")


def rank_abundance(
    ax: plt.Axes,
    series_by_group: dict[str, np.ndarray],
    cfg: FigureConfig,
    ylabel: str = "UMIs",
    xlabel: str = "rank",
    log_y: bool = True,
    normalise: bool = True,
) -> None:
    """Rank-abundance curves, one line per group.

    ``normalise`` divides by each group's total so libraries of different
    depths are comparable -- otherwise the curve mostly reflects sequencing
    depth, and comparing groups is meaningless.
    """
    colors = palette(cfg, len(series_by_group))
    plotted = False
    for (label, vals), c in zip(series_by_group.items(), colors):
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        v = np.sort(v)[::-1]
        if normalise and v.sum() > 0:
            v = v / v.sum()
        ax.plot(np.arange(1, v.size + 1), np.clip(v, 1e-12, None),
                label=str(label), color=c, lw=1.4)
        plotted = True
    if not plotted:
        annotate_empty(ax, "no features with counts", cfg)
        return
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(("fraction of " + ylabel) if normalise else ylabel)
    if len(series_by_group) > 1:
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")


def histogram_by_group(
    ax: plt.Axes,
    values_by_group: dict[str, np.ndarray],
    cfg: FigureConfig,
    xlabel: str = "",
    bins: int = 60,
    log_x: bool = False,
    vlines: Sequence[float] = (),
    density: bool = True,
) -> None:
    """Overlaid step histograms, one per group."""
    colors = palette(cfg, len(values_by_group))
    allv = np.concatenate(
        [np.asarray(v, float).ravel() for v in values_by_group.values()]
    ) if values_by_group else np.array([])
    allv = allv[np.isfinite(allv)]
    if log_x:
        allv = allv[allv > 0]
    if allv.size < 3:
        annotate_empty(ax, "too few finite values", cfg)
        return
    if log_x:
        edges = np.logspace(np.log10(allv.min()), np.log10(allv.max()), bins)
    else:
        edges = np.linspace(allv.min(), allv.max(), bins)

    for (label, vals), c in zip(values_by_group.items(), colors):
        v = np.asarray(vals, float).ravel()
        v = v[np.isfinite(v)]
        if log_x:
            v = v[v > 0]
        if v.size == 0:
            continue
        ax.hist(v, bins=edges, histtype="step", lw=1.3, color=c,
                label=f"{label} (n={v.size:,})", density=density)
    for x in vlines:
        if x is not None and np.isfinite(x):
            ax.axvline(x, color="#C44E52", ls="--", lw=1.1)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density" if density else "cells")
    if len(values_by_group) > 1:
        ax.legend()


def scatter_embedding(
    ax: plt.Axes,
    xy: np.ndarray,
    values: pd.Series,
    cfg: FigureConfig,
    title: str = "",
    categorical: bool | None = None,
    label_clusters: bool = False,
    point_size: float | None = None,
    cap_percentile: float | None = None,
    legend: bool = True,
    max_legend_entries: int = 24,
) -> None:
    """Embedding scatter coloured by a categorical or continuous variable.

    ``cap_percentile`` clips the continuous colour scale (99 for total counts),
    which the collaborator's report does for good reason: without it a handful
    of high-count multiplets compress the colour range so the bulk of cells are
    all one shade.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2 or xy.shape[0] == 0:
        annotate_empty(ax, "no embedding available", cfg)
        return
    n = xy.shape[0]
    s = point_size if point_size is not None else max(0.4, min(6.0, 12000 / max(n, 1)))

    if categorical is None:
        categorical = not pd.api.types.is_numeric_dtype(values)

    if categorical:
        cats = pd.Series(values).astype(str)
        order = sorted(cats.unique())
        colors = palette(cfg, len(order))
        for cat, c in zip(order, colors):
            m = (cats == cat).to_numpy()
            ax.scatter(xy[m, 0], xy[m, 1], s=s, c=c, linewidths=0,
                       label=f"{cat} ({int(m.sum()):,})", rasterized=n > 20000)
            if label_clusters and m.sum() > 0:
                cx, cy = np.median(xy[m, 0]), np.median(xy[m, 1])
                ax.text(cx, cy, str(cat), fontsize=fs(cfg, "annot"), fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.75), zorder=10)
        if legend and len(order) <= max_legend_entries:
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", markerscale=3,
                      fontsize=fs(cfg, "legend"))
    else:
        v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        vmax = (
            np.nanpercentile(v, cap_percentile)
            if cap_percentile is not None and np.isfinite(v).any() else np.nanmax(v)
        )
        vmin = np.nanmin(v) if np.isfinite(v).any() else 0.0
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=s, c=v, cmap=cfg.continuous_cmap,
                        norm=Normalize(vmin=vmin, vmax=vmax), linewidths=0,
                        rasterized=n > 20000)
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        if cap_percentile is not None:
            cb.ax.tick_params(labelsize=fs(cfg, "cbar"))
        if cap_percentile is not None:
            cb.set_label(f"capped at p{cap_percentile:g}",
                         fontsize=fs(cfg, "cbar"), fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if title:
        ax.set_title(title)


def dotplot(
    ax: plt.Axes,
    size_matrix: pd.DataFrame,
    color_matrix: pd.DataFrame,
    cfg: FigureConfig,
    size_label: str = "size",
    color_label: str = "colour",
    size_scale: float = 90.0,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    highlight_columns: Sequence[str] = (),
    column_blocks: Sequence[tuple[str, int, int]] = (),
) -> None:
    """Generic dot plot: rows on y, features on x, size and colour matrices.

    ``highlight_columns`` draws those x tick labels in red -- used for
    "this gene is a top DEG for more than one perturbation", the
    shared-response flag from the collaborator's dotplot.  ``column_blocks``
    draws separators and a header per perturbation block.
    """
    if size_matrix.empty:
        annotate_empty(ax, "nothing to plot", cfg)
        return
    rows = list(size_matrix.index)
    cols = list(size_matrix.columns)
    S = size_matrix.to_numpy(dtype=float)
    C = color_matrix.reindex(index=rows, columns=cols).to_numpy(dtype=float)

    xs, ys = np.meshgrid(np.arange(len(cols)), np.arange(len(rows)))
    smax = np.nanmax(S) if np.isfinite(S).any() else 1.0
    smax = smax if smax > 0 else 1.0
    sizes = np.clip(S / smax, 0, 1) * size_scale
    finite = np.isfinite(S) & np.isfinite(C)

    sc = ax.scatter(
        xs[finite], ys[finite], s=sizes[finite], c=C[finite],
        cmap=cmap or cfg.diverging_cmap,
        norm=Normalize(vmin=vmin, vmax=vmax), edgecolors="#444", linewidths=0.3,
    )
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([shorten(c, 22) for c in cols], rotation=90,
                       fontsize=fs(cfg, "tick"), fontweight="bold")
    hi = set(highlight_columns)
    for tick, col in zip(ax.get_xticklabels(), cols):
        if col in hi:
            tick.set_color("#C44E52")
            tick.set_fontweight("bold")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([shorten(r, 26) for r in rows],
                       fontsize=fs(cfg, "tick"), fontweight="bold")
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.grid(alpha=0.15)

    for label, start, end in column_blocks:
        if start > 0:
            ax.axvline(start - 0.5, color="#999", lw=0.6, ls=":")
        ax.text((start + end - 1) / 2, -0.9, str(label), ha="center", va="bottom",
                fontsize=fs(cfg, "annot"), rotation=0, color="#333",
                fontweight="bold")

    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.02, pad=0.01)
    cb.ax.tick_params(labelsize=fs(cfg, "cbar"))
    cb.set_label(color_label, fontsize=fs(cfg, "cbar"), fontweight="bold")

    # Size legend: a dot plot without one is not readable.
    handles = []
    for frac in (0.25, 0.5, 1.0):
        handles.append(
            ax.scatter([], [], s=size_scale * frac, c="#888", edgecolors="#444",
                       linewidths=0.3, label=f"{frac * smax:.3g}")
        )
    ax.legend(handles=handles, title=size_label, bbox_to_anchor=(1.06, 0),
              loc="lower left", fontsize=fs(cfg, "legend"),
              title_fontsize=fs(cfg, "legend"), labelspacing=1.1)


def heatmap(
    ax: plt.Axes,
    M: pd.DataFrame,
    cfg: FigureConfig,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
    annotate: bool = False,
    fmt: str = "{:.2f}",
    log_color: bool = False,
    mask_diagonal: bool = False,
    robust: bool = False,
    max_annot_cells: int = 400,
    cbar_fmt: str | None = None,
) -> Any:
    """Annotated heatmap with a colour scale that spans the data actually shown.

    Three changes over the original, each fixing a way the old panel misled or
    simply could not be read:

    ``mask_diagonal`` drops the self-comparison cells from both the colour
    scaling and the annotation.  On a correlation matrix the diagonal is 1.0 by
    construction, so leaving it in pins the top of the scale at 1.0 and
    squeezes every informative off-diagonal value into the dark end of the
    colormap.  That is why a matrix spanning 0.977-0.993 -- a perfectly healthy
    result -- rendered as a black grid with a bright diagonal.

    Annotation colour is chosen per cell from the luminance of the colour
    underneath it, instead of a fixed ``#222``.  The old fixed dark grey was
    invisible on any cell in the lower half of a sequential colormap.

    ``robust`` scales to the 2nd/98th percentile of the plotted values, so a
    single outlier cell cannot flatten every other one.
    """
    if M.empty:
        annotate_empty(ax, "no data", cfg)
        return None

    A = M.to_numpy(dtype=float).copy()
    diag = None
    if mask_diagonal and A.shape[0] == A.shape[1] and A.shape[0] > 1:
        diag = np.diag(A).copy()
        np.fill_diagonal(A, np.nan)

    finite = A[np.isfinite(A)]
    if finite.size == 0:
        annotate_empty(ax, "no finite values to plot", cfg)
        return None

    if log_color:
        pos = finite[finite > 0]
        norm = LogNorm(
            vmin=max(float(pos.min()) if pos.size else 1e-3, 1e-12),
            vmax=float(finite.max()),
        )
    else:
        if robust and finite.size > 4:
            lo_d, hi_d = (float(x) for x in np.nanpercentile(finite, [2, 98]))
        else:
            lo_d, hi_d = float(finite.min()), float(finite.max())
        lo = lo_d if vmin is None else float(vmin)
        hi = hi_d if vmax is None else float(vmax)
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:                      # constant matrix
            lo, hi = lo - 0.5, hi + 0.5
        norm = Normalize(vmin=lo, vmax=hi)

    cm = cmap or cfg.continuous_cmap
    if isinstance(cm, str):
        try:
            cm = plt.get_cmap(cm)
        except Exception:
            cm = plt.get_cmap("viridis")
    cm = cm.copy()
    cm.set_bad("#F0F0F0")                 # masked diagonal / missing cells

    im = ax.imshow(A, aspect="auto", cmap=cm, norm=norm,
                   interpolation="nearest")

    xrot, xha = tick_layout(M.columns)
    ax.set_xticks(np.arange(M.shape[1]))
    ax.set_xticklabels([shorten(c) for c in M.columns], rotation=xrot, ha=xha,
                       fontsize=fs(cfg, "tick"), fontweight="bold")
    ax.set_yticks(np.arange(M.shape[0]))
    ax.set_yticklabels([shorten(i) for i in M.index],
                       fontsize=fs(cfg, "tick"), fontweight="bold")
    ax.grid(False)

    if annotate and A.size <= max_annot_cells:
        asize = fs(cfg, "annot")
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                v = A[i, j]
                if np.isfinite(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center",
                            fontsize=asize, fontweight="bold",
                            color=text_on(cm, norm, v))
                elif diag is not None and i == j and np.isfinite(diag[i]):
                    # State what the masked cell held rather than leaving a
                    # blank square the reader has to interpret.
                    ax.text(j, i, fmt.format(diag[i]), ha="center", va="center",
                            fontsize=round(asize * 0.9, 1), color="#8A8A8A",
                            style="italic")

    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=fs(cfg, "cbar"))
    if cbar_fmt:
        cb.ax.yaxis.set_major_formatter(StrMethodFormatter(cbar_fmt))
    if cbar_label:
        cb.set_label(cbar_label, fontsize=fs(cfg, "cbar"), fontweight="bold")
    return im


def ridge(
    ax: plt.Axes,
    values_by_row: dict[str, np.ndarray],
    cfg: FigureConfig,
    xlabel: str = "",
    vline: float | None = None,
    bins: int = 80,
) -> None:
    """Ridge (joy) plot: one smoothed histogram per row, vertically offset."""
    rows = list(values_by_row)
    if not rows:
        annotate_empty(ax, "no data", cfg)
        return
    colors = palette(cfg, len(rows))
    allv = np.concatenate([np.asarray(v, float).ravel() for v in values_by_row.values()])
    allv = allv[np.isfinite(allv)]
    if allv.size < 3:
        annotate_empty(ax, "too few finite values", cfg)
        return
    edges = np.linspace(allv.min(), allv.max(), bins)
    centres = (edges[:-1] + edges[1:]) / 2
    for i, (label, c) in enumerate(zip(rows, colors)):
        v = np.asarray(values_by_row[label], float).ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        h, _ = np.histogram(v, bins=edges, density=True)
        h = h / (h.max() or 1.0) * 0.85
        ax.fill_between(centres, i, i + h, color=c, alpha=0.75, lw=0.6,
                        edgecolor="white")
    if vline is not None and np.isfinite(vline):
        ax.axvline(vline, color="#C44E52", ls="--", lw=1.1)
    ax.set_yticks(np.arange(len(rows)) + 0.35)
    ax.set_yticklabels([shorten(r, 26) for r in rows],
                       fontsize=fs(cfg, "tick"), fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0)


def hierarchical_order(M: np.ndarray) -> list[int]:
    """Leaf order from average-linkage clustering of a similarity matrix.

    Uses scipy when available; otherwise a dependency-free greedy
    nearest-neighbour seriation, which is not identical to a dendrogram order
    but groups similar rows adjacently, which is all the plot needs.  The point
    is that the figure should not fail to render because scipy is absent.
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    if n <= 2:
        return list(range(n))
    D = 1.0 - np.nan_to_num(M, nan=0.0)
    np.fill_diagonal(D, 0.0)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        D = (D + D.T) / 2.0
        np.fill_diagonal(D, 0.0)
        return list(leaves_list(linkage(squareform(D, checks=False), "average")))
    except Exception:
        remaining = set(range(n))
        cur = int(np.argmin(D.sum(axis=1)))
        order = [cur]
        remaining.discard(cur)
        while remaining:
            nxt = min(remaining, key=lambda j: D[cur, j])
            order.append(nxt)
            remaining.discard(nxt)
            cur = nxt
        return order


def triangular_similarity(
    ax: plt.Axes,
    upper: pd.DataFrame,
    lower: pd.DataFrame,
    labels: Sequence[str],
    cfg: FigureConfig,
    upper_label: str = "Jaccard",
    lower_label: str = "Spearman",
    upper_sqrt: bool = True,
    upper_vmax: float | None = None,
) -> None:
    """Two similarity matrices in one square: upper and lower triangles.

    The collaborator's design, and a good one -- it puts set overlap and
    profile correlation side by side for the same pairs, so a pair that shares
    DEGs but disagrees on direction is immediately visible.
    """
    n = len(labels)
    if n == 0:
        annotate_empty(ax, "no perturbations", cfg)
        return
    U = upper.to_numpy(dtype=float).copy()
    L = lower.to_numpy(dtype=float).copy()
    if upper_sqrt:
        U = np.sqrt(np.clip(U, 0, 1))

    iu = np.triu_indices(n, k=1)
    il = np.tril_indices(n, k=-1)

    canvas_u = np.full((n, n), np.nan)
    canvas_u[iu] = U[iu]
    canvas_l = np.full((n, n), np.nan)
    canvas_l[il] = L[il]

    # aspect="auto" so the matrix fills the axes width. With aspect="equal" the
    # image is letterboxed inside the axes, which silently breaks alignment
    # with any marginal bar chart sharing the x range.
    # Jaccard overlaps between perturbation DEG sets are almost always small,
    # so a fixed 0..1 scale rendered the entire upper triangle as one shade of
    # pale yellow -- the same defect as pinning a correlation heatmap's vmax at
    # 1.0.  Scale to the observed spread and say so on the colourbar.
    u_off = U[iu][np.isfinite(U[iu])]
    u_hi = (float(upper_vmax) if upper_vmax is not None
            else (max(float(np.nanpercentile(u_off, 98)), 0.05)
                  if u_off.size else 1.0))
    im_u = ax.imshow(canvas_u, cmap="YlOrRd", vmin=0, vmax=u_hi, aspect="auto",
                     interpolation="nearest")
    im_l = ax.imshow(canvas_l, cmap=cfg.diverging_cmap, vmin=-1, vmax=1,
                     aspect="auto", interpolation="nearest")
    xrot, xha = tick_layout(labels)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([shorten(x, 22) for x in labels], rotation=xrot, ha=xha,
                       fontsize=fs(cfg, "tick"), fontweight="bold")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([shorten(x, 22) for x in labels],
                       fontsize=fs(cfg, "tick"), fontweight="bold")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.grid(False)
    for k in range(n):
        ax.plot([k - 0.5, k + 0.5], [k - 0.5, k - 0.5], color="#bbb", lw=0.6)

    # Two colorbars in explicitly positioned axes. Stacking them with
    # `fraction`/`pad` on the same parent makes them overlap once the figure is
    # tight_layout'd, which is what happened here.
    fig = ax.figure
    box = ax.get_position()
    w, gap = 0.018, 0.012
    cax_u = fig.add_axes([box.x1 + gap, box.y0 + box.height * 0.55,
                          w, box.height * 0.40])
    cax_l = fig.add_axes([box.x1 + gap, box.y0 + box.height * 0.05,
                          w, box.height * 0.40])
    cb1 = fig.colorbar(im_u, cax=cax_u)
    cb1.set_label(f"{upper_label}{' (sqrt)' if upper_sqrt else ''}",
                  fontsize=fs(cfg, "cbar"), fontweight="bold")
    cb1.ax.tick_params(labelsize=fs(cfg, "cbar"))
    cb2 = fig.colorbar(im_l, cax=cax_l)
    cb2.set_label(lower_label, fontsize=fs(cfg, "cbar"), fontweight="bold")
    cb2.ax.tick_params(labelsize=fs(cfg, "cbar"))
