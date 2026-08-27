"""
Numerical kernels for the Perturb-seq pipeline.

Everything in this module operates on numpy arrays and pandas objects only --
no AnnData, no scanpy, no scipy required (scipy is used opportunistically
where it is available and materially better, always behind a fallback).

That is deliberate.  In the original pipeline the maths was tangled up with
AnnData plumbing and module-level globals, which made it effectively
untestable; a wrong sign or a mis-specified axis could not be caught without
running the whole notebook against real data.  Here every formula is a pure
function with a docstring stating what it computes, and ``tests/test_stats.py``
checks each against an independently-derived expected value.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - environment dependent
    from scipy import stats as _scipy_stats
    from scipy.sparse import issparse as _issparse
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_stats = None
    _HAVE_SCIPY = False

    def _issparse(x) -> bool:  # type: ignore
        return hasattr(x, "toarray") and hasattr(x, "tocsr")


# ===========================================================================
# Array helpers
# ===========================================================================
def to_dense(X) -> np.ndarray:
    """Return a dense float64 2-D ndarray from a dense or sparse matrix.

    WARNING: densifying a whole single-cell expression matrix is almost always
    a mistake. A 200,000 x 25,000 matrix is 40 GB dense and about 3 GB sparse.
    Use this only on something already known to be small -- a guide or hashtag
    matrix, or a slice produced by ``take_columns`` / ``take_rows``.
    """
    if _issparse(X):
        X = X.toarray()
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(np.float64, copy=False)


def is_sparse(X) -> bool:
    return bool(_issparse(X))


def nbytes_dense(X) -> int:
    """Bytes this matrix would occupy if densified as float64."""
    shape = getattr(X, "shape", None)
    if not shape or len(shape) != 2:
        return 0
    return int(shape[0]) * int(shape[1]) * 8


def take_columns(X, idx) -> np.ndarray:
    """Dense ``(n_cells, len(idx))`` block, sparse-safe.

    The workhorse for "give me these few genes for every cell". Densifies only
    the requested columns, so cost is bounded by ``len(idx)`` rather than by
    the width of the matrix.
    """
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        rows = int(getattr(X, "shape", (0, 0))[0])
        return np.zeros((rows, 0), dtype=np.float64)
    if _issparse(X):
        return np.asarray(X[:, idx].todense(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)[:, idx]


def take_column(X, j: int) -> np.ndarray:
    """Dense 1-D vector for a single gene."""
    return take_columns(X, [int(j)]).ravel()


def take_rows(X, mask):
    """Row subset, preserving sparsity."""
    mask = np.asarray(mask)
    idx = np.flatnonzero(mask) if mask.dtype == bool else mask.astype(int)
    return X[idx]


def row_sums(X) -> np.ndarray:
    if _issparse(X):
        return np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    return np.asarray(X, dtype=np.float64).sum(axis=1)


def col_means(X) -> np.ndarray:
    if _issparse(X):
        return np.asarray(X.mean(axis=0)).ravel().astype(np.float64)
    return np.asarray(X, dtype=np.float64).mean(axis=0)


def col_variances(X) -> np.ndarray:
    """Per-column variance without densifying (E[x^2] - E[x]^2)."""
    if _issparse(X):
        n = X.shape[0]
        mean = np.asarray(X.mean(axis=0)).ravel()
        sq = X.copy()
        sq.data = sq.data ** 2
        mean_sq = np.asarray(sq.sum(axis=0)).ravel() / max(n, 1)
        return np.maximum(mean_sq - mean**2, 0.0)
    A = np.asarray(X, dtype=np.float64)
    return A.var(axis=0)


def col_nonzero_fraction(X) -> np.ndarray:
    if _issparse(X):
        n = X.shape[0]
        counts = np.asarray((X != 0).sum(axis=0)).ravel()
        return counts / max(n, 1)
    A = np.asarray(X)
    return (A != 0).mean(axis=0)


def to_csc(X):
    """Column-oriented sparse form, for repeated column slicing."""
    if _issparse(X):
        return X.tocsc() if X.format != "csc" else X
    return X


def to_csr(X):
    """Row-oriented sparse form, for repeated row slicing."""
    if _issparse(X):
        return X.tocsr() if X.format != "csr" else X
    return X


def sparse_log1p(X):
    """log1p preserving sparsity (log1p(0) == 0, so zeros stay zero)."""
    if _issparse(X):
        out = X.copy()
        out.data = np.log1p(out.data)
        return out
    return np.log1p(np.asarray(X, dtype=np.float64))


def normalize_rows(X, target_sum: float):
    """Scale each cell to a fixed total, preserving sparsity.

    The sparse path scales ``X.data`` directly rather than going through
    ``diags(scale) @ X``. Mathematically the two are identical, but
    ``diags(scale) @ X`` builds an n x n sparse diagonal matrix and runs a
    general sparse-sparse matrix multiply to do what is really just "multiply
    each row's nonzero entries by that row's scale factor" -- on a matrix with
    hundreds of millions of nonzeros (a reused, already-analysed h5ad easily
    gets here with its full gene set, not just the HVG subset) that overhead
    is the difference between seconds and tens of minutes. This is exactly
    what scanpy's own ``normalize_total`` does internally, which is why the
    fresh-computation path (going through scanpy) was never slow here while
    the reused-embedding path (going through this function) was.
    """
    totals = row_sums(X)
    totals[totals == 0] = 1.0
    scale = target_sum / totals
    if _issparse(X):
        X = X.tocsr(copy=True)
        row_nnz = np.diff(X.indptr)
        X.data = X.data * np.repeat(scale, row_nnz)
        return X
    return np.asarray(X, dtype=np.float64) * scale[:, None]


def safe_divide(num, den, fill: float = np.nan):
    """Elementwise division that yields ``fill`` wherever the denominator is
    zero or non-finite, instead of raising or emitting inf/nan warnings.

    The original pipeline guarded divisions with ``if total else nan``, which
    is wrong for NaN denominators: ``bool(np.nan)`` is ``True``, so a missing
    metric sailed straight through the guard.
    """
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.full(np.broadcast(num, den).shape, float(fill), dtype=np.float64)
    ok = np.isfinite(den) & (den != 0) & np.isfinite(num)
    np.divide(num, den, out=out, where=ok)
    out[~ok] = fill
    return out if out.ndim else float(out)


# ===========================================================================
# Robust thresholds
# ===========================================================================
def median_abs_deviation(x: np.ndarray, scale_to_normal: bool = True) -> float:
    """Median absolute deviation.

    With ``scale_to_normal`` the result is multiplied by 1.4826 so that, for
    normally distributed data, one MAD equals one standard deviation.  This is
    what makes "3 MAD" mean the same thing as "3 sigma" for clean data while
    staying robust to outliers.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    mad = float(np.median(np.abs(x - med)))
    return mad * 1.4826 if scale_to_normal else mad


def mad_bounds(
    x: np.ndarray,
    n_mad_lower: float = 3.0,
    n_mad_upper: float = 5.0,
    log: bool = True,
) -> tuple[float, float]:
    """Robust (lower, upper) bounds at ``n_mad_*`` MADs from the median.

    ``log=True`` computes the bounds on log1p-transformed values and maps them
    back, which is the right thing for count-like data: UMI and gene counts
    are roughly log-normal, so a symmetric rule on the raw scale produces a
    negative lower bound and a far-too-tight upper bound.

    Asymmetric multipliers are the default because, as the collaborator's
    report correctly notes, a symmetric MAD rule "over-trims the healthy
    high-RNA tail".
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    v = np.log1p(x) if log else x
    med = float(np.median(v))
    mad = median_abs_deviation(v)
    if not np.isfinite(mad) or mad == 0:
        # Degenerate spread: fall back to percentiles so we still return
        # something usable rather than (med, med), which would filter
        # everything.
        lo_v, hi_v = np.percentile(v, [1, 99])
    else:
        lo_v, hi_v = med - n_mad_lower * mad, med + n_mad_upper * mad
    if log:
        return float(np.expm1(lo_v)), float(np.expm1(hi_v))
    return float(lo_v), float(hi_v)


def otsu_threshold(x: np.ndarray, n_bins: int = 256) -> float:
    """Otsu's method: the cut that minimises within-class variance.

    Used as a dependency-free alternative to k-means for splitting a bimodal
    1-D distribution (hashtag CLR values, guide UMI counts).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    lo, hi = float(np.min(x)), float(np.max(x))
    if not np.isfinite(lo) or hi <= lo:
        return float(lo)
    hist, edges = np.histogram(x, bins=n_bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centres = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return float(lo)
    w0 = np.cumsum(hist) / total
    w1 = 1.0 - w0
    csum = np.cumsum(hist * centres)
    mu_t = csum[-1] / total
    mu0 = safe_divide(csum, np.cumsum(hist), fill=0.0)
    mu1 = safe_divide(mu_t * total - csum, (total - np.cumsum(hist)), fill=0.0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = -np.inf
    if not np.isfinite(between).any():
        return float(lo)

    # When the two modes are well separated there is no data between them, so
    # the between-class variance is *identical* for every cut inside the gap --
    # a plateau, not a peak. A bare argmax then returns the plateau's left edge,
    # i.e. a threshold hugging the background mode rather than sitting in the
    # valley. Taking the middle of the plateau puts the cut where a human would.
    best = float(np.max(between))
    tol = max(abs(best) * 1e-9, 1e-12)
    plateau = np.flatnonzero(between >= best - tol)
    return float(centres[int(np.median(plateau))])


def split_bimodal_1d(
    x: np.ndarray, random_state: int = 0, max_iter: int = 100
) -> tuple[np.ndarray, float, float]:
    """Split a 1-D distribution into background/signal.

    Returns ``(background_mask, background_mean, signal_mean)``.

    Uses 1-D k-means (k=2) seeded deterministically at the 10th and 90th
    percentiles.  Deterministic seeding matters: the original used
    scikit-learn KMeans with a random seed on skewed, zero-inflated hashtag
    data, where the result can flip between runs.  Percentile seeding on
    sorted 1-D data converges to the same answer every time.

    A degenerate (constant) input is reported as all-background, which
    correctly prevents spurious positive calls.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    finite = np.isfinite(x)
    if finite.sum() == 0 or np.allclose(x[finite], x[finite][0]):
        return np.ones(x.shape, dtype=bool), float("nan"), float("nan")

    v = x[finite]
    c0, c1 = np.percentile(v, [10, 90])
    if c0 == c1:
        c0, c1 = v.min(), v.max()
    for _ in range(max_iter):
        mid = (c0 + c1) / 2.0
        lo_mask = v <= mid
        if lo_mask.all() or (~lo_mask).all():
            break
        n0, n1 = v[lo_mask].mean(), v[~lo_mask].mean()
        if np.isclose(n0, c0) and np.isclose(n1, c1):
            c0, c1 = n0, n1
            break
        c0, c1 = n0, n1

    mid = (c0 + c1) / 2.0
    bg = np.zeros(x.shape, dtype=bool)
    bg[finite] = v <= mid
    bg[~finite] = True          # non-finite cells cannot be called positive
    return bg, float(c0), float(c1)


# ===========================================================================
# Hashtag normalisation
# ===========================================================================
def clr_by_feature(X) -> np.ndarray:
    """log1p, then centre each *feature* by its mean across cells.

    Per-feature normalisation (each HTO's own counts divided by that HTO's own
    geometric mean across cells) is the ORIGINAL Cell Hashing definition --
    Stoeckius et al. 2018, Genome Biology, Methods: "HTO raw counts were
    normalized using centered log ratio (CLR) transformation, where counts
    were divided by the geometric mean of an HTO across cells." That is the
    same axis as Seurat's ``margin = 2``.

    Earlier versions of this docstring justified this axis by citing "what the
    WNN/ADT tutorial uses" and implying it is also what the canonical HTODemux
    hashing vignette runs. That second claim is wrong: the hashing vignette
    calls ``NormalizeData(assay = "HTO", normalization.method = "CLR")``
    without setting ``margin``, which defaults to ``margin = 1`` (per-cell).
    ``margin = 2`` is a separate, later Seurat convention recommended
    specifically for ADT/protein visualisation (the multimodal/WNN vignette),
    not for HTO demultiplexing. The per-feature axis used here is still
    correct -- it matches the original paper's own formula -- but on its own
    authority, not by attribution to a vignette that actually runs the other
    axis by default. Hashtags differ in overall abundance by an order of
    magnitude, and a per-feature threshold rule needs one axis that is
    meaningful across all of them, which per-feature normalisation gives.

    It is **not the same transform** as Seurat's CLR, and earlier versions of
    this docstring said it was. Seurat computes
    ``log1p(x / exp(sum(log1p(x[x > 0])) / length(x)))``: the division happens
    on the raw scale, inside the log, and the geometric mean sums ``log1p``
    over the nonzero entries only while dividing by the count of all of them.
    This function computes ``log1p(x) - mean(log1p(x))``, i.e.
    ``log((1 + x) / G)`` with ``G`` the geometric mean of ``1 + x`` over every
    cell. Two visible consequences: Seurat's output is >= 0 and maps zero
    counts to exactly 0, whereas this maps them to ``-mu``.

    See ``clr_true_seurat`` for the Seurat formula and ``clr_by_cell`` for the
    compositional definition. In practice all three are monotone in the raw
    count for a fixed feature, so a per-feature threshold rule produces the
    same *kind* of partition under any of them -- what moves is where the
    cutoff lands. ``hto.compare_normalisations`` quantifies that on real data.

    Input is cells x features.
    """
    X = to_dense(X)
    logX = np.log1p(np.clip(X, 0, None))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu = np.nanmean(logX, axis=0, keepdims=True)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return logX - mu


def clr_true_seurat(X, margin: int = 2) -> np.ndarray:
    """Seurat's actual CLR, for comparison against ``clr_by_feature``.

    Reproduces ``NormalizeData(normalization.method = "CLR")``::

        clr_function <- function(x) {
          log1p(x = x / exp(x = sum(log1p(x = x[x > 0]), na.rm = TRUE)
                                   / length(x = x)))
        }

    ``margin=2`` applies it down each feature across cells (matches the
    original Stoeckius et al. 2018 Cell Hashing formula, and Seurat's own
    ADT/protein-visualisation recommendation in the multimodal/WNN vignette);
    ``margin=1`` applies it across features within each cell, which is what
    the canonical HTODemux hashing vignette actually runs, since its
    ``NormalizeData(..., normalization.method = "CLR")`` call does not
    override ``margin`` and Seurat's default is 1. The two vignettes use
    different axes for different purposes -- neither is "more correct" in the
    abstract, they answer different questions -- so pick the margin to match
    what you are trying to reproduce rather than assuming one is universal.

    Note the asymmetry in the geometric mean: the sum runs over nonzero entries
    only, the divisor is the length of the whole vector. That is Seurat's
    behaviour rather than an oversight here -- it is a known oddity in their
    tracker -- and it is reproduced faithfully so the comparison is against
    what people actually run, not against a tidied-up version of it.

    Input is cells x features. Output is >= 0, with zero counts mapping to 0.
    """
    X = np.asarray(to_dense(X), dtype=float)
    X = np.clip(X, 0, None)
    axis = 0 if int(margin) == 2 else 1

    def _one(v: np.ndarray) -> np.ndarray:
        n = v.size
        if n == 0:
            return v
        nz = v[v > 0]
        if nz.size == 0:
            # No signal at all: dividing by exp(0) == 1 leaves log1p(0) == 0.
            return np.zeros_like(v)
        g = np.exp(float(np.sum(np.log1p(nz))) / float(n))
        if not np.isfinite(g) or g <= 0:
            return np.log1p(v)
        return np.log1p(v / g)

    return np.apply_along_axis(_one, axis, X)




def clr_by_cell(X) -> np.ndarray:
    """True compositional CLR: per cell, log(x_i) minus the mean log across
    features for that cell.

    Provided for completeness and comparison; ``clr_by_feature`` is the
    default for demultiplexing.  Input is cells x features.
    """
    X = to_dense(X)
    logX = np.log1p(np.clip(X, 0, None))
    mu = logX.mean(axis=1, keepdims=True)
    return logX - mu


#: Selectable hashtag normalisations, keyed by ``HTOConfig.normalisation``.
#:
#: The default is unchanged from every previous version -- this exists so the
#: choice is visible and testable rather than implicit, and so the alternatives
#: can be compared on real data instead of argued about.
HTO_NORMALISATIONS = {
    "mean_centred_log1p": clr_by_feature,
    "seurat_clr": clr_true_seurat,
    "compositional": clr_by_cell,
}


# ===========================================================================
# Guide purity
# ===========================================================================
@dataclass
class GuideStats:
    """Per-cell guide statistics, all computed in a single pass.

    Attributes
    ----------
    total : total guide UMIs in the cell
    top1, top2 : largest and second-largest guide UMI counts
    top1_over_top2 : top1 / (top1 + top2), in percent -- "is there a clear
        winner between the best two guides?"  This is the quantity the
        assignment rule uses (>75% by default).
    top1_over_total : top1 / total, fraction -- "does the winner dominate the
        whole cell?"
    top12_over_total : (top1 + top2) / total, fraction.
    n_detected : number of guides with >0 UMIs (the MOI proxy).
    top1_index : column index of the winning guide, -1 if the cell has no
        guide reads at all.
    """

    total: np.ndarray
    top1: np.ndarray
    top2: np.ndarray
    top1_over_top2: np.ndarray
    top1_over_total: np.ndarray
    top12_over_total: np.ndarray
    n_detected: np.ndarray
    top1_index: np.ndarray

    def to_frame(self, index: Sequence[str] | None = None) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "guide_total_umis": self.total,
                "guide_top1_umis": self.top1,
                "guide_top2_umis": self.top2,
                "guide_purity_pct": self.top1_over_top2,
                "guide_top1_over_total": self.top1_over_total,
                "guide_top12_over_total": self.top12_over_total,
                "n_guides_detected": self.n_detected,
            }
        )
        if index is not None:
            df.index = pd.Index(list(index))
        return df


def compute_guide_stats(X) -> GuideStats:
    """Compute all per-cell guide statistics from a cells x guides count matrix.

    A cell with exactly one guide detected gets ``top1_over_top2 == 100``
    (top2 is 0), which is the desired behaviour -- a single guide is maximally
    pure.  A cell with *no* guide reads gets NaN for the ratios rather than
    0 or 100, so it can be excluded rather than silently counted as impure.
    """
    X = to_dense(X)
    X = np.clip(X, 0, None)
    n_cells, n_guides = X.shape

    total = X.sum(axis=1)
    n_detected = (X > 0).sum(axis=1)

    if n_guides == 0:
        nan = np.full(n_cells, np.nan)
        return GuideStats(
            total=np.zeros(n_cells), top1=np.zeros(n_cells), top2=np.zeros(n_cells),
            top1_over_top2=nan, top1_over_total=nan, top12_over_total=nan,
            n_detected=np.zeros(n_cells, dtype=int),
            top1_index=np.full(n_cells, -1, dtype=int),
        )

    top1_index = np.argmax(X, axis=1)
    top1 = X[np.arange(n_cells), top1_index]
    if n_guides == 1:
        top2 = np.zeros(n_cells)
    else:
        # Partition once and read the two largest values.  np.partition with a
        # single kth guarantees element kth is in its sorted position, so
        # [-1] is the true max and [-2] the true second max.
        part = np.partition(X, n_guides - 2, axis=1)
        top2 = part[:, n_guides - 2]
        # Guard: if there are ties, partition can place an equal value; the
        # relation top2 <= top1 must still hold.
        top2 = np.minimum(top2, top1)

    has_reads = total > 0
    top12 = top1 + top2
    top1_over_top2 = np.full(n_cells, np.nan)
    np.divide(100.0 * top1, top12, out=top1_over_top2, where=top12 > 0)
    top1_over_top2[~has_reads] = np.nan

    top1_over_total = np.full(n_cells, np.nan)
    np.divide(top1, total, out=top1_over_total, where=has_reads)
    top12_over_total = np.full(n_cells, np.nan)
    np.divide(top12, total, out=top12_over_total, where=has_reads)

    idx = np.where(has_reads, top1_index, -1)
    return GuideStats(
        total=total, top1=top1, top2=top2,
        top1_over_top2=top1_over_top2,
        top1_over_total=top1_over_total,
        top12_over_total=top12_over_total,
        n_detected=n_detected.astype(int),
        top1_index=idx.astype(int),
    )


def assign_guides(
    stats: GuideStats,
    guide_names: Sequence[str],
    min_reads: int = 10,
    purity_min: float = 75.0,
) -> pd.DataFrame:
    """Apply the assignment rule to precomputed guide statistics.

    Rule: a guide is assigned to a cell when the cell has ``> min_reads``
    total guide UMIs **and** ``top1/(top1+top2) > purity_min`` percent.  This
    matches the collaborator's stated criterion exactly.

    Returns a DataFrame with ``eligible``, ``is_assigned`` and
    ``assigned_guide`` (pandas NA where unassigned).
    """
    names = np.asarray(list(guide_names), dtype=object)
    eligible = stats.total > min_reads
    pure = np.nan_to_num(stats.top1_over_top2, nan=-1.0) > purity_min
    assigned = eligible & pure & (stats.top1_index >= 0)

    guide = np.full(stats.total.shape, None, dtype=object)
    if names.size:
        sel = assigned & (stats.top1_index >= 0)
        guide[sel] = names[stats.top1_index[sel]]

    return pd.DataFrame(
        {
            "guide_eligible": eligible,
            "guide_is_assigned": assigned,
            "assigned_guide": pd.Series(guide, dtype="object"),
        }
    )


def assignment_sweep(
    stats: GuideStats,
    purity_thresholds: Iterable[float],
    min_reads: int = 10,
) -> pd.DataFrame:
    """Fraction of cells assigned as a function of the purity cut-off.

    Vectorised over thresholds -- the original recomputed the whole
    assignment inside a Python loop over 101 thresholds, per group.
    """
    eligible = stats.total > min_reads
    n_cells = stats.total.size
    purity = np.nan_to_num(stats.top1_over_top2, nan=-1.0)
    rows = []
    for t in purity_thresholds:
        n_assigned = int(np.sum(eligible & (purity > t)))
        rows.append(
            {
                "purity_threshold": float(t),
                "n_assigned": n_assigned,
                "frac_assigned": n_assigned / n_cells if n_cells else np.nan,
                "n_eligible": int(eligible.sum()),
                "frac_of_eligible": (
                    n_assigned / int(eligible.sum()) if eligible.sum() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


# ===========================================================================
# Energy distance ("E-distance")
# ===========================================================================
def _mean_sq_dist_within(A: np.ndarray) -> float:
    """Mean squared Euclidean distance between distinct pairs within A.

    Closed form: for n points, mean_{i != j} ||a_i - a_j||^2
                 = 2n/(n-1) * mean_i ||a_i - mean(A)||^2
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if n < 2:
        return float("nan")
    centred = A - A.mean(axis=0, keepdims=True)
    m2 = float(np.mean(np.sum(centred**2, axis=1)))
    return 2.0 * n / (n - 1) * m2


def _mean_sq_dist_between(A: np.ndarray, B: np.ndarray) -> float:
    """Mean squared Euclidean distance between all cross pairs.

    Closed form: ||mean(A) - mean(B)||^2 + var(A) + var(B), where var is the
    mean squared distance to the group's own centroid.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float("nan")
    da = A - A.mean(axis=0, keepdims=True)
    db = B - B.mean(axis=0, keepdims=True)
    m2a = float(np.mean(np.sum(da**2, axis=1)))
    m2b = float(np.mean(np.sum(db**2, axis=1)))
    delta = A.mean(axis=0) - B.mean(axis=0)
    return float(delta @ delta) + m2a + m2b


def energy_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Squared-distance energy statistic between two point clouds.

        E = 2 * E[||a - b||^2] - E[||a - a'||^2] - E[||b - b'||^2]

    This is the "E-distance" used throughout the Perturb-seq / scPerturb
    literature, computed in PCA space.  Note it uses *squared* Euclidean
    distances, which is what makes the closed-form identities above exact; the
    classical Szekely-Rizzo energy distance uses unsquared norms and is a
    different quantity.  The original pipeline computed this same statistic but
    documented it as "energy distance" without the qualifier, which is the kind
    of ambiguity that gets numbers misinterpreted downstream.

    With squared distances the statistic reduces to
    ``2 * ||mean(A) - mean(B)||^2 + (bias terms)``; it is zero for identical
    distributions only in the mean/dispersion sense, not in full distribution.
    """
    within_a = _mean_sq_dist_within(A)
    within_b = _mean_sq_dist_within(B)
    cross = _mean_sq_dist_between(A, B)
    if not all(np.isfinite([within_a, within_b, cross])):
        return float("nan")
    return float(2.0 * cross - within_a - within_b)


def edistance_permutation_pvalue(
    A: np.ndarray,
    B: np.ndarray,
    n_perm: int = 1000,
    random_state: int = 0,
) -> tuple[float, float]:
    """E-distance plus a label-permutation p-value.

    Returns ``(edistance, p_value)``.  The original reported bare E-distances
    with no null, which makes them impossible to interpret across
    perturbations with very different cell numbers -- E-distance is biased
    upward at small n, so a weak perturbation with 12 cells can outrank a
    strong one with 900.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    obs = energy_distance(A, B)
    if not np.isfinite(obs) or n_perm <= 0:
        return obs, float("nan")
    pooled = np.vstack([A, B])
    n_a = A.shape[0]
    rng = np.random.default_rng(random_state)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled.shape[0])
        pa = pooled[perm[:n_a]]
        pb = pooled[perm[n_a:]]
        if energy_distance(pa, pb) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


# ===========================================================================
# Differential expression
# ===========================================================================
def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared.  Equivalent to scipy.stats.rankdata."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    # Average ties
    sx = x[order]
    i = 0
    while i < sx.size:
        j = i
        while j + 1 < sx.size and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def rank_columns(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average ranks down each column, plus each column's tie term.

    Column-wise equivalent of :func:`rankdata`, with no Python loop over
    columns. Returns ``(ranks, tie_term)`` where ``tie_term`` is
    ``sum(t**3 - t)`` over each column's tie groups -- the quantity the
    tie-corrected Mann-Whitney variance needs.
    """
    A = np.asarray(A, dtype=np.float64)
    n, k = A.shape
    if n == 0 or k == 0:
        return np.zeros((n, k)), np.zeros(k)

    order = np.argsort(A, axis=0, kind="stable")
    srt = np.take_along_axis(A, order, axis=0)

    pos = np.arange(n, dtype=np.float64)[:, None]
    # Run starts and ends within each sorted column.
    starts = np.ones((n, k), dtype=bool)
    starts[1:] = srt[1:] != srt[:-1]
    ends = np.ones((n, k), dtype=bool)
    ends[:-1] = srt[:-1] != srt[1:]

    # Position of the first / last element of the run each row belongs to.
    start_pos = np.maximum.accumulate(np.where(starts, pos, -np.inf), axis=0)
    end_pos = np.minimum.accumulate(
        np.where(ends, pos, np.inf)[::-1], axis=0
    )[::-1]

    # Average rank of a run spanning [start_pos, end_pos], 1-based.
    avg_sorted = (start_pos + end_pos) / 2.0 + 1.0
    ranks = np.empty((n, k), dtype=np.float64)
    np.put_along_axis(ranks, order, avg_sorted, axis=0)

    run_len = end_pos - start_pos + 1.0
    tie_term = np.where(starts, run_len ** 3 - run_len, 0.0).sum(axis=0)
    return ranks, tie_term


def mannwhitney_u_columns(
    Bg: np.ndarray, Br: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided Mann-Whitney U for every column at once.

    Identical statistic to :func:`mannwhitney_u` -- Wilcoxon's rank sum and
    Mann-Whitney's U are the same test, related by
    ``U1 = W1 - n1(n1+1)/2`` -- computed for all genes in one pass instead of
    one Python call per gene. This is what scanpy's
    ``rank_genes_groups(method='wilcoxon')`` and presto do, and it is the
    reason the per-gene loop this replaces took hours: 40 targets x 38,000
    genes is 1.5 million scipy calls, each ranking ~30,000 values.

    Tie correction and the continuity correction match scipy's defaults, so
    the p-values agree with the scalar path to floating point. That agreement
    is asserted in the test suite rather than assumed.
    """
    Bg = np.asarray(Bg, dtype=np.float64)
    Br = np.asarray(Br, dtype=np.float64)
    n1, k = Bg.shape
    n2 = Br.shape[0]
    if n1 == 0 or n2 == 0 or k == 0:
        return np.full(k, np.nan), np.full(k, np.nan)

    n = n1 + n2
    ranks, tie_term = rank_columns(np.vstack([Bg, Br]))
    r1 = ranks[:n1].sum(axis=0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    var = (
        n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
        if n > 1 else np.zeros(k)
    )

    # NaN for zero-variance columns, matching the sparse path: untestable is
    # not the same as not significant, and only NaN is excluded from the BH
    # family. See the note in ``mannwhitney_u_sparse_columns``.
    p = np.full(k, np.nan, dtype=np.float64)
    ok = var > 0
    if np.any(ok):
        num = u1[ok] - mu
        num = num - 0.5 * np.sign(num)          # scipy use_continuity=True
        z = num / np.sqrt(var[ok])
        p[ok] = np.clip(_erfc(np.abs(z) / np.sqrt(2.0)), 0.0, 1.0)
    return u1, p


def _csc_col_sums(data: np.ndarray, indptr: np.ndarray, k: int) -> np.ndarray:
    """Per-column sums of a CSC block, safe for empty columns.

    The obvious ``np.add.reduceat(data, indptr[:-1])`` is wrong: when the last
    column of the block is empty, ``indptr[k-1] == len(data)``, which reduceat
    rejects as out of bounds. It raised
    ``IndexError: index 191684 out-of-bounds in add.reduceat`` on the first
    real object whose gene block ended in an all-zero gene -- and 11,504 of
    38,402 genes in that object are detected in no cell, so such a block was
    close to certain.

    bincount over repeated column indices has no such edge case: an empty
    column simply contributes nothing and lands at zero.
    """
    counts = np.diff(indptr)
    if data.size == 0:
        return np.zeros(k, dtype=np.float64)
    col_idx = np.repeat(np.arange(k, dtype=np.intp), counts)
    return np.bincount(col_idx, weights=np.asarray(data, dtype=np.float64),
                       minlength=k)[:k]


def _rank_1d(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Average ranks and tie term for one vector, without a Python tie loop."""
    n = x.size
    if n == 0:
        return np.zeros(0), 0.0
    order = np.argsort(x, kind="stable")
    srt = x[order]
    pos = np.arange(n, dtype=np.float64)
    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = srt[1:] != srt[:-1]
    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = srt[:-1] != srt[1:]
    start_pos = np.maximum.accumulate(np.where(starts, pos, -np.inf))
    end_pos = np.minimum.accumulate(np.where(ends, pos, np.inf)[::-1])[::-1]
    avg_sorted = (start_pos + end_pos) / 2.0 + 1.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg_sorted
    run_len = end_pos - start_pos + 1.0
    tie = float(np.where(starts, run_len ** 3 - run_len, 0.0).sum())
    return ranks, tie


def mannwhitney_u_sparse_columns(
    Ag, Ar, n1: int, n2: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mann-Whitney U per column, exploiting the zeros.

    The dense column-wise version is correct but no faster than the per-gene
    loop it replaced: ranking a 21,000 x 500 block is dominated by the argsort
    and by memory traffic, and that cost does not go away just because the
    loop moved into numpy. Measured at 1.0x.

    Single-cell expression is ~96% zeros, and after log1p every zero is still
    zero, so they form one enormous tie group at the bottom of every column's
    ranking. Their average rank is ``(n_zero + 1) / 2`` by inspection, and only
    the nonzeros need sorting -- roughly 4% of the data. The Python loop over
    columns is then cheap, because each iteration sorts a few hundred values
    instead of tens of thousands. This is the same trick presto uses.

    Requires non-negative input, so that zeros really are the smallest values.
    The caller checks; log1p-normalised counts always satisfy it.
    """
    k = Ag.shape[1]
    n = n1 + n2
    u1 = np.full(k, np.nan)
    # NaN, not 1.0. A gene that carries no information -- all-zero in both
    # groups, or constant across every cell -- was never a testable hypothesis.
    # Emitting p = 1.0 for it puts it in the BH family, and
    # ``benjamini_hochberg`` excludes NaN but counts 1.0, so every other gene's
    # adjusted p is inflated by the ratio of screened to testable genes. On the
    # reference object above, 11,504 of 38,402 genes are detected in no cell,
    # which inflated every padj from this path by about 1.43x relative to the
    # dense path on identical data.
    pv = np.full(k, np.nan)
    if n1 == 0 or n2 == 0 or k == 0:
        return u1, pv

    Ag = Ag.tocsc()
    Ar = Ar.tocsc()
    mu = n1 * n2 / 2.0

    for j in range(k):
        gs, ge = Ag.indptr[j], Ag.indptr[j + 1]
        rs, re_ = Ar.indptr[j], Ar.indptr[j + 1]
        g = Ag.data[gs:ge]
        r = Ar.data[rs:re_]
        n_nz = g.size + r.size
        n_zero = n - n_nz
        if n_nz == 0:
            continue                     # all-zero column: no information

        ranks_nz, tie_nz = _rank_1d(np.concatenate([g, r]))
        ranks_nz = ranks_nz + n_zero     # zeros occupy the lowest n_zero ranks

        zeros_in_g = n1 - g.size
        r1 = zeros_in_g * (n_zero + 1) / 2.0 + float(ranks_nz[:g.size].sum())
        u = r1 - n1 * (n1 + 1) / 2.0
        u1[j] = u

        tie_term = tie_nz + float(n_zero) ** 3 - float(n_zero)
        var = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
        if var <= 0:
            continue
        num = u - mu
        num -= 0.5 * np.sign(num)        # scipy use_continuity=True
        z = num / np.sqrt(var)
        pv[j] = float(np.clip(_erfc(abs(z) / np.sqrt(2.0)), 0.0, 1.0))
    return u1, pv


def mannwhitney_u(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided Mann-Whitney U with tie-corrected normal approximation.

    Returns ``(U, p)``.  Uses scipy when available (exact for small n);
    otherwise the normal approximation, which is accurate for the group sizes
    that occur here (tens to thousands of cells).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    if _HAVE_SCIPY:
        try:
            res = _scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
            return float(res.statistic), float(res.pvalue)
        except Exception:  # pragma: no cover
            pass

    combined = np.concatenate([a, b])
    ranks = rankdata(combined)
    r1 = float(ranks[:n1].sum())
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    _, counts = np.unique(combined, return_counts=True)
    tie_term = float(np.sum(counts**3 - counts))
    var = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return float(u1), 1.0
    # Continuity correction, matching scipy's use_continuity=True default.
    # Without it this fallback and the scipy branch above return slightly
    # different p-values for the same data, so which one you got depended on
    # whether scipy happened to be installed.
    num = u1 - mu
    num -= 0.5 * np.sign(num)
    z = num / np.sqrt(var)
    # two-sided normal tail without scipy
    p = float(np.clip(2.0 * 0.5 * _erfc(abs(z) / np.sqrt(2.0)), 0.0, 1.0))
    return float(u1), p


def _erfc(x: float | np.ndarray) -> np.ndarray:
    """Complementary error function (vectorised), Abramowitz & Stegun 7.1.26.

    Max absolute error ~1.5e-7, which is far below the resolution at which
    these p-values are used (they are BH-adjusted and thresholded at 0.05).
    """
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t * np.exp(-ax * ax)
    erf = sign * y
    return 1.0 - erf


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (monotone, clipped at 1).

    NaN inputs stay NaN and are excluded from the ranking, rather than being
    silently treated as 1.0 (which would inflate everyone else's adjusted p).
    """
    p = np.asarray(p, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out


def spearman_matrix(M: np.ndarray) -> np.ndarray:
    """Column-wise Spearman correlation matrix of a genes x groups array.

    Implemented as Pearson on ranks, which is the definition, and is both
    faster and dependency-free versus looping scipy.stats.spearmanr over
    pairs.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError("spearman_matrix expects a 2-D array")
    R = np.column_stack([rankdata(M[:, j]) for j in range(M.shape[1])])
    R = R - R.mean(axis=0, keepdims=True)
    sd = np.sqrt((R**2).sum(axis=0))
    sd[sd == 0] = np.nan
    Rn = R / sd
    C = Rn.T @ Rn
    return np.clip(C, -1.0, 1.0)


def jaccard_matrix(sets: Sequence[set]) -> np.ndarray:
    """Pairwise Jaccard index between sets. Diagonal is 1 (0 for empty sets)."""
    n = len(sets)
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            a, b = sets[i], sets[j]
            union = len(a | b)
            J[i, j] = J[j, i] = (len(a & b) / union) if union else 0.0
    return J


@dataclass
class DEResult:
    """Differential expression of one group vs a reference."""

    table: pd.DataFrame        # gene, log2fc, pvalue, padj, mean_group,
                               # mean_ref, frac_detected_group, frac_detected_ref
    n_group: int
    n_ref: int


def differential_expression(
    X_group,
    X_ref,
    gene_names: Sequence[str],
    log_input: bool = True,
    min_frac_detected_ref: float = 0.10,
    pseudocount: float = 1e-9,
    block: int = 2000,
    rank_block_elements: int = 20_000_000,
) -> DEResult:
    """Per-gene log2 fold change and Mann-Whitney p-value, group vs reference.

    ``log_input=True`` (the default) means ``X_*`` hold log1p-normalised
    values.  In that case fold changes are computed on ``expm1``-inverted
    values, because the mean of logs is not the log of means -- averaging in
    log space understates the effect for genes with skewed expression.  The
    original pipeline got this right for target-gene knockdown but it is worth
    stating explicitly, because it is the single easiest thing to get wrong
    here.

    The Mann-Whitney test runs on the values as supplied; rank tests are
    invariant to the monotone log transform, so this is correct either way.

    Genes detected in fewer than ``min_frac_detected_ref`` of reference cells
    are still reported but flagged ``low_expression``, so they can be excluded
    from DEG calls without being silently deleted from the table.
    """
    genes = list(gene_names)
    n_genes = len(genes)
    for name, M in (("X_group", X_group), ("X_ref", X_ref)):
        if getattr(M, "shape", (0, 0))[1] != n_genes:
            raise ValueError(
                f"gene_names length {n_genes} does not match {name} "
                f"({getattr(M, 'shape', ('?', '?'))[1]})"
            )

    # Processed in gene blocks so peak memory is bounded by `block` columns
    # rather than by the full matrix. Densifying all genes for both groups at
    # once is what made this unusable on a real experiment: 20,000 control
    # cells x 25,000 genes is 4 GB per group, per perturbation.
    Xg_c = to_csc(X_group)
    Xr_c = to_csc(X_ref)

    mean_g = np.zeros(n_genes)
    mean_r = np.zeros(n_genes)
    frac_g = np.zeros(n_genes)
    frac_r = np.zeros(n_genes)
    pvals = np.full(n_genes, np.nan)

    n1 = int(X_group.shape[0])
    n2 = int(X_ref.shape[0])

    # The sparse path needs zeros to be the smallest values, so that they can
    # be treated as one tie group at the bottom of the ranking without being
    # sorted. True for log1p-normalised counts; checked rather than assumed,
    # because a scaled or residual matrix would silently break it.
    use_sparse = _issparse(Xg_c) and _issparse(Xr_c)
    if use_sparse:
        gmin = float(Xg_c.data.min()) if Xg_c.data.size else 0.0
        rmin = float(Xr_c.data.min()) if Xr_c.data.size else 0.0
        use_sparse = gmin >= 0.0 and rmin >= 0.0

    for start in range(0, n_genes, block):
        stop = min(start + block, n_genes)
        cols = np.arange(start, stop)

        if use_sparse:
            Sg = Xg_c[:, start:stop].tocsc()
            Sr = Xr_c[:, start:stop].tocsc()
            # expm1(0) == 0, so the inverse transform applies to the stored
            # values alone and the matrix stays sparse. Densifying here is
            # what made the old path allocate ~400 MB per block.
            gd = np.clip(np.expm1(Sg.data) if log_input else Sg.data, 0, None)
            rd = np.clip(np.expm1(Sr.data) if log_input else Sr.data, 0, None)
            k_blk = stop - start
            mean_g[start:stop] = _csc_col_sums(gd, Sg.indptr, k_blk) / n1
            mean_r[start:stop] = _csc_col_sums(rd, Sr.indptr, k_blk) / n2
            frac_g[start:stop] = np.diff(Sg.indptr) / n1
            frac_r[start:stop] = np.diff(Sr.indptr) / n2

            _u, p = mannwhitney_u_sparse_columns(Sg, Sr, n1, n2)
            pvals[start:stop] = p
            del Sg, Sr, gd, rd
            continue

        Bg = take_columns(Xg_c, cols)
        Br = take_columns(Xr_c, cols)

        lin_g = np.clip(np.expm1(Bg) if log_input else Bg, 0, None)
        lin_r = np.clip(np.expm1(Br) if log_input else Br, 0, None)

        mean_g[start:stop] = lin_g.mean(axis=0)
        mean_r[start:stop] = lin_r.mean(axis=0)
        frac_g[start:stop] = (Bg > 0).mean(axis=0)
        frac_r[start:stop] = (Br > 0).mean(axis=0)

        # Genes that are all-zero in both groups carry no information and have
        # zero tie-corrected variance; skip them rather than feed degenerate
        # columns through the ranking.
        informative = np.flatnonzero(~((mean_g[start:stop] == 0)
                                       & (mean_r[start:stop] == 0)))
        if informative.size:
            # Ranking materialises several (n1+n2) x k arrays. Sub-block so
            # peak memory stays bounded by an element budget rather than by
            # the caller's gene-block size -- with 26,000 control cells a
            # 2,000-gene block would otherwise need several GB.
            n_rows = Bg.shape[0] + Br.shape[0]
            per = max(1, int(rank_block_elements // max(n_rows, 1)))
            for s in range(0, informative.size, per):
                sel = informative[s:s + per]
                _u, p = mannwhitney_u_columns(Bg[:, sel], Br[:, sel])
                pvals[start + sel] = p

        del Bg, Br, lin_g, lin_r

    log2fc = np.log2((mean_g + pseudocount) / (mean_r + pseudocount))
    padj = benjamini_hochberg(pvals)
    table = pd.DataFrame(
        {
            "gene": genes,
            "log2fc": log2fc,
            "pvalue": pvals,
            "padj": padj,
            "mean_group": mean_g,
            "mean_ref": mean_r,
            "frac_detected_group": frac_g,
            "frac_detected_ref": frac_r,
            "low_expression": frac_r < min_frac_detected_ref,
        }
    )
    return DEResult(
        table=table,
        n_group=int(X_group.shape[0]),
        n_ref=int(X_ref.shape[0]),
    )


def select_degs(
    de_table: pd.DataFrame,
    padj_max: float = 0.05,
    abs_log2fc_min: float = 0.5,
    exclude_low_expression: bool = True,
) -> pd.DataFrame:
    """Apply the DEG definition: BH-padj < cut AND |log2FC| > cut.

    Matches the collaborator's rule, including the low-expression filter
    ("a gene must be detected in >=10% of the non-targeting control cells").
    """
    df = de_table
    mask = (df["padj"] < padj_max) & (df["log2fc"].abs() > abs_log2fc_min)
    if exclude_low_expression and "low_expression" in df.columns:
        mask &= ~df["low_expression"].astype(bool)
    return df.loc[mask].copy()


def rank_degs(
    degs: pd.DataFrame, top_n: int = 10, always_first: str | None = None
) -> pd.DataFrame:
    """Top-N DEGs ranked by significance, ties broken by |log2FC|.

    ``always_first`` (the perturbation's own target gene) is hoisted to the
    front if present anywhere in the DE table, which is what makes the
    per-perturbation dotplot readable: the first dot in each block is always
    the gene you meant to knock down.
    """
    df = degs.copy()
    df["_abs"] = df["log2fc"].abs()
    df = df.sort_values(["padj", "_abs"], ascending=[True, False])
    if always_first is not None and always_first in set(df["gene"]):
        own = df[df["gene"] == always_first]
        rest = df[df["gene"] != always_first].head(max(top_n - 1, 0))
        df = pd.concat([own, rest])
    else:
        df = df.head(top_n)
    return df.drop(columns="_abs")


# ===========================================================================
# Knockdown quantification
# ===========================================================================
def percent_knockdown(
    expr_perturbed: np.ndarray,
    expr_control: np.ndarray,
    log_input: bool = True,
) -> dict[str, float]:
    """Percent knockdown of one gene, perturbed vs control.

    ``pct_knockdown = (1 - mean_perturbed / mean_control) * 100`` on the
    *linear* scale.  Positive means suppressed.

    ``log_input`` inverts the log1p first: this is essential and is the most
    common error in knockdown quantification.  Taking means in log space
    systematically understates knockdown, because log1p compresses the high
    end where the control cells live.

    Returns pct_knockdown, log2fc, both means, cell counts and a
    Mann-Whitney p-value.  ``pct_knockdown`` is NaN when the control mean is
    zero (the gene is undetectable, so knockdown is undefined -- reporting 0
    or 100 there would be a fabrication).
    """
    a = np.asarray(expr_perturbed, dtype=np.float64).ravel()
    b = np.asarray(expr_control, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {
            "pct_knockdown": np.nan, "log2fc": np.nan,
            "mean_perturbed": np.nan, "mean_control": np.nan,
            "n_perturbed": int(a.size), "n_control": int(b.size),
            "pvalue": np.nan,
        }
    lin_a = np.expm1(a) if log_input else a
    lin_b = np.expm1(b) if log_input else b
    lin_a = np.clip(lin_a, 0, None)
    lin_b = np.clip(lin_b, 0, None)
    ma, mb = float(lin_a.mean()), float(lin_b.mean())
    pct = (1.0 - ma / mb) * 100.0 if mb > 0 else np.nan
    l2 = float(np.log2((ma + 1e-9) / (mb + 1e-9)))
    _, p = mannwhitney_u(a, b)
    return {
        "pct_knockdown": pct, "log2fc": l2,
        "mean_perturbed": ma, "mean_control": mb,
        "n_perturbed": int(a.size), "n_control": int(b.size),
        "pvalue": p,
    }


def resampling_test(
    expr_perturbed: np.ndarray,
    expr_control: np.ndarray,
    n_resample: int = 2000,
    random_state: int = 0,
    log_input: bool = True,
) -> dict[str, float]:
    """Calibrated knockdown test by resampling control cells.

    This is a lightweight stand-in for SCEPTRE's conditional-resampling
    approach.  Rather than assuming a parametric null, it builds the null
    distribution of the log2 fold change by repeatedly drawing groups of
    ``n_perturbed`` cells *from the control pool* and recomputing the
    statistic.  The observed log2FC is then placed in that null.

    Why bother, when we already have a Mann-Whitney p-value?  Because the
    Mann-Whitney null assumes exchangeability that single-cell count data
    violates (depth differences between groups create apparent shifts).
    Resampling within the control pool automatically absorbs that: the null
    groups have the same size and come from the same depth distribution.

    Returns the observed log2FC, an empirical two-sided p-value, and a z-score
    against the resampled null.
    """
    a = np.asarray(expr_perturbed, dtype=np.float64).ravel()
    b = np.asarray(expr_control, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n_a = a.size
    if n_a == 0 or b.size < 2:
        return {"log2fc": np.nan, "p_resample": np.nan, "z_resample": np.nan,
                "n_resample": 0}

    lin_a = np.clip(np.expm1(a) if log_input else a, 0, None)
    lin_b = np.clip(np.expm1(b) if log_input else b, 0, None)
    mb = float(lin_b.mean())
    obs = float(np.log2((float(lin_a.mean()) + 1e-9) / (mb + 1e-9)))

    rng = np.random.default_rng(random_state)
    draw = min(n_a, b.size)
    null = np.empty(n_resample, dtype=np.float64)
    for i in range(n_resample):
        samp = lin_b[rng.integers(0, lin_b.size, size=draw)]
        null[i] = np.log2((float(samp.mean()) + 1e-9) / (mb + 1e-9))

    centre = float(np.mean(null))
    spread = float(np.std(null))
    n_extreme = int(np.sum(np.abs(null - centre) >= abs(obs - centre)))
    p = (n_extreme + 1) / (n_resample + 1)
    z = (obs - centre) / spread if spread > 0 else np.nan
    return {"log2fc": obs, "p_resample": float(p), "z_resample": float(z),
            "n_resample": int(n_resample)}


def significance_stars(p: float) -> str:
    """Conventional star annotation, blank when p is missing."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


# ===========================================================================
# Perturbation score (Mixscale-like)
# ===========================================================================
def perturbation_score(
    X_perturbed,
    X_control,
    de_genes_idx: np.ndarray,
    de_log2fc: np.ndarray,
) -> np.ndarray:
    """Continuous per-cell perturbation score.

    Projects each perturbed cell onto the perturbation's own expression
    signature: the score is the weighted sum of that cell's centred
    expression over the perturbation's DE genes, weighted by each gene's
    log2 fold change and normalised by the signature's magnitude.

    Cells with a high score look strongly perturbed; cells near zero look
    like escapers.  This is the ranking used for the target-gene expression
    heatmap, and it is the concept Mixscale's perturbation score captures --
    reimplemented here directly so the pipeline has no R dependency.
    """
    idx = np.asarray(de_genes_idx, dtype=int)
    w = np.asarray(de_log2fc, dtype=np.float64)
    n_pert = int(getattr(X_perturbed, "shape", (0, 0))[0])
    if idx.size == 0:
        return np.full(n_pert, np.nan)
    # Only the signature genes are densified, never the whole matrix.
    Xp = take_columns(X_perturbed, idx)
    Xc = take_columns(X_control, idx)
    ctrl_mean = Xc.mean(axis=0)
    centred = Xp - ctrl_mean[None, :]
    norm = float(np.sqrt(np.sum(w**2)))
    if norm == 0:
        return np.full(n_pert, np.nan)
    if w.size != centred.shape[1]:
        w = w[: centred.shape[1]]
    return (centred @ w) / norm


# ===========================================================================
# Misc
# ===========================================================================
def summarise_numeric(s: pd.Series) -> dict[str, float]:
    """Compact numeric summary used in report tables."""
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return {k: np.nan for k in ("n", "mean", "median", "sd", "min", "max")}
    return {
        "n": float(v.size),
        "mean": float(v.mean()),
        "median": float(v.median()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
    }


def gini(x: np.ndarray) -> float:
    """Gini coefficient -- how unevenly guide/hashtag abundance is distributed.

    0 = perfectly even library, 1 = one feature takes everything.  A useful
    single number for "is my guide library skewed?", which the original
    pipeline only ever showed as a rank-abundance curve.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v) & (v >= 0)]
    if v.size == 0 or v.sum() == 0:
        return float("nan")
    v = np.sort(v)
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)
