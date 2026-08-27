"""
Hashtag (HTO) demultiplexing and hashtag-performance QC.

Consolidates the original's sections 15-18, 21 and 22, which had accumulated:

* two independent hashtag-matrix getters with different candidate lists
  (``get_hto_count_matrix`` and ``_get_total_hto_counts``) that could disagree
  about which columns were hashtags;
* a ``_hto_name_in_call`` fallback that took a short suffix of the hashtag name
  and substring-matched it against the whole call string, so a one-character
  hashtag ID could match an unrelated token in a doublet call and mis-colour
  the ridge plot;
* scikit-learn KMeans with a random seed on skewed, zero-inflated 1-D data,
  where the background/signal split can flip between runs;
* a stray copy-paste block near the end of the file that silently reassigned
  the global ``FIG_DIR``/``TABLE_DIR``, so the final hashtag heatmap could be
  written somewhere other than every other figure in the run.

The default normalisation is unchanged in substance -- log1p then per-hashtag
mean centring across cells -- and is named for what it is
(``stats.clr_by_feature``) rather than "CLR", which invites the reader to assume
the compositional definition.

What v1.3.0 corrects is the *claim* that went with it. This module and
``clr_by_feature`` both used to say the transform was Seurat's
``CLR, margin = 2``. The axis matches Seurat's ``margin = 2`` -- per feature,
across cells -- and matches the ORIGINAL Cell Hashing definition (Stoeckius et
al. 2018): each HTO's counts divided by that HTO's own geometric mean across
cells. It is NOT, however, what the canonical HTODemux hashing vignette runs
by default: that vignette's ``NormalizeData(..., normalization.method =
"CLR")`` call omits ``margin``, so it uses Seurat's default of ``margin = 1``
(per-cell). ``margin = 2`` is the WNN/ADT-visualisation vignette's
recommendation, a different downstream use case from demultiplexing. Citing
"what the WNN tutorial uses" as the reason to run margin=2 for HTO calling was
an overclaim in earlier versions of this docstring; the honest justification
is the original paper's own formula, which happens to be the same axis. Even
so the arithmetic was not Seurat's either way: Seurat divides on the raw scale
inside the log and forms its geometric mean by summing ``log1p`` over nonzero
entries while dividing by the length of the whole vector. Renaming the
function and then
restating the overclaim in the comment beside it defeated the point of the
rename. ``stats.clr_true_seurat`` now implements the Seurat formula,
``HTOConfig.normalisation`` selects between them, and
``compare_normalisations`` reports what the choice actually costs on the data
in front of you.

Also worth stating plainly: the calling structure here is per-hashtag
positivity, plus matching against declared combinations where a whitelist
provides them. That is deliberately not ``HTODemux``, which assumes one tag per
cell in its classification step and is therefore the wrong tool for
combinatorial hashing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotting as P
from . import text as T
from .artifacts import Registry
from .config import FigureConfig, HTOConfig, PipelineConfig
from .modalities import Modality
from .stats import (
    HTO_NORMALISATIONS, clr_by_feature, otsu_threshold, split_bimodal_1d,
)

NEGATIVE = "Negative"
MULTIPLET = "Multiplet"


AMBIGUOUS = "Ambiguous"
RESOLVED = "Resolved"


@dataclass
class HTOCalls:
    """Per-cell hashtag calls plus per-hashtag thresholds."""

    per_cell: pd.DataFrame           # call, n_positive, per-hashtag booleans
    clr: pd.DataFrame                # normalised intensities, cells x hashtags
    thresholds: pd.DataFrame         # hashtag, threshold, bg_mean, signal_mean, ...
    names: list[str]
    mode: str
    notes: list[str] = field(default_factory=list)
    # Set when a hashtag whitelist declared the design.
    design_declared: bool = False
    unexpected_sets: pd.DataFrame | None = None

    @property
    def rates(self) -> dict[str, float]:
        n = len(self.per_cell)
        if n == 0:
            return {}
        vc = self.per_cell["hto_class"].value_counts()
        out = {
            "pct_singlet": 100.0 * vc.get("Singlet", 0) / n,
            "pct_multiplet": 100.0 * vc.get(MULTIPLET, 0) / n,
            "pct_negative": 100.0 * vc.get(NEGATIVE, 0) / n,
        }
        if self.design_declared:
            out["pct_resolved"] = 100.0 * vc.get(RESOLVED, 0) / n
            out["pct_ambiguous"] = 100.0 * vc.get(AMBIGUOUS, 0) / n
        return out


# ===========================================================================
# Normalisation and thresholds
# ===========================================================================
def normalise(
    hto: Modality,
    precomputed: pd.DataFrame | None = None,
    cfg: HTOConfig | None = None,
) -> pd.DataFrame:
    """Per-hashtag normalised intensities.

    A precomputed matrix (e.g. ``*_CLR`` obs columns written by an upstream
    tool) is used only when it covers *every* hashtag; a partially precomputed
    set would mix two normalisations on one threshold scale, which is worse
    than recomputing.

    Which transform is used is now explicit (``HTOConfig.normalisation``) rather
    than hard-coded. The default is unchanged.
    """
    if precomputed is not None and list(precomputed.columns) == list(hto.names):
        return precomputed.astype(float)
    name = (cfg.normalisation if cfg is not None else "mean_centred_log1p")
    fn = HTO_NORMALISATIONS.get(str(name), clr_by_feature)
    X = fn(hto.X)
    return pd.DataFrame(
        X, columns=list(hto.names),
        index=hto.obs_names or list(range(hto.n_cells)),
    )


def compare_normalisations(
    X: Any, names: Sequence[str], cfg: HTOConfig
) -> pd.DataFrame:
    """Per-hashtag threshold and call rate under every available transform.

    The question "is this really best-practice CLR?" is not settled by reading
    the Seurat source, because the transforms differ in shape but are all
    monotone in the raw count -- so the classification structure is identical
    and only the cutoff moves. How much it moves is an empirical question about
    a particular dataset, and this is the table that answers it.

    One row per (transform, hashtag). Cheap: a threshold fit per hashtag per
    transform, on a matrix that is cells x (a handful of hashtags).
    """
    rows: list[dict[str, Any]] = []
    for label, fn in HTO_NORMALISATIONS.items():
        try:
            M = pd.DataFrame(fn(X), columns=list(names))
        except Exception as exc:                       # pragma: no cover
            rows.append({"normalisation": label, "hashtag": "(all)",
                         "threshold": np.nan, "pct_positive": np.nan,
                         "valley_ratio": np.nan, "error": str(exc)})
            continue
        thr = compute_thresholds(M, cfg)
        for _, r in thr.iterrows():
            rows.append({
                "normalisation": label,
                "hashtag": r.get("hashtag"),
                "threshold": r.get("threshold"),
                "pct_positive": 100.0 * float(r.get("frac_positive", np.nan)),
                "valley_ratio": r.get("valley_ratio", np.nan),
                "separation": r.get("separation", np.nan),
            })
    out = pd.DataFrame(rows)
    if not out.empty and "pct_positive" in out.columns:
        out = out.sort_values(["hashtag", "normalisation"]).reset_index(drop=True)
    return out


def compute_thresholds(clr: pd.DataFrame, cfg: HTOConfig) -> pd.DataFrame:
    """One positivity threshold per hashtag, with a separability diagnostic.

    Three modes:

    ``background_quantile`` (default)
        Split the hashtag's own intensity distribution into background and
        signal, then take a high quantile of the *background* cluster.  This
        adapts per hashtag, which matters because hashtags differ in overall
        abundance by an order of magnitude.
    ``otsu``
        The variance-minimising cut. More stable than a quantile when the two
        modes are well separated, less forgiving when they overlap.
    ``fixed``
        One number for every hashtag. Only defensible when the normalisation
        has genuinely put all hashtags on the same scale.

    ``separation`` is the gap between the cluster means. It is reported so a
    hashtag whose distribution is unimodal -- i.e. one that did not work -- is
    visible as a number rather than only as a shape in a plot.
    """
    rows = []
    for name in clr.columns:
        v = clr[name].to_numpy(dtype=float)
        finite = v[np.isfinite(v)]
        bg_mask, bg_mean, sig_mean = split_bimodal_1d(v, cfg.random_state)
        bg = v[bg_mask & np.isfinite(v)]
        sig = v[(~bg_mask) & np.isfinite(v)]

        bg_sd = float(np.std(bg)) if bg.size > 1 else float("nan")
        sig_sd = float(np.std(sig)) if sig.size > 1 else float("nan")

        raw_gap = (
            float(sig_mean - bg_mean)
            if np.isfinite(sig_mean) and np.isfinite(bg_mean) else np.nan
        )
        # Size-WEIGHTED pooled SD. v1.1.0 used np.nanmean([bg_sd, sig_sd]),
        # which weights a 5%-of-cells cluster the same as a 95% one. On
        # MDL-1856 that inflated the denominator for prot:hash.C -- the hashtag
        # with the widest background -- and pushed it under the cut.
        pooled = _pooled_sd(bg.size, bg_sd, sig.size, sig_sd)
        sep_sd = (
            float(raw_gap / pooled)
            if np.isfinite(raw_gap) and np.isfinite(pooled) and pooled > 0
            else np.nan
        )

        if cfg.threshold_mode == "fixed":
            thr = float(cfg.fixed_threshold)
        elif cfg.threshold_mode == "otsu":
            thr = float(otsu_threshold(v))
        else:
            thr = (
                float(np.quantile(bg, cfg.positive_quantile))
                if bg.size else float(cfg.fixed_threshold)
            )

        # Safety floor. Keeps a non-working hashtag from being handed a
        # threshold inside its own bulk, which would call most cells positive
        # for it and inflate the multiplet rate for the whole experiment.
        floor = (
            bg_mean + cfg.min_threshold_background_sd * bg_sd
            if np.isfinite(bg_mean) and np.isfinite(bg_sd) else -np.inf
        )
        thr_final = float(max(thr, floor))
        floored = thr_final > thr + 1e-12

        frac_bg = float(bg_mask.mean())
        frac_pos = float(np.mean(finite > thr_final)) if finite.size else float("nan")
        valley = _valley_ratio(finite, thr_final, cfg.valley_bins)
        verdict = _separability_verdict(
            cfg, valley, sep_sd, frac_bg, bg_sd, frac_pos
        )
        rows.append(
            {
                "hashtag": name,
                "threshold": thr_final,
                "threshold_from_quantile": thr,
                "raised_to_floor": floored,
                "background_mean": bg_mean,
                "background_sd": bg_sd,
                "signal_mean": sig_mean,
                "separation": raw_gap,
                "separation_sd": sep_sd,
                "valley_ratio": valley,
                "frac_background": frac_bg,
                "frac_positive": frac_pos,
                "separability": verdict,
                "well_separated": verdict in ("clean", "shallow"),
                "mode": cfg.threshold_mode,
            }
        )
    return pd.DataFrame(rows)


def _pooled_sd(n0: int, s0: float, n1: int, s1: float) -> float:
    """Size-weighted pooled standard deviation of two clusters."""
    if not (np.isfinite(s0) and np.isfinite(s1)) or n0 + n1 <= 2:
        return float(np.nanmean([s0, s1]))
    num = (n0 - 1) * s0 ** 2 + (n1 - 1) * s1 ** 2
    return float(np.sqrt(num / (n0 + n1 - 2)))


def _smoothed_density(
    v: np.ndarray, bins: int
) -> tuple[np.ndarray | None, np.ndarray]:
    """Histogram convolved with a Gaussian whose width is set by the data.

    The bandwidth has to come from the data's spread, not from the bin count.
    A low-abundance hashtag is a handful of small integer counts, and after
    log1p its "distribution" is a comb of discrete spikes with empty bins
    between them. Any fixed-width smoother reads those empty bins as troughs
    and pronounces a hashtag that captured nothing perfectly bimodal -- which
    is precisely the failure this metric exists to catch.

    Silverman's rule with the robust scale estimate ``min(sd, IQR/1.349)``:
    the IQR term stops a genuinely bimodal distribution, whose SD is inflated
    by the gap between its modes, from being over-smoothed into one bump.
    """
    n = v.size
    sd = float(np.std(v))
    q75, q25 = np.percentile(v, [75, 25])
    iqr = float(q75 - q25)
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    if not np.isfinite(spread) or spread <= 0:
        spread = sd
    if not np.isfinite(spread) or spread <= 0:
        return None, np.array([])

    hist, edges = np.histogram(v, bins=bins, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_w = float(edges[1] - edges[0])
    if bin_w <= 0:
        return None, centres

    h = 0.9 * spread * n ** (-0.2)
    sigma_bins = max(h / bin_w, 1.0)
    radius = int(min(np.ceil(3 * sigma_bins), max(bins // 2 - 1, 1)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return np.convolve(hist, kernel, mode="same"), centres


def _valley_ratio(v: np.ndarray, cut: float, bins: int) -> float:
    """Depth of the trough between the two modal peaks, as a fraction.

    This is the statistic that actually answers "is this distribution
    bimodal?". The v1.1.0 metric -- distance between two k-means centroids,
    standardised -- measures how far apart the modes are, which is a different
    question and gets the answer wrong in both directions:

        prot:hash.C  sep_sd 2.59 -> flagged, but has a deep trough
        prot:hash.F  sep_sd 2.84 -> passed,  but has no trough at all

    Method: locate the highest density peak either side of the cut, then take
    the minimum density between them, divided by the smaller of the two peaks.
    Near 0 means a clean separation; near 1 means no dip and the "two modes"
    are one blob split in half.

    Two details that matter:

    * The histogram is smoothed before anything is measured. Reading the
      density *at the cut* on a raw histogram, as the first version of this
      did, scores a unimodal distribution as perfectly separated whenever the
      threshold happens to land in an empty bin out in the tail -- which is
      exactly where a 3-SD cut on a unimodal hashtag lands.
    * The trough is measured between the PEAKS, not at the cut. A hashtag can
      be genuinely bimodal with the threshold placed slightly off the trough,
      and that is a threshold problem, not a separability problem.
    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 50 or not np.isfinite(cut):
        return float("nan")
    smooth, centres = _smoothed_density(v, bins)
    if smooth is None:
        return float("nan")

    i = int(np.argmin(np.abs(centres - cut)))
    if i <= 0 or i >= len(smooth) - 1:
        # The cut sits outside the data: no valley to measure, which is itself
        # a failure of separation rather than a missing value.
        return 1.0

    left_peak = int(np.argmax(smooth[:i]))
    right_peak = i + int(np.argmax(smooth[i:]))
    if right_peak <= left_peak:
        return 1.0
    peak = min(float(smooth[left_peak]), float(smooth[right_peak]))
    if peak <= 0:
        return float("nan")
    valley = float(smooth[left_peak:right_peak + 1].min())
    return float(min(valley / peak, 1.0))


def _separability_verdict(
    cfg: HTOConfig,
    valley: float,
    sep_sd: float,
    frac_bg: float,
    bg_sd: float,
    frac_pos: float = float("nan"),
) -> str:
    """Graded verdict: degenerate | unimodal | shallow | clean.

    Graded rather than boolean because the underlying quantity is continuous:
    v1.1.0 reported hash.C (2.587) and hash.F (2.838) as different kinds of
    thing when they sat 0.25 SD apart on a metric with a hard cut at 2.5.
    """
    # A "background cluster" that is 98% of cells with near-zero spread is a
    # hashtag that captured almost nothing. It produces an enormous separation
    # score -- 29 SD for prot:hash.D -- which v1.1.0 reported as excellent.
    if (
        np.isfinite(frac_bg) and frac_bg >= cfg.degenerate_frac_background_min
    ) or (np.isfinite(bg_sd) and bg_sd < cfg.degenerate_bg_sd_min):
        return "degenerate"
    # Almost nothing above the threshold is the same failure seen from the
    # other side: there is no signal population to separate from.
    if np.isfinite(frac_pos) and frac_pos < cfg.degenerate_frac_positive_min:
        return "degenerate"
    if not np.isfinite(valley):
        # Fall back to the standardised gap when the density estimate failed.
        if np.isfinite(sep_sd) and sep_sd >= cfg.min_separation_sd:
            return "clean"
        return "unimodal"
    if valley <= cfg.valley_ratio_clean_max:
        return "clean"
    if valley <= cfg.valley_ratio_shallow_max:
        return "shallow"
    return "unimodal"


def call_hashtags(
    hto: Modality,
    cfg: HTOConfig,
    precomputed: pd.DataFrame | None = None,
    whitelist: Any = None,
) -> HTOCalls:
    """Classify each cell from its set of positive hashtags.

    Two modes:

    **With a hashtag whitelist** the cell's positive set is matched against the
    declared combinations. A match is ``Resolved`` and carries the sample's
    ``demux_id``, ``aliquot``, ``family`` and metadata; a non-empty set matching
    nothing is ``Ambiguous``; an empty set is ``Negative``.

    **Without one** the v1.1.0 rule applies: 1 positive is a Singlet, 2 or more
    a Multiplet. That rule cannot express combinatorial tagging, which is why
    MDL-1856 reported 57.3% "multiplets" and was graded ``poor`` for what is
    partly the intended design.

    Note that Resolved/Ambiguous and Singlet/Multiplet are reported side by
    side, never merged: the second pair still says something useful about how
    many tags a cell carries, independent of whether that was intended.
    """
    clr = normalise(hto, precomputed, cfg)
    th = compute_thresholds(clr, cfg)
    thr_vec = th.set_index("hashtag")["threshold"].reindex(clr.columns).to_numpy()

    total = hto.X.sum(axis=1)
    above_depth = total > cfg.min_reads

    positive = clr.to_numpy(dtype=float) > thr_vec[None, :]
    # A cell without enough hashtag reads cannot be called positive for
    # anything, regardless of where its normalised value lands.
    positive[~above_depth, :] = False

    n_pos = positive.sum(axis=1)
    names = list(clr.columns)
    call = np.where(
        n_pos == 0, NEGATIVE,
        np.where(n_pos == 1, np.array(names, dtype=object)[np.argmax(positive, axis=1)],
                 MULTIPLET),
    )
    klass = np.where(n_pos == 0, NEGATIVE, np.where(n_pos == 1, "Singlet", MULTIPLET))

    per_cell = pd.DataFrame(
        {
            "hto_call": call,
            "hto_class": klass,
            "n_hto_positive": n_pos,
            "hto_total_umis": total,
            "hto_above_min_reads": above_depth,
        },
        index=clr.index,
    )
    # Explicit per-hashtag booleans. Downstream code asks
    # `per_cell["hto_pos_HTO3"]` rather than substring-matching the call
    # string, which is what made the original's ridge-plot colouring wrong.
    for j, name in enumerate(names):
        per_cell[f"hto_pos_{name}"] = positive[:, j]
    # The multiplet composition, kept as a list so it is unambiguous.
    per_cell["hto_positive_set"] = [
        "+".join([names[j] for j in np.flatnonzero(row)]) or NEGATIVE
        for row in positive
    ]

    notes: list[str] = []
    unimodal = th.loc[th["separability"] == "unimodal", "hashtag"].tolist()
    if unimodal:
        notes.append(
            f"{len(unimodal)} hashtag(s) show no trough between background and "
            f"signal ({', '.join(map(str, unimodal[:6]))}): their intensity "
            f"distributions are effectively unimodal, so the positivity cut is "
            f"splitting one population rather than separating two. Every call "
            f"involving them is unreliable."
        )
    degenerate = th.loc[th["separability"] == "degenerate", "hashtag"].tolist()
    if degenerate:
        rows = th.set_index("hashtag").loc[degenerate]
        detail = ", ".join(
            f"{h} ({100 * rows.loc[h, 'frac_background']:.1f}% background)"
            for h in degenerate[:4]
        )
        notes.append(
            f"{len(degenerate)} hashtag(s) captured almost nothing ({detail}). "
            f"Their background 'cluster' is a spike, which produces a very "
            f"large but meaningless separation score -- they are not working, "
            f"despite scoring well on the standardised gap."
        )
    shallow = th.loc[th["separability"] == "shallow", "hashtag"].tolist()
    if shallow:
        notes.append(
            f"{len(shallow)} hashtag(s) separate only shallowly "
            f"({', '.join(map(str, shallow[:6]))}). Calls involving them are "
            f"sensitive to the exact threshold; check the sweep panel."
        )
    if bool(th["raised_to_floor"].all()) and cfg.threshold_mode != "fixed":
        notes.append(
            f"Every threshold was set by the {cfg.min_threshold_background_sd:g}-SD "
            f"safety floor rather than by the configured "
            f"{cfg.threshold_mode!r} rule, because the rule's value fell below "
            f"the floor for all {len(th)} hashtags. The effective rule here is "
            f"'{cfg.min_threshold_background_sd:g} SD above background mean'."
        )
    n_low = int((~above_depth).sum())
    if n_low:
        notes.append(
            f"{n_low:,} cells ({100.0 * n_low / len(per_cell):.1f}%) have "
            f"{cfg.min_reads} or fewer hashtag UMIs and were forced Negative on "
            f"depth alone, before any threshold was applied."
        )

    # --- match against the declared design -----------------------------------
    design_declared = whitelist is not None
    unexpected: pd.DataFrame | None = None
    if design_declared:
        per_cell, unexpected, wl_notes = _apply_design(
            per_cell, whitelist, cfg
        )
        notes.extend(wl_notes)
    else:
        per_cell["hto_class"] = per_cell["hto_class"]
        notes.append(
            "No hashtag whitelist was supplied, so cells are classified by tag "
            "count alone (1 = singlet, >=2 = multiplet). If this experiment "
            "used combinatorial tagging, the multiplet rate below counts "
            "intended combinations as failures. Declare the design with an "
            "expected-combination whitelist to classify them properly."
        )

    return HTOCalls(
        per_cell=per_cell, clr=clr, thresholds=th, names=names,
        mode=cfg.threshold_mode, notes=notes,
        design_declared=design_declared, unexpected_sets=unexpected,
    )


def _apply_design(
    per_cell: pd.DataFrame, whitelist: Any, cfg: HTOConfig
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Classify each cell's positive set against the declared combinations.

    Unexpected sets are called ``Ambiguous`` and reported verbatim. They are
    never reassigned and never rescued by subset matching: a cell positive for
    only one tag of a declared pair may be a dropout, but it may equally be a
    cell from a different sample with one spurious tag, and guessing would put
    it silently into a comparison group.
    """
    lookup = whitelist.lookup()
    meta_cols = ["demux_id", "aliquot", "family"] + whitelist.metadata_columns()

    sets = per_cell["hto_positive_set"].astype(str)
    keys = [
        tuple(sorted(s.split("+"))) if s and s != NEGATIVE else ()
        for s in sets
    ]

    # An intentionally untagged sample is declared as the EMPTY set, so a cell
    # with no positive hashtag resolves to it rather than falling into
    # Negative. Without that declaration the untagged sample's cells are
    # indistinguishable from hashtag capture failure, and -- worse -- they
    # carry no family, so they drop out of the family-scoped comparisons
    # entirely.
    untagged_key = () if () in lookup else None

    klass, matched = [], []
    for k in keys:
        if not k:
            if untagged_key is not None:
                klass.append(RESOLVED)
                matched.append(())
            else:
                klass.append(NEGATIVE)
                matched.append(None)
        elif k in lookup:
            klass.append(RESOLVED)
            matched.append(k)
        else:
            klass.append(AMBIGUOUS)
            matched.append(None)

    per_cell = per_cell.copy()
    per_cell["hto_class"] = klass
    for col in meta_cols:
        per_cell[f"hto_{col}"] = [
            lookup[m].get(col, "") if m is not None else "" for m in matched
        ]

    n = len(per_cell)
    n_res = int(sum(c == RESOLVED for c in klass))
    n_amb = int(sum(c == AMBIGUOUS for c in klass))
    notes = [
        f"Hashtag design declared: {len(lookup)} valid combination(s) across "
        f"{len(set(whitelist.demux_ids))} sample(s). "
        f"{n_res:,} cells ({100.0 * n_res / n:.1f}%) match a declared "
        f"combination, {n_amb:,} ({100.0 * n_amb / n:.1f}%) carry a positive "
        f"set that matches none."
    ]

    # Which unexpected sets, and how often -- the actionable diagnostic.
    amb = [
        "+".join(k) for k, c in zip(keys, klass) if c == AMBIGUOUS
    ]
    unexpected = (
        pd.Series(amb, dtype=object)
        .value_counts()
        .head(cfg.unexpected_sets_top_n)
        .rename_axis("positive_set")
        .reset_index(name="n_cells")
    )
    if len(unexpected):
        unexpected["pct_cells"] = 100.0 * unexpected["n_cells"] / n
        unexpected["n_hashtags"] = unexpected["positive_set"].str.count(r"\+") + 1
        top = unexpected.iloc[0]
        notes.append(
            f"Most common unexpected combination: {top['positive_set']} "
            f"({top['n_cells']:,} cells, {top['pct_cells']:.1f}%). A long tail "
            f"of high-order combinations usually means ambient hashtag rather "
            f"than a missing row in the whitelist -- compare against the "
            f"ambiguous-rate-by-depth panel before editing the design."
        )

    if untagged_key is not None:
        row = lookup[()]
        n_untagged = int(sum(1 for k in keys if not k))
        low_depth = (
            int((~per_cell["hto_above_min_reads"]).sum())
            if "hto_above_min_reads" in per_cell.columns else 0
        )
        fam = row.get("family") or "none declared"
        notes.append(
            f"Sample {row.get('demux_id')!r} is declared as carrying no "
            f"hashtag by design; {n_untagged:,} cells ({100.0 * n_untagged / n:.1f}%) "
            f"with no positive hashtag were assigned to it. "
            f"UNAVOIDABLE CAVEAT: that group also contains every cell from a "
            f"TAGGED sample whose hashtag capture failed -- the two are "
            f"identical by hashtag alone ({low_depth:,} cells in the object "
            f"fell below the depth floor). If this sample has its own guide "
            f"family ({fam}), the guide-family cross-check separates them; "
            f"otherwise treat its cell count as an upper bound."
        )

    if whitelist.has_aliquots:
        n_al = int(whitelist.df["demux_id"].duplicated().sum())
        notes.append(
            f"{n_al} sample(s) were split across more than one tagging scheme "
            f"(e.g. one aliquot combinatorial, one single). Their cells pool "
            f"into one demux_id; hto_aliquot is retained on obs so the aliquots "
            f"can be compared before the pool is trusted."
        )
    return per_cell, unexpected, notes


def threshold_sweep(
    clr: pd.DataFrame, hto: Modality, cfg: HTOConfig
) -> pd.DataFrame:
    """Call rates as a function of the background quantile."""
    total = hto.X.sum(axis=1)
    above = total > cfg.min_reads
    rows = []
    bg_masks = {
        name: split_bimodal_1d(clr[name].to_numpy(float), cfg.random_state)[0]
        for name in clr.columns
    }
    for q in cfg.quantile_sweep:
        thr = np.array(
            [
                np.quantile(
                    clr[name].to_numpy(float)[bg_masks[name]], q
                ) if bg_masks[name].any() else np.inf
                for name in clr.columns
            ]
        )
        pos = clr.to_numpy(float) > thr[None, :]
        pos[~above, :] = False
        n = pos.sum(axis=1)
        total_cells = len(clr)
        rows.append(
            {
                "quantile": float(q),
                "pct_singlet": 100.0 * float((n == 1).mean()),
                "pct_multiplet": 100.0 * float((n >= 2).mean()),
                "pct_negative": 100.0 * float((n == 0).mean()),
                "n_cells": total_cells,
            }
        )
    return pd.DataFrame(rows)


def efficiency_by_group(calls: HTOCalls, group: pd.Series) -> pd.DataFrame:
    """Per-condition hashtag performance.

    ``_apply_design`` overwrites ``per_cell["hto_class"]`` in place, from the
    count-based labels (``"Singlet"``/``MULTIPLET``/``NEGATIVE``) to the
    whitelist-based ones (``RESOLVED``/``AMBIGUOUS``/``NEGATIVE``), when a
    hashtag whitelist is declared. This used to always count against the
    count-based labels regardless, so ``pct_singlet``/``pct_multiplet`` came
    out 0% for every condition on any run with a declared design -- those
    labels no longer exist in the column by the time this runs. Branches on
    ``calls.design_declared`` the same way the Summary metrics in
    ``run_hto_stage`` already correctly do, so the reported columns match
    whichever classification was actually used.
    """
    g = group.astype(str).reindex(calls.per_cell.index)
    rows = []
    for value in sorted(g.dropna().unique()):
        sub = calls.per_cell.loc[(g == value).to_numpy()]
        if sub.empty:
            continue
        vc = sub["hto_class"].value_counts()
        n = len(sub)
        row = {
            "group": value,
            "n_cells": n,
            "pct_negative": 100.0 * vc.get(NEGATIVE, 0) / n,
            "median_hto_umis": float(sub["hto_total_umis"].median()),
            "mean_hto_umis": float(sub["hto_total_umis"].mean()),
        }
        if calls.design_declared:
            row["pct_resolved"] = 100.0 * vc.get(RESOLVED, 0) / n
            row["pct_ambiguous"] = 100.0 * vc.get(AMBIGUOUS, 0) / n
        else:
            row["pct_singlet"] = 100.0 * vc.get("Singlet", 0) / n
            row["pct_multiplet"] = 100.0 * vc.get(MULTIPLET, 0) / n
        rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Figures
# ===========================================================================
def plot_diagnostic_grid(
    calls: HTOCalls, fcfg: FigureConfig, path: Path
) -> Path:
    """Per-hashtag intensity histogram with its threshold -- the key HTO panel.

    A hashtag that did not work is only visible here.  Poorly separated
    hashtags are titled in red so the reader does not have to compare a number
    in a table against a shape in a plot.
    """
    names = calls.names
    nrows, ncols = P.grid_dims(len(names), max_cols=4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.5 * nrows),
                             squeeze=False)
    thr = calls.thresholds.set_index("hashtag")
    for ax, name in zip(axes.ravel(), names):
        v = calls.clr[name].to_numpy(float)
        row = thr.loc[name]
        P.histogram_by_group(ax, {name: v}, fcfg, xlabel="normalised intensity",
                             vlines=[row["threshold"]], bins=70, density=True)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ok = bool(row["well_separated"])
        sep = row.get("separation_sd", float("nan"))
        label = f"{name}  (separation = {sep:.1f} SD)"
        if not ok:
            label += "  — DID NOT SEPARATE"
        ax.set_title(
            label,
            color="#222" if ok else "#C44E52",
            fontweight="normal" if ok else "bold",
            fontsize=8,
        )
    P.blank_unused_axes(axes, len(names))
    fig.suptitle("per-hashtag intensity and positivity threshold", fontsize=10)
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_sweep(sweep: pd.DataFrame, cfg: HTOConfig, fcfg: FigureConfig,
               path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for col, c, label in (
        ("pct_singlet", "#55A868", "singlet"),
        ("pct_multiplet", "#C44E52", "multiplet"),
        ("pct_negative", "#8C8C8C", "negative"),
    ):
        ax.plot(sweep["quantile"], sweep[col], "-o", ms=3, color=c, label=label)
    ax.axvline(cfg.positive_quantile, color="#4C72B0", ls="--", lw=1.1,
               label="chosen")
    ax.set_xlabel("background quantile used as the positivity threshold")
    ax.set_ylabel("% of cells")
    ax.legend(fontsize=7)
    ax.set_title("sensitivity of hashtag calls to the threshold")
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_call_breakdown(
    calls: HTOCalls, group: pd.Series | None, fcfg: FigureConfig, path: Path
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))

    vc = calls.per_cell["hto_call"].value_counts()
    order = [n for n in calls.names if n in vc.index] + [
        x for x in (NEGATIVE, MULTIPLET) if x in vc.index
    ]
    vals = [vc.get(k, 0) for k in order]
    colors = P.palette(fcfg, len(calls.names)) + ["#8C8C8C", "#C44E52"]
    axes[0].bar(range(len(order)), vals, color=colors[: len(order)])
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(order, rotation=45, ha="right")
    for i, v in enumerate(vals):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=6.5)
    axes[0].set_ylabel("cells")
    axes[0].set_title("cells per hashtag call")

    if group is not None:
        g = group.astype(str).reindex(calls.per_cell.index)
        frac = pd.crosstab(g, calls.per_cell["hto_class"], normalize="index")
        P.stacked_fraction_bars(axes[1], frac, fcfg, legend_title="class")
        axes[1].set_title("call class by group")
    else:
        cls = calls.per_cell["hto_class"].value_counts(normalize=True).to_frame().T
        cls.index = ["all cells"]
        P.stacked_fraction_bars(axes[1], cls, fcfg, legend_title="class")
        axes[1].set_title("call class")
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_heatmap(calls: HTOCalls, fcfg: FigureConfig, path: Path,
                 max_cells_per_call: int = 300) -> Path:
    """Normalised intensity per cell, grouped by call -- the diagonal check."""
    per_cell, clr = calls.per_cell, calls.clr
    order_calls = [n for n in calls.names] + [MULTIPLET, NEGATIVE]
    rows, boundaries, labels = [], [], []
    rng = np.random.default_rng(0)
    for c in order_calls:
        idx = np.flatnonzero((per_cell["hto_call"] == c).to_numpy())
        if idx.size == 0:
            continue
        if idx.size > max_cells_per_call:
            idx = rng.choice(idx, max_cells_per_call, replace=False)
            idx.sort()
        rows.append(clr.iloc[idx])
        boundaries.append(sum(len(r) for r in rows))
        labels.append(f"{c} (n={len(idx):,})")
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        P.annotate_empty(ax, "no cells to display")
        return P.save_figure(fig, path, fcfg)

    M = pd.concat(rows)
    fig, ax = plt.subplots(figsize=(1.1 * len(calls.names) + 3.5, 5))
    im = ax.imshow(M.to_numpy(float), aspect="auto", cmap=fcfg.continuous_cmap,
                   interpolation="nearest")
    ax.set_xticks(range(len(calls.names)))
    ax.set_xticklabels(calls.names, rotation=45, ha="right", fontsize=7)
    mids = [0] + boundaries
    ax.set_yticks([(mids[i] + mids[i + 1]) / 2 for i in range(len(labels))])
    ax.set_yticklabels(labels, fontsize=7)
    for b in boundaries[:-1]:
        ax.axhline(b - 0.5, color="white", lw=1.2)
    ax.grid(False)
    ax.set_title("normalised hashtag intensity, grouped by call")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("normalised intensity", fontsize=7)
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_ridge(calls: HTOCalls, fcfg: FigureConfig, path: Path) -> Path:
    """Intensity distribution per hashtag, split by whether the cell was called.

    Membership comes from the explicit ``hto_pos_<name>`` boolean columns, not
    from substring-matching the call string -- the bug in the original.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.2, 0.42 * len(calls.names) + 2)))
    P.ridge(axes[0], {n: calls.clr[n].to_numpy(float) for n in calls.names},
            fcfg, xlabel="normalised intensity (all cells)")
    axes[0].set_title("all cells")

    called = {}
    for n in calls.names:
        m = calls.per_cell[f"hto_pos_{n}"].to_numpy(bool)
        called[f"{n} (+, n={int(m.sum()):,})"] = calls.clr[n].to_numpy(float)[m]
    P.ridge(axes[1], called, fcfg, xlabel="normalised intensity (positive cells only)")
    axes[1].set_title("cells called positive for that hashtag")
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_efficiency(eff: pd.DataFrame, group_name: str, fcfg: FigureConfig,
                    path: Path) -> Path:
    # `efficiency_by_group` emits pct_resolved/pct_ambiguous when a hashtag
    # whitelist was declared, and pct_singlet/pct_multiplet otherwise -- pick
    # whichever pair is actually present rather than assuming the count-based
    # one, which silently plotted two all-zero series on any whitelisted run.
    if "pct_resolved" in eff.columns:
        class_cols = ["pct_resolved", "pct_ambiguous", "pct_negative"]
    else:
        class_cols = ["pct_singlet", "pct_multiplet", "pct_negative"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    frac = eff.set_index(eff["group"].astype(str))[class_cols] / 100.0
    P.stacked_fraction_bars(axes[0], frac, fcfg, legend_title="class")
    axes[0].set_title(f"call rates by {group_name}")

    axes[1].bar(eff["group"].astype(str), eff["median_hto_umis"],
                color=P.palette(fcfg, len(eff)))
    for i, v in enumerate(eff["median_hto_umis"]):
        if np.isfinite(v):
            axes[1].text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=6.5)
    axes[1].set_ylabel("median hashtag UMIs per cell")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("hashtag depth")
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


def plot_composition(
    calls: HTOCalls, group: pd.Series, group_name: str, fcfg: FigureConfig,
    path: Path,
) -> Path:
    """Hashtag-derived composition per group -- the pooling check.

    Filtered on ``hto_class == "Singlet"`` unconditionally, but
    ``_apply_design`` overwrites ``hto_class`` to ``RESOLVED``/``AMBIGUOUS``/
    ``NEGATIVE`` whenever a hashtag whitelist is declared -- "Singlet" no
    longer exists in the column at that point, so this always rendered "no
    singlets to compose" on any whitelisted run, even with a healthy resolved
    rate. Uses whichever label actually means "successfully called" for this
    run's classification mode.
    """
    g = group.astype(str).reindex(calls.per_cell.index)
    called_label = RESOLVED if calls.design_declared else "Singlet"
    called = calls.per_cell.loc[calls.per_cell["hto_class"] == called_label]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    if called.empty:
        P.annotate_empty(ax, f"no {called_label.lower()} cells to compose")
        return P.save_figure(fig, path, fcfg)
    # `hto_call` is the naive per-tag-count label (a specific hashtag name for
    # exactly one positive tag, else "Multiplet") -- for a Resolved cell under
    # a declared combinatorial design that carries two-or-more tags on
    # purpose, `hto_call` is just "Multiplet" for every one of them, which
    # collapses the whole composition into one meaningless bucket. Compose by
    # the resolved sample (`hto_demux_id`, set by `_apply_design`) instead,
    # since that is what "composition" is actually asking about once a design
    # is declared.
    composition_col = (
        "hto_demux_id" if calls.design_declared and "hto_demux_id" in called.columns
        else "hto_call"
    )
    frac = pd.crosstab(
        g.reindex(called.index), called[composition_col], normalize="index"
    )
    legend_title = "sample" if composition_col == "hto_demux_id" else "hashtag"
    P.stacked_fraction_bars(ax, frac, fcfg, legend_title=legend_title)
    ax.set_title(f"{called_label.lower()} composition by {group_name}")
    fig.tight_layout()
    return P.save_figure(fig, path, fcfg)


# ===========================================================================
# Stage driver
# ===========================================================================
def run_hto_stage(
    hto: Modality,
    cfg: PipelineConfig,
    reg: Registry,
    group_columns: dict[str, pd.Series],
    manifest_declares_hto: bool | None = None,
    precomputed: pd.DataFrame | None = None,
    whitelist: Any = None,
) -> HTOCalls | None:
    if not hto.present:
        reason = f"No hashtag matrix was found in the h5ad. {hto.source}"
        if manifest_declares_hto:
            # A manifest/data disagreement is itself a QC finding and belongs in
            # the report rather than being silently resolved either way.
            reason += (
                " The manifest declares HTO=yes for this experiment, so either "
                "the hashtag matrix is stored under an unexpected key or the "
                "hashtag library was not included in this h5ad. Check "
                "ModalityConfig.hto_obsm_keys / hto_obs_prefixes."
            )
            reg.note("hashtags", "manifest_conflict",
                     "Manifest declares hashtags but none were found", reason,
                     level="poor", order=1)
        reg.skipped("hashtags", "all", "Hashtag analysis", reason)
        return None

    hcfg, fcfg = cfg.hto, cfg.figures
    fig_dir, table_dir = cfg.fig_dir, cfg.table_dir

    calls = call_hashtags(hto, hcfg, precomputed, whitelist=whitelist)
    if manifest_declares_hto is False:
        reg.note(
            "hashtags", "manifest_conflict",
            "Hashtags found but not declared in the manifest",
            f"A hashtag matrix with {hto.n_features} hashtags was found "
            f"({hto.source}), but the manifest's HTO column says this experiment "
            f"did not use hashtags. The data has been analysed; check which is "
            f"correct before reporting.",
            level="warn", order=1,
        )
    for i, note in enumerate(calls.notes):
        reg.note("hashtags", f"note_{i}", "Hashtag calling", note,
                 level="warn", order=3 + i)

    rates = calls.rates
    thr_repr = (
        f"background quantile {hcfg.positive_quantile:g}"
        if hcfg.threshold_mode == "background_quantile"
        else f"fixed at {hcfg.fixed_threshold:g}"
        if hcfg.threshold_mode == "fixed" else "Otsu cut"
    )

    reg.metric("summary", "n_hashtags", "Hashtags", hto.n_features, order=30)
    if calls.design_declared:
        # With a declared design, "resolved" is the number that means
        # something. Reporting a singlet rate here would grade an intended
        # two-tag combination as a failure, which is what v1.1.0 did to
        # MDL-1856's 57.3% "multiplets".
        pr = rates.get("pct_resolved", float("nan"))
        reg.metric(
            "summary", "pct_hto_resolved", "Hashtag resolved to a sample",
            round(pr, 1), unit="%",
            level=("good" if pr > hcfg.resolved_good_min
                   else "warn" if pr > hcfg.resolved_warn_min else "poor"),
            order=31,
        )
        reg.metric("summary", "pct_hto_ambiguous", "Hashtag ambiguous",
                   round(rates.get("pct_ambiguous", float("nan")), 1),
                   unit="%", order=32)
        if calls.unexpected_sets is not None and len(calls.unexpected_sets):
            calls.unexpected_sets.to_csv(
                table_dir / "hto_unexpected_sets.csv", index=False
            )
            reg.table(
                "hashtags", "unexpected", "Unexpected hashtag combinations",
                path=table_dir / "hto_unexpected_sets.csv",
                inline=calls.unexpected_sets.round(3).to_dict("records"),
                columns=list(calls.unexpected_sets.columns),
                caption=(
                    "Positive-hashtag combinations carried by cells that match "
                    "no declared combination. A few high-count pairs suggest a "
                    "missing row in the whitelist; a long tail of high-order "
                    "combinations suggests ambient hashtag instead."
                ),
                order=12,
            )
    else:
        reg.metric(
            "summary", "pct_hto_singlet", "Hashtag singlets",
            round(rates.get("pct_singlet", float("nan")), 1), unit="%",
            level=("good" if rates.get("pct_singlet", 0) > 70
                   else "warn" if rates.get("pct_singlet", 0) > 45 else "poor"),
            order=31,
        )
        reg.metric("summary", "pct_hto_multiplet", "Hashtag multiplets",
                   round(rates.get("pct_multiplet", float("nan")), 1), unit="%",
                   order=32)

    calls.thresholds.to_csv(table_dir / "hto_thresholds.csv", index=False)
    reg.table(
        "hashtags", "thresholds", "Per-hashtag thresholds and separability",
        path=table_dir / "hto_thresholds.csv",
        inline=calls.thresholds.round(4).to_dict("records"),
        columns=list(calls.thresholds.columns),
        caption=T.hto_desc(hcfg.threshold_mode, thr_repr), order=10,
    )
    reg.figure(
        "hashtags", "diagnostic", "Per-hashtag intensity and thresholds",
        plot_diagnostic_grid(calls, fcfg, fig_dir / "hto_diagnostic_grid.png"),
        caption=T.HTO_DIAGNOSTIC_DESC, order=20, width="full",
    )

    if hcfg.compare_normalisations:
        try:
            cmp_tab = compare_normalisations(hto.X, list(hto.names), hcfg)
        except Exception as exc:                       # pragma: no cover
            cmp_tab = pd.DataFrame()
            reg.skipped("hashtags", "normalisation_compare",
                        "Normalisation comparison",
                        f"Could not be computed ({exc}).")
        if not cmp_tab.empty:
            cmp_tab.to_csv(table_dir / "hto_normalisation_compare.csv",
                           index=False)
            spread = float(
                cmp_tab.groupby("hashtag")["pct_positive"].std().max()
            ) if "pct_positive" in cmp_tab.columns else float("nan")
            reg.table(
                "hashtags", "normalisation_compare",
                "Threshold and call rate by normalisation",
                path=table_dir / "hto_normalisation_compare.csv",
                inline=cmp_tab.round(4).to_dict("records"),
                columns=list(cmp_tab.columns),
                caption=(
                    f"This run used <code>{hcfg.normalisation}</code>. The table "
                    f"repeats the whole threshold fit under each available "
                    f"normalisation so the choice can be judged on this dataset "
                    f"rather than argued from the literature. "
                    f"<code>mean_centred_log1p</code> is log1p followed by "
                    f"per-hashtag mean centring; <code>seurat_clr</code> is "
                    f"Seurat's actual CLR formula, which divides on the raw "
                    f"scale and maps zero counts to exactly 0; "
                    f"<code>compositional</code> is the per-cell CLR and is "
                    f"included for completeness rather than recommended for "
                    f"demultiplexing. All three are monotone in the raw count "
                    f"for a fixed hashtag, so the calling rule has the same "
                    f"shape under each and only the cut-off moves &mdash; the "
                    f"largest spread in call rate across normalisations here is "
                    f"{spread:.1f} percentage points. A large spread means the "
                    f"hashtag's two modes are poorly separated and the call "
                    f"rate is being decided by the transform rather than by the "
                    f"data."
                ),
                level=("warn" if np.isfinite(spread) and spread > 10 else "info"),
                order=25,
            )
    reg.figure(
        "hashtags", "breakdown", "Hashtag call breakdown",
        plot_call_breakdown(calls, next(iter(group_columns.values()), None), fcfg,
                            fig_dir / "hto_call_breakdown.png"),
        caption=(
            "Cells assigned to each hashtag, and the singlet/multiplet/negative "
            "split. Large imbalance between hashtags that were pooled equally "
            "points to a capture or pooling bias."
        ),
        order=30, width="full",
    )

    sweep = threshold_sweep(calls.clr, hto, hcfg)
    sweep.to_csv(table_dir / "hto_threshold_sweep.csv", index=False)
    reg.figure(
        "hashtags", "sweep", "Call rates vs threshold",
        plot_sweep(sweep, hcfg, fcfg, fig_dir / "hto_threshold_sweep.png"),
        caption=T.HTO_SWEEP_DESC, order=40,
    )
    reg.figure(
        "hashtags", "heatmap", "Hashtag intensity by call",
        plot_heatmap(calls, fcfg, fig_dir / "hto_heatmap.png"),
        caption=T.HTO_HEATMAP_DESC, order=50,
    )
    reg.figure(
        "hashtags", "ridge", "Hashtag intensity distributions",
        plot_ridge(calls, fcfg, fig_dir / "hto_ridge.png"),
        caption=(
            "Left: every cell's intensity for each hashtag. Right: only cells "
            "called positive for that hashtag. Membership comes from the "
            "per-hashtag positivity flags, so multiplets are counted for every "
            "hashtag they are positive for."
        ),
        order=60, width="full",
    )

    for axis_name, series in group_columns.items():
        eff = efficiency_by_group(calls, series)
        if eff.empty:
            continue
        eff.to_csv(table_dir / f"hto_efficiency_by_{axis_name}.csv", index=False)
        reg.figure(
            "hashtags", f"efficiency_{axis_name}",
            f"Hashtag performance by {axis_name}",
            plot_efficiency(eff, axis_name, fcfg,
                            fig_dir / f"hto_efficiency_by_{axis_name}.png"),
            caption=(
                "Call rates and hashtag depth per condition. A condition with a "
                "high negative rate at comparable depth has a staining or "
                "washing problem, not a sequencing one."
            ),
            order=70, width="full",
        )
        reg.table(
            "hashtags", f"efficiency_table_{axis_name}",
            f"Hashtag performance by {axis_name}",
            path=table_dir / f"hto_efficiency_by_{axis_name}.csv",
            inline=eff.round(2).to_dict("records"), columns=list(eff.columns),
            order=75,
        )
        reg.figure(
            "hashtags", f"composition_{axis_name}",
            f"Singlet composition by {axis_name}",
            plot_composition(calls, series, axis_name, fcfg,
                             fig_dir / f"hto_composition_by_{axis_name}.png"),
            caption=T.COMPOSITION_DESC, order=80,
        )

    reg.note("hashtags", "method", "Method", T.HTO_NOTE, order=200)
    calls.per_cell.to_csv(table_dir / "hto_calls_per_cell.csv.gz",
                          compression="gzip")
    return calls


# ===========================================================================
# Cross-checks
# ===========================================================================
def family_crosscheck(
    guide_per_cell: pd.DataFrame,
    hto_calls: HTOCalls,
    cfg: PipelineConfig,
    reg: Registry,
) -> pd.DataFrame | None:
    """Guide family against hashtag family -- a direct contamination estimate.

    Only possible when both whitelists are declared, and it is the strongest
    check the pipeline has: the two labels are measured by different assays on
    different molecules. A cell whose hashtag says family A while its guide says
    family B is a doublet or an index hop, and the off-diagonal rate estimates
    how many such cells there are.

    The conflicting cells are flagged on ``guide_per_cell`` in place, so the
    perturbation stage can exclude them: whichever family they were counted in,
    they would carry another population's transcriptome into that comparison.
    """
    if "family" not in guide_per_cell.columns:
        return None
    if "hto_family" not in hto_calls.per_cell.columns:
        return None

    idx = guide_per_cell.index
    g_fam = guide_per_cell["family"].astype("object")
    h_fam = hto_calls.per_cell["hto_family"].reindex(idx).astype("object")
    h_fam = h_fam.where(h_fam.astype(str).str.len() > 0)

    both = g_fam.notna() & h_fam.notna()
    if not bool(both.any()):
        reg.skipped(
            "crosschecks", "family", "Guide family vs hashtag family",
            "No cell has both a guide family and a hashtag-derived family, so "
            "the two cannot be compared. This needs both whitelists, with a "
            "shared family vocabulary.",
        )
        return None

    conflict = both & (g_fam.astype(str) != h_fam.astype(str))
    guide_per_cell["family_conflict"] = conflict.reindex(idx).fillna(False)

    ct = pd.crosstab(g_fam[both].astype(str), h_fam[both].astype(str))
    ct.index.name = "guide_family"
    ct.columns.name = "hashtag_family"
    n_both = int(both.sum())
    n_conf = int(conflict.sum())
    rate = 100.0 * n_conf / n_both if n_both else float("nan")

    path = cfg.table_dir / "crosscheck_family.csv"
    ct.to_csv(path)
    reg.table(
        "crosschecks", "family_matrix", "Guide family vs hashtag family",
        path=path,
        inline=ct.reset_index().to_dict("records"),
        columns=["guide_family"] + list(ct.columns),
        caption=(
            "Rows are the family of the cell's assigned guide; columns are the "
            "family implied by its hashtag combination. These come from two "
            "independent assays, so off-diagonal cells are doublets or index "
            "hops rather than a labelling choice."
        ),
        order=5,
    )
    reg.metric(
        "summary", "pct_family_conflict", "Guide/hashtag family conflict",
        round(rate, 2), unit="%",
        level="good" if rate < 2 else "warn" if rate < 8 else "poor",
        order=45,
    )
    reg.note(
        "crosschecks", "family_conflict", "Reading the family cross-check",
        (
            f"{n_conf:,} of {n_both:,} cells ({rate:.1f}%) carry a guide from "
            f"one family while their hashtag says another. This is a direct "
            f"estimate of the doublet and index-hop rate, and it is independent "
            f"of any doublet caller. Those cells are excluded from knockdown, "
            f"E-distance and DE, because they would carry a second population's "
            f"transcriptome into whichever comparison group they landed in."
        ),
        level="warn" if rate >= 8 else "info",
        order=6,
    )
    return ct


def run_crosscheck_stage(
    guide_per_cell: pd.DataFrame | None,
    hto_calls: HTOCalls | None,
    qc: pd.DataFrame,
    cfg: PipelineConfig,
    reg: Registry,
) -> None:
    """Compare independent measurements against each other.

    Two independent failures that co-occur more than chance usually share a
    cause -- normally droplet quality -- and that is worth stating, because the
    remedy is different from fixing either assay.
    """
    if guide_per_cell is None or hto_calls is None:
        reg.skipped(
            "crosschecks", "guide_vs_hto", "Guide status vs hashtag call",
            "Needs both a guide matrix and a hashtag matrix; at least one is "
            "absent from this experiment.",
        )
        return

    common = guide_per_cell.index.intersection(hto_calls.per_cell.index)
    if len(common) == 0:
        reg.skipped(
            "crosschecks", "guide_vs_hto", "Guide status vs hashtag call",
            "Guide and hashtag tables share no barcodes.",
        )
        return

    gstat = np.where(
        guide_per_cell.loc[common, "guide_is_assigned"].to_numpy(bool),
        "guide assigned", "guide unassigned",
    )
    hclass = hto_calls.per_cell.loc[common, "hto_class"].to_numpy()

    ct = pd.crosstab(pd.Series(gstat, name="guide"), pd.Series(hclass, name="hashtag"))
    pct = ct / ct.to_numpy().sum() * 100.0
    ct.to_csv(cfg.table_dir / "crosscheck_guide_vs_hto_counts.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    P.heatmap(axes[0], ct, cfg.figures, cmap="Blues", annotate=True, fmt="{:,.0f}",
              cbar_label="cells")
    axes[0].set_title("cells")
    P.heatmap(axes[1], pct, cfg.figures, cmap="Blues", annotate=True, fmt="{:.1f}",
              cbar_label="% of all cells")
    axes[1].set_title("% of all cells")
    fig.tight_layout()

    reg.figure(
        "crosschecks", "guide_vs_hto", "Guide assignment vs hashtag call",
        P.save_figure(fig, cfg.fig_dir / "crosscheck_guide_vs_hto.png", cfg.figures),
        caption=T.HTO_CROSSCHECK_DESC, order=10, width="full",
    )

    # Quantify the association rather than leaving it to the eye.
    assigned = gstat == "guide assigned"
    negative = hclass == NEGATIVE
    p_neg = negative.mean()
    p_neg_given_unassigned = negative[~assigned].mean() if (~assigned).any() else np.nan
    if np.isfinite(p_neg_given_unassigned) and p_neg > 0:
        enrich = p_neg_given_unassigned / p_neg
        level = "warn" if enrich > 1.5 else "info"
        reg.note(
            "crosschecks", "association", "Do the two failures co-occur?",
            (
                f"Hashtag-negative cells make up {p_neg * 100:.1f}% of all cells, "
                f"but {p_neg_given_unassigned * 100:.1f}% of guide-unassigned cells "
                f"&mdash; an enrichment of {enrich:.2f}&times;. "
                + (
                    "That is well above chance, so both calls are likely being "
                    "driven by a shared cause such as low-quality or ambient-"
                    "dominated droplets, rather than by two independent assay "
                    "failures."
                    if enrich > 1.5
                    else "That is close to chance, so the two assays appear to be "
                    "failing independently."
                )
            ),
            level=level, order=20,
        )
