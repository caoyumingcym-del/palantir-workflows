"""
All human-readable prose for the report, in one place.

Much of this is adapted from the collaborator's ``15_build_model_report.py``,
whose interpretive language is the best part of that script: it tells the
reader what was done, how to read each panel, and what a bad result would look
like.  A QC report that shows plots without saying how to interpret them
mostly gets skimmed.

Two changes from the collaborator's version:

* Everything project-specific is parameterised.  Their text hardcoded gene
  names, model names (``ColO``, ``DopaN``, ...), the control label
  ``ONE_INTERGENIC_SITE``, and per-model interpretation dictionaries that had
  to be hand-maintained (and had already gone stale -- the file contains a
  block of "corrected interpretations ... override stale scrambled-era
  entries").  Here the numbers are formatted in from real results.

* Cluster interpretation is *derived* rather than hardcoded.  Their
  ``CLUSTER_INTERP`` dict mapped Leiden cluster IDs to hand-written biology
  for 14 cell models; that cannot survive a re-run at a different resolution,
  where cluster 3 is no longer the same cells.  ``describe_cluster`` builds a
  description from the cluster's own markers, size and cell-cycle composition.
"""
from __future__ import annotations

import html
from typing import Sequence


def esc(text: object) -> str:
    """HTML-escape any value.

    Applied to *every* interpolated value in the report.  Neither the original
    report builder nor the collaborator's escaped anything, so a gene name or
    file path containing ``<`` or ``&`` silently corrupted the page.
    """
    return html.escape(str(text), quote=True)


def _n(value: object, fmt: str = "{:,.0f}", dash: str = "&ndash;") -> str:
    """Format a number for prose, degrading to an en-dash when unknown.

    Borrowed from the collaborator's ``PERT_HOWTO(n)`` / ``PERT_SIM_DESC(n)``
    trick: when the count is unavailable the sentence still reads correctly
    rather than printing "None" or "0".
    """
    if value is None:
        return dash
    try:
        if value != value:      # NaN
            return dash
        return fmt.format(value)
    except (TypeError, ValueError):
        return esc(value)


# ===========================================================================
# Section-level descriptions
# ===========================================================================
SEQ_QC_DESC = (
    "Sequencing-level QC from the upstream DRAGEN run, summarised per library "
    "(10x lane). These numbers say whether enough reads were captured, and "
    "captured <em>in cells</em>, for the downstream calls to mean anything: a "
    "transcriptome analysis can survive shallow guide sequencing, but guide "
    "assignment cannot."
)

SEQ_QC_NOTE = (
    "Read <strong>fraction of reads in passing cells</strong> first. A high total "
    "read count with a low in-cell fraction means the library was sequenced but "
    "the reads went to ambient RNA or empty droplets, and adding more reads will "
    "not fix it. Mean reads per cell is reported per modality because gene "
    "expression and guide capture are sequenced to very different depths by design."
)

CELL_QC_DESC = (
    "Per-cell QC metrics with the applied cut-offs shown as dashed lines, plotted "
    "as hexbin density so the heavily-overlapping bulk of cells stays readable "
    "&mdash; colour is the number of cells per hexagonal bin (log scale). The lower "
    "bounds remove low-complexity and ambient droplets; the %mito ceiling removes "
    "stressed and dying cells; the upper count/gene bounds trim probable "
    "multiplets. Cells outside the bounds are excluded from everything downstream."
)

CELL_QC_NOTE = (
    "Upper bounds deserve scepticism. A symmetric median-absolute-deviation rule "
    "over-trims the healthy high-RNA tail, so the automatic upper gates here are "
    "deliberately looser than the lower ones, and are computed on log-transformed "
    "counts because UMI and gene counts are roughly log-normal. If a real "
    "population in your experiment is genuinely high-RNA, set the upper bounds by "
    "hand rather than accepting the automatic ones."
)

THRESHOLD_PROVENANCE_NOTE = (
    "Every threshold below records where it came from. <em>auto</em> means it was "
    "derived from this experiment's own distributions; <em>manifest</em> and "
    "<em>cli</em> mean a human set it. Automatic thresholds are a starting point "
    "for a first look, not a substitute for inspecting the distributions."
)

QC_COMPARISON_DESC = (
    "The same per-cell QC metrics split by experimental condition. The filtering "
    "thresholds are <strong>global</strong> &mdash; one set of cut-offs applied to "
    "every cell regardless of which condition it came from &mdash; so these panels "
    "are diagnostic: they show whether a single global gate is defensible, or "
    "whether one condition is being disproportionately filtered."
)

RETENTION_DESC = (
    "Cells recovered and retained relative to the number loaded, per sample. This "
    "is the end-to-end yield of the experiment: loading, capture, sequencing and "
    "QC filtering combined."
)

# --- transcriptome ---------------------------------------------------------
EMBEDDING_DESC = (
    "<p><strong>What was done.</strong></p><ul>"
    "<li>Cells were QC-filtered on total counts, genes detected and % mitochondrial "
    "reads.</li>"
    "<li>Passing cells were normalised to a fixed count depth and "
    "log1p-transformed, and cell-cycle phase was scored.</li>"
    "<li>The top highly variable genes &mdash; <strong>excluding mitochondrial and "
    "ribosomal genes</strong> &mdash; were taken as features, and sequencing depth, "
    "%mito and S/G2M cell-cycle scores were regressed out.</li>"
    "<li>Cells were then scaled and embedded with PCA &rarr; nearest-neighbour "
    "graph &rarr; UMAP &rarr; Leiden clustering.</li>"
    "<li>Tiny Leiden fragments are merged into their nearest cluster in PCA space, "
    "so no cells are dropped.</li>"
    "<li>Per-cluster marker genes come from a one-vs-rest test (mitochondrial and "
    "ribosomal genes are excluded from this step too, so clusters get biologically "
    "interpretable markers).</li>"
    "</ul>"
)

UMAP_LEIDEN_DESC = (
    "UMAP of QC-passing cells coloured by Leiden cluster, with cluster labels drawn "
    "on the clusters. Each cluster is a transcriptionally distinct cell state; "
    "annotations are listed below."
)

UMAP_PHASE_DESC = (
    "The same UMAP coloured by inferred cell-cycle phase (G1 / S / G2M). Comparing "
    "it with the Leiden panel shows whether a cluster reflects cell-cycle state "
    "rather than a distinct cell identity."
)

UMAP_DOUBLET_DESC = (
    "UMAP of <strong>all</strong> QC-passing cells, coloured by the doublet call. "
    "Predicted doublets are <strong>annotated only and retained</strong> &mdash; "
    "they are <strong>not</strong> removed from this embedding and <strong>not"
    "</strong> excluded from any downstream analysis. This panel exists to "
    "visualise the predicted-doublet rate and where the calls fall. Doublets "
    "scattered uniformly across clusters indicate over-calling on a homogeneous "
    "population; a distinct island or bridge would indicate genuine multiplets."
)

UMAP_TOTAL_COUNTS_DESC = (
    "The same embedding coloured by total UMI counts per cell (library size). The "
    "colour scale is capped at the 99th percentile so rare high-count doublets do "
    "not compress the range for the bulk of cells. Compare with the doublet panel: "
    "genuine multiplets tend to track a high-count region, whereas over-called "
    "doublets do not."
)

UMAP_BATCH_DESC = (
    "The same embedding coloured by sample and by condition. This is the batch "
    "check: clusters that correspond one-to-one with a sample are a batch effect, "
    "not biology. If they do, re-run with batch correction enabled and compare."
)

MARKERS_DESC = (
    "Top marker genes per Leiden cluster (one-vs-rest rank-sum test; "
    "mitochondrial and ribosomal genes excluded). Dot size is the fraction of "
    "cells in the cluster expressing the gene; colour is mean expression. These "
    "markers drive the cluster interpretation below."
)

CLUSTER_INTERP_NOTE = (
    "Cluster descriptions below are generated from each cluster's own markers, "
    "size and cell-cycle composition. They are a starting point for annotation, "
    "not an annotation &mdash; assigning cell identity requires domain knowledge "
    "this pipeline does not have."
)

# --- guides ----------------------------------------------------------------
def guide_section_desc(
    min_reads: int, purity_min: float, dual_guide: bool = False
) -> str:
    """Guide-assignment description, with the actual rule used."""
    base = (
        "Quality control of guide-RNA (gRNA) assignment. A guide is assigned to a "
        f"cell only if the cell has more than <strong>{esc(min_reads)}</strong> "
        "total guide UMIs <strong>and</strong> guide purity &mdash; "
        "top1 / (top1 + top2) &mdash; exceeds "
        f"<strong>{esc(f'{purity_min:g}')}%</strong>."
    )
    if dual_guide:
        base += (
            " This library is dual-guide: each construct carries a pair of barcoded "
            "guides (iBAR1 + iBAR2) targeting the same gene, and each position is "
            "thresholded <strong>independently</strong>. Because the two positions "
            "are scored separately, a cell can legitimately carry 0, 1 or 2 "
            "assigned guides."
        )
    return base

GUIDE_PURITY_NOTE = (
    "Two different quantities are commonly both called \"purity\", and they answer "
    "different questions. <strong>top1/(top1+top2)</strong> asks whether there is a "
    "clear winner between the best two guides &mdash; it is the assignment "
    "criterion, and it reads 100% for a cell with exactly one guide. "
    "<strong>top1/total</strong> asks whether that winner dominates the whole cell, "
    "and is the stricter test: a cell with one big guide and a long tail of "
    "ambient guide reads scores high on the first and low on the second. Both are "
    "reported here so a purity number is never ambiguous."
)

PURITY_SWEEP_DESC = (
    "Fraction of cells receiving a guide assignment as the purity cut-off is "
    "swept. A curve with a long flat plateau means the choice of cut-off barely "
    "matters and assignment is robust; a steep slope through the chosen threshold "
    "means the result is sensitive to it, and the number of assigned cells should "
    "be treated as approximate."
)

PURITY_GATE_DESC = (
    "Per-cell top1/total against (top1+top2)/total, as hexbin density with the "
    "stricter secondary gate drawn on. Clean single-guide cells sit in the "
    "top-right corner. A dense population along the diagonal at low top1/total is "
    "the signature of ambient guide contamination or genuine high-MOI infection."
)

MOI_DESC = (
    "Number of distinct guides detected per cell &mdash; the multiplicity-of-"
    "infection proxy. For a single-guide screen the mass should sit at 1, with the "
    "0 bin reflecting capture failure. A heavy tail above 1 means either genuine "
    "multiple infection or ambient guide reads bleeding across droplets; the "
    "purity panels distinguish these, because ambient contamination produces many "
    "detected guides but still a dominant top1."
)

GUIDE_ABUNDANCE_DESC = (
    "Guide representation across the library, ranked from most to least abundant, "
    "with the Gini coefficient quantifying skew (0 = perfectly even, 1 = one guide "
    "takes everything). A steep curve means some guides are effectively absent, "
    "and any perturbation carried only by those guides is underpowered no matter "
    "how good the rest of the experiment is."
)

GUIDE_EFFICIENCY_DESC = (
    "Assignment rate and guide sequencing depth per condition. This is the panel "
    "to read when comparing protocols: a condition with a lower assignment rate at "
    "equal depth has a real capture problem, whereas one with a lower rate and "
    "lower depth may simply need more reads."
)

RECOMBINATION_DESC = (
    "For dual-guide constructs, the two guide positions in a cell should belong to "
    "the same designed construct. Cells carrying two guides from <em>different</em> "
    "constructs indicate template switching or recombination during library "
    "preparation, and their apparent perturbation is a mixture of two."
)

# --- perturbation ----------------------------------------------------------
def kd_desc(ntc_label: str) -> str:
    return (
        "Target-gene knockdown per perturbation, computed as "
        "<code>(1 &minus; mean(perturbed) / mean(control)) &times; 100</code> on "
        "the <strong>linear</strong> expression scale against "
        f"<strong>{esc(ntc_label)}</strong> control cells. Inverting the log "
        "transform before averaging matters: the mean of logs is not the log of "
        "means, and averaging in log space systematically understates knockdown."
    )

KD_NOTE = (
    "A knockdown value is only interpretable alongside its cell count and the "
    "control-cell expression of the target. A gene that is undetectable in control "
    "cells has no defined knockdown &mdash; it is reported as missing rather than "
    "as 0% or 100%, because either number would be an invention. Perturbations "
    "below the minimum cell count are excluded and listed separately, since "
    "small-n knockdown estimates are dominated by sampling noise."
)

RESAMPLE_NOTE = (
    "Significance comes from a <strong>conditional resampling</strong> test rather "
    "than a parametric one: the null distribution of the fold change is built by "
    "repeatedly drawing groups of the same size from the control pool and "
    "recomputing the statistic. This absorbs the depth and group-size differences "
    "that make a plain rank test anti-conservative on single-cell counts. It is "
    "the same idea as SCEPTRE's calibrated test, implemented here directly so the "
    "pipeline has no R dependency; it is not the SCEPTRE algorithm and should not "
    "be reported as such."
)

EDISTANCE_DESC = (
    "E-distance between each perturbation and control in PCA space &mdash; a "
    "single number for \"how far did the whole transcriptome move?\", as opposed "
    "to knockdown, which only asks whether the intended gene went down. A "
    "perturbation with strong knockdown but near-zero E-distance knocked the gene "
    "down without downstream consequence in these cells."
)

EDISTANCE_NOTE = (
    "E-distance here is the squared-distance energy statistic used in the "
    "Perturb-seq literature, not the classical unsquared Szekely-Rizzo energy "
    "distance. It is <strong>biased upward at small cell numbers</strong>, so raw "
    "values are not comparable between perturbations with very different n. The "
    "accompanying permutation p-value is the number to compare across "
    "perturbations."
)

def deg_howto(
    n_perturbations: int | None,
    padj_max: float,
    log2fc_min: float,
    min_frac_ntc: float,
    top_n: int,
    ntc_label: str,
) -> str:
    """The DEG-selection method note, adapted from the collaborator's PERT_HOWTO."""
    npert = _n(n_perturbations, "{:,.0f} ")
    npert = "" if npert == "&ndash;" else npert
    return (
        "<p><strong>How the DEGs are selected.</strong></p><ul>"
        f"<li>For each of the {npert}target-gene knockdowns, every gene is assigned "
        "a log<sub>2</sub> fold-change and a Benjamini-Hochberg-adjusted "
        "<em>p</em>-value versus the non-targeting "
        f"<code>{esc(ntc_label)}</code> control.</li>"
        f"<li>A gene is a <strong>DEG</strong> when adjusted <em>p</em> &lt; "
        f"{esc(f'{padj_max:g}')} <strong>and</strong> |log<sub>2</sub>FC| &gt; "
        f"{esc(f'{log2fc_min:g}')}.</li>"
        "<li><strong>Lowly-expressed genes are removed</strong> &mdash; a gene must "
        f"be detected in &ge;{esc(f'{min_frac_ntc*100:g}')}% of the non-targeting "
        "control cells, which drops noisy near-zero-count genes with huge "
        "fold-changes.</li>"
        f"<li>For each perturbation the <strong>top {esc(top_n)} DEGs</strong> are "
        "kept, ranked by <strong>significance</strong> (smallest adjusted "
        "<em>p</em>; ties broken by larger |log<sub>2</sub>FC|).</li>"
        "<li>The perturbation's <strong>own knocked-down gene is always shown "
        "first</strong> in its block.</li>"
        "<li>Columns are <strong>grouped into one block per perturbation</strong>, "
        "ordered by transcriptome-wide similarity.</li>"
        "<li><strong>Red x-axis gene labels</strong> flag genes that are a top DEG "
        "for <strong>more than one</strong> perturbation (a shared response); black "
        "labels are unique to one perturbation.</li>"
        "<li><strong>Dot size</strong> is &minus;log<sub>10</sub>(adjusted "
        "<em>p</em>); <strong>dot colour</strong> is log<sub>2</sub>FC (red = up, "
        "blue = down).</li>"
        "</ul>"
    )

def similarity_desc(n: int | None) -> str:
    """Adapted from the collaborator's PERT_SIM_DESC."""
    nn = _n(n, "{:,.0f}")
    size = "" if nn == "&ndash;" else f"{nn}&times;{nn} "
    return (
        "<p><strong>What this shows.</strong></p><ul>"
        f"<li>A {size}perturbation&ndash;perturbation matrix. <strong>Upper triangle "
        "= Jaccard index</strong> of the two DEG sets; a non-linear (square-root) "
        "colour scale is used because most overlaps are small.</li>"
        "<li><strong>Lower triangle = Spearman correlation</strong> of the "
        "log<sub>2</sub>FC profiles over a <strong>fixed</strong> gene set (every "
        "gene that is a DEG in <em>any</em> perturbation), so values are comparable "
        "across all pairs.</li>"
        "<li>Perturbations are <strong>hierarchically clustered</strong> on the "
        "Spearman matrix.</li>"
        "<li>The <strong>top bar</strong> is the number of DEGs per perturbation &mdash; "
        "a near-zero bar flags a sparse responder whose correlations are "
        "under-powered.</li></ul>"
    )

PERT_SCORE_DESC = (
    "Cells ranked by a continuous perturbation score: the projection of each cell "
    "onto its own perturbation's expression signature. Target-gene expression is "
    "shown alongside, with control cells appended for reference. A perturbation "
    "whose target expression falls smoothly with the score is behaving as intended; "
    "one where high-score cells still express the target has an off-target or "
    "indirect signature rather than knockdown."
)

TRANSCRIPTOME_LOG2FC_DESC = (
    "Transcriptome-wide log<sub>2</sub> fold-change distribution per perturbation, "
    "with the perturbation's own target gene highlighted. A well-behaved "
    "perturbation shows a distribution centred near zero with its target as a "
    "clear negative outlier. A distribution shifted bodily away from zero usually "
    "means a depth or composition artefact rather than thousands of real changes."
)

# --- hashtags --------------------------------------------------------------
def hto_desc(mode: str, threshold_repr: str) -> str:
    return (
        "Hashtag demultiplexing. Counts are log1p-transformed and each hashtag is "
        "then centred by its own mean across cells &mdash; Seurat's "
        "<code>CLR, margin=2</code> convention for hashtag and antibody data, "
        "which puts every hashtag on a comparable scale so one threshold is "
        "meaningful across hashtags of different overall abundance. "
        f"Positive calls use the <strong>{esc(mode)}</strong> rule "
        f"({esc(threshold_repr)}). Cells with 0 positive hashtags are negative, "
        "1 is a singlet, and 2 or more is a multiplet."
    )

HTO_NOTE = (
    "This is a simplified demultiplexer: an independent per-hashtag background/"
    "signal split with an empirical threshold, rather than a joint model. It is "
    "sufficient for judging <em>hashtag performance</em>, which is what this report "
    "is for. If the hashtag assignment itself is a scientific result, use a "
    "dedicated demultiplexer and compare. The threshold-sweep and per-hashtag "
    "diagnostic panels below exist so a badly-separated hashtag is visible rather "
    "than silently mis-called."
)

HTO_SWEEP_DESC = (
    "Singlet, doublet and negative rates as the positivity threshold is swept. As "
    "with the guide purity sweep, a flat region means the calls are robust to the "
    "exact cut-off. A monotone trade-off with no plateau means the "
    "singlet/doublet split is essentially a choice of threshold, and the reported "
    "doublet rate should be quoted with that caveat."
)

HTO_DIAGNOSTIC_DESC = (
    "Per-hashtag normalised-intensity distributions with the chosen threshold "
    "drawn on. Each panel should be visibly bimodal, with the threshold sitting in "
    "the valley. A unimodal panel means that hashtag did not work, and every call "
    "involving it is unreliable regardless of what the summary rates say."
)

HTO_HEATMAP_DESC = (
    "Normalised hashtag intensity per cell, grouped by call. A clean experiment "
    "shows a strong diagonal: cells called for a hashtag are high in that hashtag "
    "and low in all others. Off-diagonal signal is cross-contamination or "
    "ambient hashtag."
)

HTO_CROSSCHECK_DESC = (
    "Guide-assignment status against hashtag call. These are two independent "
    "measurements on the same cells, so their agreement is informative: if "
    "hashtag-negative cells are also disproportionately guide-unassigned, both "
    "are being driven by a shared cause &mdash; usually low-quality droplets "
    "&mdash; rather than by two separate failures."
)

COMPOSITION_DESC = (
    "Sample composition by hashtag-derived identity. For a multiplexed experiment "
    "this is the pooling check: substantial departures from the intended ratios "
    "point to a pooling or capture bias affecting one population."
)

# ===========================================================================
# Derived, data-driven interpretation
# ===========================================================================
def describe_cluster(
    cluster_id: str,
    n_cells: int,
    frac: float,
    markers: Sequence[str],
    phase_frac: dict[str, float] | None = None,
    top_sample: tuple[str, float] | None = None,
    doublet_frac: float | None = None,
) -> str:
    """Build a one-line description of a cluster from its own statistics.

    This replaces the collaborator's hand-maintained ``CLUSTER_INTERP``
    dictionary.  That dictionary was the most fragile thing in their script:
    cluster identities are keyed by integer ID, but Leiden IDs are not stable
    across resolutions, subsets or software versions, so the annotations
    silently detach from the clusters they describe on the next run.  A
    description generated from the cluster's current markers cannot go stale.
    """
    bits = [
        f"{_n(n_cells)} cells ({frac * 100:.1f}%)"
    ]
    if markers:
        shown = ", ".join(esc(m) for m in list(markers)[:6])
        bits.append(f"markers <em>{shown}</em>")
    if phase_frac:
        dominant = max(phase_frac, key=lambda k: phase_frac[k])
        if phase_frac[dominant] > 0.60:
            bits.append(
                f"{phase_frac[dominant] * 100:.0f}% {esc(dominant)} &mdash; likely a "
                f"cell-cycle cluster rather than a distinct identity"
            )
    if top_sample and top_sample[1] > 0.80:
        bits.append(
            f"{top_sample[1] * 100:.0f}% from sample {esc(top_sample[0])} "
            f"&mdash; check for a batch effect"
        )
    if doublet_frac is not None and doublet_frac > 0.30:
        bits.append(
            f"{doublet_frac * 100:.0f}% predicted doublets &mdash; likely a "
            f"multiplet cluster"
        )
    return "; ".join(bits)


def verdict(label: str, value: float | None, good: float, poor: float,
            higher_is_better: bool = True) -> tuple[str, str]:
    """Classify a metric into pass/warn/fail with a short reason.

    Returns ``(level, message)`` where level is "good", "warn", "poor" or
    "unknown".  Used to build the report's summary scorecard.  Thresholds are
    passed in by the caller rather than hardcoded here, because what counts as
    an acceptable assignment rate depends on the assay.
    """
    if value is None or value != value:
        return "unknown", f"{label}: not measured"
    if higher_is_better:
        if value >= good:
            return "good", f"{label}: {value:.1f}"
        if value >= poor:
            return "warn", f"{label}: {value:.1f}"
        return "poor", f"{label}: {value:.1f}"
    if value <= good:
        return "good", f"{label}: {value:.1f}"
    if value <= poor:
        return "warn", f"{label}: {value:.1f}"
    return "poor", f"{label}: {value:.1f}"


FINAL_CHECKLIST = [
    "Do the QC thresholds actually match this experiment's distributions, or were "
    "they taken from the automatic defaults without being looked at?",
    "Does any Leiden cluster correspond one-to-one with a single sample or lane? "
    "If so, that is a batch effect being reported as biology.",
    "Is the guide assignment rate consistent across conditions at comparable "
    "sequencing depth?",
    "For every perturbation you intend to report: is its cell count large enough, "
    "and is the target gene detectable in control cells at all?",
    "Is each hashtag's intensity distribution visibly bimodal? A unimodal hashtag "
    "invalidates every call involving it.",
    "Do guide-unassigned and hashtag-negative cells overlap more than chance? That "
    "points to a shared droplet-quality cause rather than two independent "
    "failures.",
    "Are E-distance comparisons being made between perturbations with similar cell "
    "numbers? The statistic is biased upward at small n.",
]
