"""
Central configuration for the Perturb-seq report pipeline.

Every tunable number in the pipeline lives here, in a dataclass, with a
documented default.  Nothing downstream reads a module-level global: config
objects are passed explicitly into every stage.  This replaces the ~50
scattered ALL_CAPS constants of the previous notebook-style script, where
functions silently bound globals as default arguments at *definition* time
(so editing the global after import had no effect -- a real bug in the
original).

Load order for any setting, lowest priority first:

    dataclass default  <  config YAML/JSON  <  manifest column  <  CLI flag
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, asdict, replace
from pathlib import Path
from typing import Any, Sequence


# ===========================================================================
# QC thresholds
# ===========================================================================
@dataclass
class QCThresholds:
    """Per-cell QC gates applied to the GEX matrix.

    ``None`` means "decide from the data" (see ``qc.auto_thresholds``).  This
    is the key difference from the original pipeline, which hardcoded
    MIN_GENES=1000 / MAX_GENES=4000 / MIN_COUNTS=500 / MAX_COUNTS=12500 --
    numbers appropriate to one experiment and silently wrong for the next.

    ``source`` records provenance ("auto", "manifest", "cli", "config") so the
    report can state where each number came from.
    """

    min_genes: float | None = None
    max_genes: float | None = None
    min_counts: float | None = None
    max_counts: float | None = None
    max_mito: float | None = None

    source: dict[str, str] = field(default_factory=dict)

    # -- how auto thresholds are derived -----------------------------------
    # Median-absolute-deviation multiplier for the *lower* gates.  Lower
    # bounds remove low-complexity / ambient droplets, and a symmetric MAD
    # rule is well behaved there.
    mad_lower: float = 3.0
    # MAD multiplier for the *upper* gates.  Deliberately looser than the
    # lower one: the collaborator's report notes that "the symmetric MAD rule
    # over-trims the healthy high-RNA tail", which is correct -- real
    # high-RNA cells are not artefacts.  5 MAD is a compromise; set
    # max_counts/max_genes explicitly if you care about the exact number.
    mad_upper: float = 5.0
    # Hard floor for the auto lower gates, so a pathological run can't
    # produce a threshold of ~0 and let empty droplets through.
    auto_min_genes_floor: float = 200.0
    auto_min_counts_floor: float = 500.0
    # Auto %mito ceiling: max(mad rule, this floor).  A MAD rule alone can
    # produce an absurdly tight ceiling (e.g. 3%) in a very clean run and
    # throw away good cells.
    auto_max_mito_floor: float = 10.0
    auto_max_mito_cap: float = 30.0

    def is_complete(self) -> bool:
        return all(
            getattr(self, k) is not None
            for k in ("min_genes", "max_genes", "min_counts", "max_counts", "max_mito")
        )

    def missing(self) -> list[str]:
        return [
            k
            for k in ("min_genes", "max_genes", "min_counts", "max_counts", "max_mito")
            if getattr(self, k) is None
        ]

    def as_dict(self) -> dict[str, float | None]:
        return {
            k: getattr(self, k)
            for k in ("min_genes", "max_genes", "min_counts", "max_counts", "max_mito")
        }


# ===========================================================================
# Modality discovery
# ===========================================================================
@dataclass
class ModalityConfig:
    """How to find the GEX / guide / HTO submatrices inside one .h5ad.

    Real-world Perturb-seq h5ads store guides and hashtags in at least four
    different places depending on which upstream tool wrote them, so every
    lookup is a candidate list tried in order.
    """

    # 1. var-level feature-type annotation (CellRanger / DRAGEN convention)
    feature_type_cols: tuple[str, ...] = ("feature_types", "feature_type")
    guide_feature_type_tokens: tuple[str, ...] = (
        "guide", "crispr", "grna", "sgrna", "guide_capture",
    )
    hto_feature_type_tokens: tuple[str, ...] = (
        "hashtag", "antibody", "multiplexing", "hto", "custom",
    )

    # 2. obsm matrices
    guide_obsm_keys: tuple[str, ...] = (
        "gRNA_counts", "guide_counts", "CRISPR_counts", "grna_counts", "sgRNA_counts",
    )
    hto_obsm_keys: tuple[str, ...] = ("HTO_counts", "hto_counts", "hashtag_counts")

    # 3. uns feature-name vectors that name the columns of those matrices
    guide_feature_uns_keys: tuple[str, ...] = (
        "gRNA_features", "guide_features", "CRISPR_features", "grna_features",
    )
    hto_feature_uns_keys: tuple[str, ...] = (
        "HTO_features", "hto_features", "hashtag_features",
    )

    # 4. obs columns, one per hashtag (some pipelines flatten HTOs to obs)
    hto_obs_prefixes: tuple[str, ...] = ("prot:hash.", "hto_", "HTO_", "hash.")
    hto_obs_exclude_suffixes: tuple[str, ...] = ("_CLR", "_clr", "_log", "_norm")
    # Set True if those obs columns hold log1p values rather than raw counts.
    hto_obs_cols_are_log1p: bool = False

    # 5. DRAGEN cellhashing.tsv, read directly from the manifest's dragen_path
    # when the h5ad has no hashtag matrix at all (none of 1-4 found anything).
    # Candidate filenames, most specific first; "{prefix}" is the manifest's
    # per-run prefix column, matching the convention in seqmetrics.py.
    hto_dragen_file_patterns: tuple[str, ...] = (
        "{prefix}.scRNA.cellhashing.tsv",
        "{prefix}.scRNA.cellhashing.tsv.gz",
        "{prefix}.cellhashing.tsv",
        "{prefix}_cellhashing.tsv",
        "*.scRNA.cellhashing.tsv",
        "*cellhashing.tsv",
    )
    # Barcode transforms tried, in order, when matching cellhashing.tsv
    # barcodes against the h5ad's obs_names. The first transform that recovers
    # at least `hto_dragen_min_match_frac` of the h5ad's barcodes is used.
    hto_dragen_min_match_frac: float = 0.5

    # Pre-existing doublet annotations (we never *drop* doublets; see gex.py).
    doublet_score_col_candidates: tuple[str, ...] = (
        "doublet_score", "scrublet_score", "predicted_doublet_score",
    )
    doublet_call_col_candidates: tuple[str, ...] = (
        "predicted_doublet", "scrublet_call", "doublet_call", "is_doublet",
    )

    sample_col_candidates: tuple[str, ...] = (
        "sample", "sample_id", "Sample", "library", "batch",
    )
    cluster_col_candidates: tuple[str, ...] = (
        "leiden", "leiden_clusters", "louvain", "clusters", "leiden_gpu",
    )
    # Gene-name prefixes used to compute QC fractions and to *exclude* genes
    # from HVG selection and marker tests (a practice we take from the
    # collaborator's pipeline -- it stops clusters being defined by stress
    # and translation programmes).
    mito_prefixes: tuple[str, ...] = ("MT-", "mt-", "Mt-")
    ribo_prefixes: tuple[str, ...] = ("RPS", "RPL", "Rps", "Rpl")


# ===========================================================================
# Guide / perturbation calling
# ===========================================================================
@dataclass
class GuideConfig:
    """Guide-RNA assignment.

    NOTE on the two purity definitions.  The original pipeline used *two*
    different metrics both called "purity":

        section 10:  purity     = 100 * top1 / (top1 + top2)
        section 12:  top1_ratio = top1 / total
                     top12_ratio = (top1 + top2) / total

    That is a genuine source of confusion -- two numbers, same name, different
    meaning, reported side by side.  Here they are named distinctly and both
    are computed once, in one place (``guide.compute_guide_stats``):

        top1_over_top2 : top1 / (top1 + top2)   -- "is there a clear winner?"
        top1_over_total: top1 / total           -- "does the winner dominate?"
        top12_over_total                        -- "is this a clean 1- or 2-guide cell?"

    Assignment uses ``min_reads`` + ``purity_min`` (on top1_over_top2), which
    matches the collaborator's stated rule: ">10 total guide reads and guide
    purity -- top1 / (top1 + top2) -- >75%".
    """

    min_reads: int = 10
    purity_min: float = 75.0          # percent, on top1_over_top2
    purity_sweep: tuple[float, ...] = tuple(float(x) for x in range(100, -1, -5))

    # Stricter secondary gate, reported as a diagnostic hexbin
    gate_top1_over_total_min: float = 0.80
    gate_top12_over_total_min: float = 0.95

    # Dual-guide ("iBAR") libraries: each construct carries two barcoded
    # guides targeting the same gene, thresholded independently, so a cell can
    # legitimately carry 0, 1 or 2 assigned guides.  Detected automatically
    # from guide-ID structure unless forced here.
    dual_guide: bool | None = None          # None = autodetect
    # Only patterns with an EXPLICIT position marker are included by default.
    # A generic `^(.+?)[._-]([12])$` would also match the extremely common
    # single-guide naming `GENE_1` / `GENE_2`, which would falsely trigger
    # dual-guide detection and make every two-guide cell look like a
    # recombinant. If your library uses a bare numeric suffix to mean iBAR
    # position, add that pattern here explicitly.
    ibar_regexes: tuple[str, ...] = (
        r"^(?P<construct>.+?)[._-](?:iBAR|ibar|IBAR)[._-]?(?P<ibar>[12])$",
        r"^(?P<construct>.+?)[._-](?:pos|position)[._-]?(?P<ibar>[12])$",
    )

    # Guide-ID -> (target_gene, target_ensg) parsing, tried in order.
    #
    # v1.2.0: the first two patterns are new. v1.1.0 could not parse the
    # "{GENE}_target_version_{V}_spacer_number_{N}_spacer_target_{TARGET}_{SPACER}"
    # scheme at all -- 168 of 321 guides in MDL-1856 fell through to using the
    # raw guide ID as the target name, which fragmented 77 real targets into
    # 186 pseudo-targets and made the ENSG (sitting inside the ID) invisible to
    # perturb.resolve_gene(). The pattern is carried over from the previous
    # generation script's GUIDE_ID_PATTERNS[0], which had it right.
    guide_id_regexes: tuple[str, ...] = (
        # {GENE}_target_version_{V}_spacer_number_{N}_spacer_target_{TARGET}_{SPACER}
        #
        # `_spacer_target_+` -- one or MORE underscores -- is deliberate and
        # load-bearing: control IDs in this scheme carry a doubled underscore,
        #   INTERGENIC_CONTROL_target_version_7.0_spacer_number_1_spacer_target__INTERGENIC_CONTROL_TAAAT...
        # and a single `_` silently fails to match every one of them.
        # The `ensg` group is really "the second name": an Ensembl ID for a
        # targeting guide, or a control-site token for a control. The control
        # alternative must allow underscores on BOTH sides of INTERGENIC --
        # `ONE_INTERGENIC_SITE` is a real token in this library, and a class of
        # `[A-Z0-9]*` before it silently drops all 26 of them into the looser
        # pattern below, losing the target field entirely.
        r"^(?P<gene>.+?)_target_version_(?P<version>[\d.]+)"
        r"_spacer_number_(?P<spacer_num>\d+)"
        r"_spacer_target_+(?P<ensg>ENS[A-Z]*G\d+|[A-Z0-9_]*INTERGENIC[A-Z0-9_]*)"
        r"_(?P<spacer>[ACGT]{15,})$",
        # Same scheme, no trailing spacer or a target token we do not recognise.
        r"^(?P<gene>.+?)_target_version_(?P<version>[\d.]+)"
        r"_spacer_number_(?P<spacer_num>\d+)(?:_.*)?$",
        # GENE|ENSG00000123456|spacer-ish structured IDs
        r"^(?P<gene>[^|]+)\|(?P<ensg>ENS[A-Z]*G\d+)(?:\|.*)?$",
        # GENE_ENSG00000123456_...
        r"^(?P<gene>[^_]+)_(?P<ensg>ENS[A-Z]*G\d+)(?:_.*)?$",
        # GENE_singleguide / GENE-singleguide
        r"^(?P<gene>.+?)[._-]singleguide.*$",
        # GENE_1_ACGT... (gene, index, spacer)
        r"^(?P<gene>[A-Za-z0-9.\-]+?)_(?P<idx>\d+)_(?P<spacer>[ACGT]{15,})$",
        # GENE_sg1 / GENE-g2 / GENE_guide3
        r"^(?P<gene>.+?)[._-](?:sg|g|guide)(?P<idx>\d+)$",
    )

    # Which of the two names in a guide ID is authoritative.
    #
    # Many IDs carry two: a gene-symbol prefix and a `spacer_target` field.
    # They disagree exactly where it matters --
    #   ABT1_target_version_1.0_..._spacer_target_ENSG00000146109_...  -> agree
    #   ABT1_target_version_1.1_..._spacer_target_ONE_INTERGENIC_SITE_... -> DISAGREE
    # The `.1` versions are per-gene MATCHED INTERGENIC CONTROLS: same
    # backbone, same prefix, spacer redirected to an intergenic site. The
    # prefix records what the construct was designed from; the spacer target
    # records what it actually cuts.
    #
    #   "second" (default) -- trust the spacer target. Classifies matched
    #                         controls correctly with no special case.
    #   "first"            -- trust the gene-symbol prefix. For libraries whose
    #                         spacer_target field is unreliable or absent.
    #
    # Overridable per guide via the `parse_target_from` column of the gRNA
    # whitelist; this is only the default when that column is absent.
    parse_target_from: str = "second"

    # Anything matching this is a non-targeting control regardless of which
    # regex above matched.  The original had this logic smeared across three
    # separate ad-hoc checks; it is one list here.
    # Matched case-insensitively against the WHOLE guide ID. A leading
    # boundary is required so a gene like "PNTCX" is not mistaken for an NTC,
    # but nothing is required after the token: real control names run words
    # together ("NonTargetingControl", "ONE_INTERGENIC_SITE"), and requiring a
    # trailing delimiter is what made the original miss them and need three
    # separate patches.
    #
    # v1.2.0: this is now a FALLBACK. It runs only when no structured target
    # field was captured. Applying it to the whole ID unconditionally (v1.1.0)
    # got the right answer on this library by luck and would misclassify a gene
    # legitimately named e.g. CTRL1.
    ntc_regex: str = (
        r"(?i)(^|[._|\-])("
        r"ntc|non[-_]?targeting|scramble|safe[-_]?harbou?r|intergenic|"
        r"no[-_]?target|neg[-_]?control|negative[-_]?control|"
        r"ctrl|control"
        r")"
    )
    # A structured target field matching this is a control site.
    control_target_regex: str = r"(?i)(intergenic|safe[-_]?harbou?r|no[-_]?site)"
    ntc_label: str = "NTC"

    # ---------------------------------------------------------------- family
    # A "family" is one guide population: the set of guides whose NTCs form a
    # valid control group for each other. It is declared per guide in the gRNA
    # whitelist, never inferred -- `NTC_10_ACGT...` carries no library
    # information, and two libraries can independently use `NTC_10`.
    #
    # An experiment with four cell lines and four libraries has FOUR distinct
    # NTC populations. v1.1.0 pooled all of them into one baseline and measured
    # every knockdown, E-distance and DE comparison against it.
    pool_ntc_across_families: bool = False
    # Family assigned to guides absent from the whitelist. Treated as its own
    # family for control scoping -- never merged into a declared one.
    unassigned_family: str = "unassigned"
    # Separator between the target/guide name and the family suffix in labels.
    family_label_sep: str = "_"
    # Marker appended to the gene prefix of a matched intergenic control, so
    # `ABT1`'s matched control reads `ABT1ic` and stays traceable to its gene
    # even though its target_gene is NTC.
    matched_control_marker: str = "ic"


@dataclass
class PerturbConfig:
    """Perturbation-effect quantification."""

    min_cells_per_group: int = 10
    # Differential expression per perturbation vs NTC
    de_padj_max: float = 0.05
    de_abs_log2fc_min: float = 0.5
    # A gene must be detected in >= this fraction of NTC cells to be
    # considered at all: drops near-zero-count genes with huge, meaningless
    # fold-changes.  Straight from the collaborator's method note.
    de_min_frac_detected_in_ntc: float = 0.10
    de_top_n_per_perturbation: int = 10
    # Caps for the DEG dot plot. 40 perturbations x 10 DEGs is ~400 gene
    # columns, which renders as a figure several metres wide with every label
    # overprinting. The CSVs keep everything; the figure shows a legible slice.
    dotplot_max_genes: int = 120
    dotplot_max_targets: int = 30
    # Differential expression must examine every gene, but it does so in column
    # blocks so peak memory is bounded by this many genes rather than by the
    # width of the transcriptome. Lower it if memory is tight; it does not
    # change the result, only the allocation size.
    de_gene_block: int = 2000

    # Dot/colour scaling for the top-DEG dotplot
    dot_neglog10p_cap: float = 20.0
    dot_log2fc_clip: float = 3.0

    # Resampling ("SCEPTRE-style") calibrated test for target knockdown
    resample_n: int = 2000
    resample_random_state: int = 0

    # E-distance is computed in PCA space on this many components
    edistance_n_pcs: int = 30
    # Similarity matrix
    # NOTE: the similarity method is deliberately not configurable. Spearman
    # of log2FC over the FIXED union DEG gene set is a methodological choice
    # from the collaborator's note, not a knob: correlating over each pair's
    # own union would make every value incomparable to every other. v1.2.5 and
    # earlier carried a `similarity_method` field that nothing read.
    jaccard_sqrt_colour_scale: bool = True  # upper triangle


# ===========================================================================
# Hashtags
# ===========================================================================
@dataclass
class HTOConfig:
    """Hashtag demultiplexing.

    The default normalisation is log1p followed by per-hashtag mean centring
    across cells. Per-feature is the right AXIS for ADT/HTO -- Seurat's
    ``margin = 2`` -- but the transform is not Seurat's CLR, and saying so was
    an overclaim carried from v1.1.0 through v1.2.5. See
    ``stats.clr_by_feature`` for the difference and ``stats.clr_true_seurat``
    for the Seurat formula; ``normalisation`` below selects between them.

    The CALLING structure is per-hashtag positivity plus, where a whitelist
    declares the design, matching against declared combinations. That is
    deliberate and is not what Seurat's ``HTODemux`` does: HTODemux assumes one
    tag per cell in its classification step, which makes it the wrong tool for
    combinatorial hashing. Independent per-tag thresholds handle both single
    and combinatorial designs.
    """

    # "mean_centred_log1p" (default, unchanged behaviour) | "seurat_clr" |
    # "compositional". See stats.HTO_NORMALISATIONS.
    normalisation: str = "mean_centred_log1p"
    # Emit a table comparing per-hashtag thresholds and call rates under every
    # available transform. Cheap -- it is a threshold fit per hashtag per
    # transform -- and it turns "which CLR is right?" into a number measured on
    # your own data rather than an argument from the literature.
    compare_normalisations: bool = True

    min_reads: int = 10
    threshold_mode: str = "background_quantile"   # or "fixed" / "otsu"
    fixed_threshold: float = 2.0
    positive_quantile: float = 0.99
    quantile_sweep: tuple[float, ...] = tuple(
        round(0.80 + 0.01 * i, 2) for i in range(20)
    )
    random_state: int = 0
    # ----------------------------------------------------------- separability
    # v1.1.0 judged separability on (signal mean - background mean) divided by
    # the UNWEIGHTED mean of the two cluster SDs, against a hard cut of 2.5.
    # On MDL-1856 that metric was inverted -- it flagged the one good hashtag
    # and cleared the two bad ones:
    #
    #   hash.C  sep_sd 2.587 -> FLAGGED, but valley_ratio 0.056 (deep trough)
    #   hash.F  sep_sd 2.838 -> passed,  but valley_ratio 1.000 (NO trough)
    #   hash.D  sep_sd 7.432 -> passed,  but 98.4% zero UMIs, bg_sd 0.005
    #
    # The reason is structural: a k-means mode gap measures how far apart the
    # modes are, which is not what bimodality means. What matters is whether
    # there is a TROUGH between them. So the primary statistic is now the
    # valley ratio -- kernel density at the chosen cut divided by the smaller
    # of the two modal densities -- and the standardised gap is retained as a
    # secondary, now with a properly size-weighted pooled SD.
    #
    # Verdict is graded, not boolean: "clean" / "shallow" / "unimodal" /
    # "degenerate". A single yes/no on a continuous quantity is what made
    # hash.C (2.587) and hash.F (2.838) look like different kinds of thing.
    valley_ratio_clean_max: float = 0.30      # <= this  -> genuine trough
    valley_ratio_shallow_max: float = 0.60    # <= this  -> shallow but present
    valley_bins: int = 200
    # Secondary check, on the size-weighted pooled SD.
    min_separation_sd: float = 2.5
    # Degenerate: the "background cluster" is a spike (a hashtag that captured
    # almost nothing), which yields an enormous, meaningless separation score.
    degenerate_frac_background_min: float = 0.95
    degenerate_bg_sd_min: float = 0.01
    # ...and the same failure seen from the other side: essentially no cells
    # above the threshold means there is no signal population at all.
    degenerate_frac_positive_min: float = 0.005

    # ------------------------------------------------------------ demux design
    # When the manifest supplies a hashtag whitelist, a cell's positive set is
    # matched against the declared sets and classified Resolved / Ambiguous /
    # Negative. Without one, fall back to the v1.1.0 singlet/multiplet rule.
    #
    # Combinatorial tagging is normal, not a failure: on MDL-1856 the v1.1.0
    # rule reported 39.4% singlets / 57.3% multiplets and graded it "poor",
    # when part of that is the intended design.
    # Grading thresholds for pct_hto_resolved (replaces pct_hto_singlet).
    resolved_good_min: float = 70.0
    resolved_warn_min: float = 45.0
    # How many unexpected positive sets to list in the report.
    unexpected_sets_top_n: int = 25
    # Regardless of the quantile rule, a threshold must sit at least this many
    # background SDs above the background mean. Without this floor, a hashtag
    # that did not work gets a threshold *below* its own median (the background
    # "cluster" is just the lower half of a unimodal blob), and most cells are
    # called positive for it -- which then makes most cells look like
    # multiplets.
    min_threshold_background_sd: float = 3.0


# ===========================================================================
# Clustering / embedding
# ===========================================================================
@dataclass
class EmbeddingConfig:
    """Normalisation, feature selection, embedding and clustering.

    Defaults follow the collaborator's pipeline, which is more careful than
    the original in three ways we adopt:

      1. MT/ribosomal genes are excluded from HVG selection *and* from the
         marker-gene test, so clusters and their markers are biologically
         interpretable rather than stress/translation artefacts.
      2. Sequencing depth, %mito and cell-cycle scores are regressed out.
      3. Tiny Leiden fragments are merged into their nearest neighbour in PCA
         space rather than dropped, so no cell is silently lost.
    """

    target_sum: float = 1e4          # 1e6 for CPM; 1e4 is the scanpy default
    n_top_genes: int = 5000
    exclude_mito_from_hvg: bool = True
    exclude_ribo_from_hvg: bool = True
    hvg_flavor: str = "seurat_v3"    # selected on raw counts, before log1p

    # Candidates for the "which column looks like a batch?" resolver. The
    # resolved value is REPORTED unconditionally but CONSUMED only by harmony,
    # which is off by default (see batch_correct below).
    batch_key_candidates: tuple[str, ...] = ("sample", "prefix", "batch")
    # A column where one level holds this fraction or more of the cells is not
    # batch structure, it is one big group with a rounding error attached.
    # v1.2.5 and earlier had a docstring on pick_batch_key promising exactly
    # this check and no code implementing it.
    batch_dominance_max: float = 0.90

    # Batch-aware HVG selection, SEPARATE from batch_correct and None by
    # default.
    #
    # Until v1.3.0 the resolved batch_key was passed straight to
    # highly_variable_genes, so gene selection was batch-aware whether or not
    # anyone asked. With seurat_v3 that ranks each gene's variance *within*
    # each batch and combines ranks, which is the right thing when batch is a
    # nuisance -- and precisely the wrong thing when "batch" is `sample` and
    # sample is the condition, because it then systematically de-prioritises
    # the genes that differ between conditions, i.e. the ones the experiment
    # exists to find.
    hvg_batch_key: str | None = None

    # Work on a copy of the input matrix.
    #
    # False since v1.3.0. This is the largest single allocation in the stage
    # and the pipeline never reads the input object again afterwards. Library
    # callers who pass an AnnData they still need should set it back to True:
    # with False, the stage modifies the object it was handed.
    copy_input: bool = False

    # Covariate regression, EMPTY by default since v1.3.0.
    #
    # Two reasons. Scope: this only ever touches the HVG block used for the
    # embedding, never X_log, so no DE result, log2FC, knockdown estimate or
    # resampling p-value depends on it -- it buys a tidier UMAP and costs the
    # longest step in the stage. And for cell cycle specifically it is actively
    # wrong here: many knockouts ARE proliferation phenotypes, so regressing
    # S_score/G2M_score suppresses exactly the perturbations you are screening
    # for. Cell-cycle scores are still COMPUTED and reported (score_cell_cycle
    # below) -- they are just not removed.
    #
    # `--regress-qc` opts back in to depth and %mito only. Cell-cycle
    # regression is deliberately not offered as an option.
    regress_out: tuple[str, ...] = ()
    # Conventional outlier clip on the z-scored HVG block, from Seurat's
    # ScaleData(scale.max = 10) via the scanpy PBMC3k tutorial. Prevents a
    # handful of extreme cells dominating a gene's contribution to PC1. Ten is
    # a round number, not a derived one; there is no method paper behind it.
    # Embedding only. Set to None to disable.
    scale_max_value: float | None = 10.0

    n_pcs: int = 30
    n_neighbors: int = 15
    leiden_resolution: float = 1.0
    # Merge Leiden clusters below this fraction of cells into their nearest
    # cluster by PCA centroid distance.
    #
    # 0.0 (no merging) since v1.3.0. The merge was invented to stop the report
    # filling with 12-cell specks, but there is no method behind it, Euclidean
    # centroid distance in 30-dim PCA space is a poor test of whether two
    # populations are the same thing, and the old 0.005 default meant ~935
    # cells in a 187k experiment -- which is a real population, and in a
    # screen quite possibly the interesting one. Small clusters are now
    # reported instead of absorbed.
    min_cluster_frac: float = 0.0
    # Clusters below this fraction are counted and reported (not merged), so
    # fragmentation is visible without being silently repaired.
    small_cluster_report_frac: float = 0.01

    # Batch integration on the embedding itself.
    #
    # "none" since v1.3.0, and "auto" is now an ALIAS for "none" so old config
    # files keep loading without silently reacquiring correction.
    #
    # History worth keeping: harmony entered in v1.1.0 to resolve a
    # docstring-vs-code mismatch (the draft described "batch-aware" clustering
    # while only passing batch_key to HVG selection). The mismatch was resolved
    # by making the claim true rather than by deleting the claim, and the
    # default was "auto", i.e. yes-whenever-possible. Two consequences:
    #
    #   1. `sample` is resolved as the batch key essentially always, with no
    #      check on whether sample is confounded with the condition. Where the
    #      conditions ARE separate samples, harmony integrates away the
    #      comparison the experiment is running.
    #   2. EmbeddingResult.pca is obsm["X_pca_harmony"] when harmony runs, and
    #      that array is passed to the perturbation stage and drives
    #      edistance_table. So E-distance -- a quantitative effect size, not a
    #      visualisation -- was computed in corrected space.
    #
    # Requires an explicit `--batch-correct harmony`, and is refused when the
    # resolved key overlaps a declared condition column.
    batch_correct: str = "none"      # "none" (= "auto") | "harmony"

    score_cell_cycle: bool = True
    # OFF by default since v1.2.2.
    #
    # Three reasons, in order of weight:
    #   1. Scrublet assumes a heterogeneous population -- it detects doublets
    #      as cells that look like mixtures of transcriptionally distinct
    #      types. In a homogeneous Perturb-seq line there are no distinct types
    #      to mix, so its scores are not informative here.
    #   2. With batch_key it runs per batch, simulating doublets and building a
    #      kNN graph for each. On 187k cells across 8 batches that exhausted
    #      memory and killed the run.
    #   3. It was effectively untested: any input carrying obsm['X_pca'] takes
    #      the reuse path, which never calls it. It first executed on the first
    #      object without a precomputed embedding.
    #
    # The code stays, and `--doublets` turns it back on. Doublets are never
    # dropped either way -- the panel only ever reported a rate.
    #
    # v1.3.0 fixes a gap that made this flag half-ineffective: the scanpy path
    # ran the built-in synthetic-doublet fallback under
    # `if "predicted_doublet" not in obs.columns`, a condition that is true
    # PRECISELY when detection was switched off. So turning detection off
    # swapped scrublet for the fallback rather than skipping the step, and
    # unlike the scrublet branch it emitted no note saying so. The fallback is
    # also brute-force all-pairs and quadratic: ~6 s at 20k cells, ~26 s at
    # 40k, extrapolating to 8-11 minutes at 187k. Off now means off.
    #
    # If it is ever turned back on, note that the fallback's threshold is
    # max(quantile(scores, 0.90), expected * 1.5), which pins the reported rate
    # near 5-10% regardless of the data. Prefer scrublet.
    detect_doublets: bool = False

    n_marker_genes: int = 10
    # Markers use a Mann-Whitney one-vs-rest test ranked by effect size. Not
    # configurable, and the `marker_method` field that implied otherwise (and
    # that nothing read) was removed in v1.3.0.

    random_state: int = 0


# ===========================================================================
# Figures / report
# ===========================================================================
@dataclass
class FigureConfig:
    dpi: int = 200
    format: str = "png"
    palette: tuple[str, ...] = (
        "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
        "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    )
    continuous_cmap: str = "viridis"
    diverging_cmap: str = "RdBu_r"
    density_cmap: str = "mako_r"
    font_size: int = 9


@dataclass
class ReportConfig:
    title: str | None = None            # defaults to experiment id
    subtitle: str = "Perturb-seq QC & analysis report"
    embed_figures: bool = True          # base64 -> single self-contained file
    # A fully self-contained report must not reach out to the network.  The
    # previous report claimed to be self-contained while loading Google
    # Fonts on every open; we ship a system font stack instead.
    show_missing_placeholders: bool = True
    include_data_tables: bool = True
    max_table_rows: int = 50


# ===========================================================================
# Top-level
# ===========================================================================
@dataclass
class PipelineConfig:
    """The single object threaded through every stage."""

    manifest_path: Path | None = None
    output_path: Path | None = None
    h5ad_path: Path | None = None

    # Randomly subsample to at most this many cells straight after loading.
    # A pragmatic escape hatch: it lets a very large experiment produce a
    # complete, trustworthy-in-structure report on a machine that cannot hold
    # the full matrix. The report states the subsample size prominently, since
    # per-perturbation cell counts scale down with it.
    subsample_cells: int | None = None
    subsample_random_state: int = 0

    # Read counts from this layer instead of X. Use when X holds normalised or
    # log-transformed values and the raw counts live in e.g. layers['counts'].
    counts_layer: str | None = None

    # Which manifest columns describe experimental conditions worth
    # comparing.  Empty = autodetect (any column with 2..max_compare_levels
    # distinct non-blank values).  The old report hardcoded this list in one
    # file while its own docstring promised nothing was hardcoded.
    condition_columns: tuple[str, ...] = ()
    max_compare_levels: int = 12
    # How many comparison axes to carry. Each one multiplies the figure count,
    # but a 4-factor design (fixation x buffer x acoh x gRNA method) needs 4 --
    # and silently dropping one is worse than a longer report.
    max_condition_axes: int = 4

    qc: QCThresholds = field(default_factory=QCThresholds)
    modality: ModalityConfig = field(default_factory=ModalityConfig)
    guide: GuideConfig = field(default_factory=GuideConfig)
    perturb: PerturbConfig = field(default_factory=PerturbConfig)
    hto: HTOConfig = field(default_factory=HTOConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    figures: FigureConfig = field(default_factory=FigureConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    # Stage control
    stages: tuple[str, ...] = ()        # empty = all applicable

    # --- run mode -----------------------------------------------------------
    # The pipeline is EXPLORE-FIRST. A run with no thresholds available stops
    # after the QC stage so a human can look at the distributions before
    # anything is filtered, clustered or quantified.
    #
    #   explore_only     force the stop even if thresholds are available
    #   auto_thresholds  skip the stop and run end-to-end on derived values
    #
    # Neither is normally set by hand: `decide_run_mode` resolves the mode from
    # what the manifest and CLI actually supply. See that function for the
    # precedence table.
    explore_only: bool = False
    auto_thresholds: bool = False

    # Refuse to run on an expression matrix that fails plausibility checks.
    # See sanity.py: an h5ad with var_names permuted relative to X's columns
    # produced two complete, confident, wrong reports before anyone thought to
    # ask whether ACTB was detected.
    check_input_matrix: bool = True
    housekeeping_min_detection: float = 0.30

    use_checkpoints: bool = True
    force_recompute: bool = False
    random_state: int = 0
    # -1 means "all cores". This is assigned to sc.settings.n_jobs from
    # v1.3.0; before that the field existed and nothing read it, so scanpy's
    # regress_out and neighbors ran single-threaded regardless of what was set
    # here.
    n_jobs: int = -1
    verbose: bool = True

    # ---------------------------------------------------------------- paths
    @property
    def analysis_dir(self) -> Path:
        assert self.output_path is not None, "output_path not resolved yet"
        return Path(self.output_path) / "analysis_outputs"

    @property
    def fig_dir(self) -> Path:
        return self.analysis_dir / "figures"

    @property
    def table_dir(self) -> Path:
        return self.analysis_dir / "tables"

    @property
    def checkpoint_dir(self) -> Path:
        return self.analysis_dir / "checkpoints"

    @property
    def report_path(self) -> Path:
        """Where the report goes.

        Explore and full runs write to DIFFERENT files, so a full run cannot
        overwrite the QC panels you are still working from -- and so that
        finding ``qc_report.html`` on disk always means a complete analysis.
        """
        name = "qc_explore.html" if self.explore_only else "qc_report.html"
        return self.analysis_dir / name

    @property
    def threshold_state_path(self) -> Path:
        """Records the auto-derived thresholds from the most recent explore run.

        Used on the next full run to tell an edited threshold apart from an
        untouched auto-derived one, so the report can say which is which.
        """
        return self.analysis_dir / "threshold_state.json"

    def ensure_dirs(self) -> None:
        for d in (self.analysis_dir, self.fig_dir, self.table_dir, self.checkpoint_dir):
            d.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- serial
    def to_dict(self) -> dict[str, Any]:
        def _clean(o: Any) -> Any:
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_clean(v) for v in o]
            return o

        return _clean(asdict(self))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    # ----------------------------------------------------------- overriding
    def with_overrides(self, **kw: Any) -> "PipelineConfig":
        return replace(self, **kw)


_SUBCONFIGS = {
    "qc": QCThresholds,
    "modality": ModalityConfig,
    "guide": GuideConfig,
    "perturb": PerturbConfig,
    "hto": HTOConfig,
    "embedding": EmbeddingConfig,
    "figures": FigureConfig,
    "report": ReportConfig,
}


def _coerce(target_type: Any, value: Any) -> Any:
    """Best-effort coercion of a JSON/YAML scalar into a dataclass field."""
    if value is None:
        return None
    origin = getattr(target_type, "__origin__", None)
    if origin in (tuple, list) or target_type in (tuple, list):
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return (value,)
    if target_type in (int, float, str, bool):
        try:
            return target_type(value)
        except (TypeError, ValueError):
            return value
    return value


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Read a JSON or YAML config file into a nested dict."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                f"{p} is YAML but PyYAML is not installed; use JSON or `pip install pyyaml`"
            ) from e
        return yaml.safe_load(text) or {}
    return json.loads(text)


def build_config(
    overrides: dict[str, Any] | None = None,
    config_file: str | Path | None = None,
) -> PipelineConfig:
    """Construct a PipelineConfig from defaults + optional file + overrides.

    ``overrides`` may be flat (``{"min_genes": 800}``) or nested
    (``{"qc": {"min_genes": 800}}``).  Flat keys are routed to whichever
    subconfig declares them, so callers (notably the CLI) don't have to know
    the nesting.
    """
    data: dict[str, Any] = {}
    if config_file:
        data.update(load_config_file(config_file))
    if overrides:
        data.update(overrides)

    # Split into nested subconfig payloads and top-level keys
    nested: dict[str, dict[str, Any]] = {k: {} for k in _SUBCONFIGS}
    top: dict[str, Any] = {}

    field_owner: dict[str, str] = {}
    for name, cls in _SUBCONFIGS.items():
        for f in fields(cls):
            field_owner.setdefault(f.name, name)

    top_names = {f.name for f in fields(PipelineConfig)}

    for key, value in data.items():
        if key in _SUBCONFIGS and isinstance(value, dict):
            nested[key].update(value)
        elif key in top_names and key not in _SUBCONFIGS:
            top[key] = value
        elif key in field_owner:
            nested[field_owner[key]][key] = value
        else:
            raise ValueError(f"Unknown configuration key: {key!r}")

    kwargs: dict[str, Any] = {}
    for name, cls in _SUBCONFIGS.items():
        payload = nested[name]
        if not payload:
            kwargs[name] = cls()
            continue
        valid = {f.name: f.type for f in fields(cls)}
        clean = {}
        for k, v in payload.items():
            if k not in valid:
                raise ValueError(f"Unknown key {k!r} in config section {name!r}")
            clean[k] = _coerce(valid[k], v)
        kwargs[name] = cls(**clean)

    for k, v in top.items():
        if k in ("manifest_path", "output_path", "h5ad_path") and v is not None:
            v = Path(v)
        if k in ("condition_columns", "stages") and v is not None:
            v = tuple(v)
        kwargs[k] = v

    return PipelineConfig(**kwargs)


# The five threshold names, in one place.  The original spelled this list out
# four separate times across two files, which is exactly how they drifted.
THRESHOLD_KEYS: Sequence[str] = (
    "min_genes", "max_genes", "min_counts", "max_counts", "max_mito",
)


@dataclass
class RunMode:
    """The resolved decision about whether to explore or run end to end."""

    explore: bool
    reason: str                      # shown to the user, so it must be plain
    thresholds_from: str             # "manifest" | "cli" | "auto" | "none yet"
    missing: tuple[str, ...] = ()    # threshold columns still blank


def decide_run_mode(
    manifest_thresholds: dict[str, float | None],
    cli_thresholds: dict[str, float | None],
    explore_flag: bool = False,
    auto_flag: bool = False,
) -> RunMode:
    """Decide whether this invocation explores or runs the full pipeline.

    The pipeline is explore-first: the default for a manifest that carries no
    thresholds is to stop after QC, so nothing is filtered or quantified on
    numbers no human has seen.

    Precedence, first match wins:

    ==============================================  ===============
    condition                                       mode
    ==============================================  ===============
    ``--explore``                                   explore
    ``--auto-thresholds``                           full (auto)
    any threshold given on the command line         full (cli)
    all five present in the manifest                full (manifest)
    *some* present in the manifest                  explore
    none anywhere                                   explore
    ==============================================  ===============

    The "some but not all" case deliberately explores rather than running.  A
    half-filled threshold block is almost always an unfinished edit, and the
    previous pipeline's behaviour here was a real bug: passing a single
    threshold flag made it run the full analysis with the other four silently
    falling back to hardcoded defaults.
    """
    cli_set = {k: v for k, v in cli_thresholds.items() if v is not None}
    man_set = {k: v for k, v in manifest_thresholds.items() if v is not None}
    missing = tuple(k for k in THRESHOLD_KEYS if k not in man_set and k not in cli_set)

    if explore_flag:
        return RunMode(
            explore=True,
            reason="--explore was requested",
            thresholds_from="manifest" if man_set else "none yet",
            missing=missing,
        )

    if auto_flag:
        return RunMode(
            explore=False,
            reason=(
                "--auto-thresholds was passed, so the QC review step was skipped "
                "and thresholds were derived from the data"
            ),
            thresholds_from="auto",
            missing=missing,
        )

    if cli_set:
        return RunMode(
            explore=False,
            reason=(
                f"{len(cli_set)} threshold(s) given on the command line "
                f"({', '.join(sorted(cli_set))})"
                + (
                    f"; the remaining {len(missing)} will be derived from the data"
                    if missing else ""
                )
            ),
            thresholds_from="cli",
            missing=missing,
        )

    if not missing:
        return RunMode(
            explore=False,
            reason="all five thresholds are set in the manifest",
            thresholds_from="manifest",
            missing=(),
        )

    if man_set:
        return RunMode(
            explore=True,
            reason=(
                f"the manifest has {len(man_set)} of 5 thresholds set; "
                f"still blank: {', '.join(missing)}"
            ),
            thresholds_from="manifest",
            missing=missing,
        )

    return RunMode(
        explore=True,
        reason="no thresholds are set yet in the manifest or on the command line",
        thresholds_from="none yet",
        missing=missing,
    )
