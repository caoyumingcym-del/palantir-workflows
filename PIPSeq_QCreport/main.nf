#!/usr/bin/env nextflow
/*
 * PIPSeq_QCreport -- Nextflow wrapper around perturbseq_report
 * (a Perturb-seq QC and analysis pipeline that produces a single
 * self-contained HTML report from a sample manifest + .h5ad).
 *
 * The underlying tool does all the analysis; this workflow's job is just to
 * turn "one or many sample manifests" into "one parallel Nextflow task per
 * manifest, with resumability and proper output collection" -- replacing the
 * bundled run_batch.py, which does the same fan-out serially in a for-loop.
 *
 * Usage
 * -----
 *   nextflow run . --manifest sample_manifest.csv
 *   nextflow run . --manifest 'manifests/*.csv' --mode auto
 *   nextflow run . --manifest sample_manifest.csv -profile docker
 *   nextflow run . -profile test        # tiny synthetic dataset, no real data needed
 *
 * See README.md for the full parameter list and for the explore-first
 * workflow this tool expects (run once, review qc_explore.html, edit the
 * manifest's threshold columns if you disagree, run the same command again).
 */
nextflow.enable.dsl = 2

include { RUN_QC_REPORT } from './modules/local/run_qc_report.nf'
include { BUILD_SLIDES  } from './modules/local/build_slides.nf'

def helpMessage() {
    log.info """
    PIPSeq_QCreport -- Perturb-seq QC & analysis report pipeline
    ==============================================================

    Usage:
      nextflow run . --manifest <path-or-glob> [options]

    Required:
      --manifest PATH        Sample manifest CSV/TSV, or a glob matching several
                              (one Nextflow task is launched per manifest file).

    Run mode:
      --mode default|explore|auto   (default: 'default')
                              default : let the tool decide (explore-first --
                                        stops after QC if thresholds aren't
                                        all set in the manifest yet)
                              explore : force the QC-only stop, even if
                                        thresholds are already set
                              auto    : skip the review step; run end-to-end
                                        on thresholds derived from the data

    Common overrides (all optional; omit to derive from the data / manifest):
      --h5ad PATH             override the manifest's h5ad_path column -- needed
                              on platforms (ICA) that don't mount arbitrary
                              manifest-referenced paths into the task container
      --dragen_root DIR...    one or more directories holding runs' DRAGEN
                              output subfolders (each named like the
                              manifest's own dragen_path values); every root
                              is tried for every run -- same reason as
                              --h5ad; without it, "Sequencing QC" is skipped,
                              non-fatally
      --min_genes, --max_genes, --min_counts, --max_counts, --max_mito
      --config PATH           JSON/YAML file of perturbseq_report config overrides
      --counts_layer NAME     read raw counts from adata.layers[NAME] instead of X
      --subsample_cells N     randomly subsample to at most N cells
      --conditions 'a b c'    manifest columns to compare (default: autodetect)
      --doublets              run scrublet doublet annotation (off by default)
      --batch_correct none|harmony
      --resolution FLOAT      Leiden resolution (default 1.0)

      See README.md for the complete list (guide calling, hashtags, report
      formatting) -- every perturbseq-report CLI flag has a matching --param,
      and --extra_args passes anything else straight through verbatim.

    Output:
      --outdir DIR            where reports/figures/tables are published
                              (default: 'results')
      --build_slides          build a Google-Slides-ready .pptx from
                              artifacts.json on every run (default: true;
                              --build_slides false to skip)

    Execution:
      -profile docker|conda|test   see README.md / nextflow.config
    """.stripIndent()
}

if (params.help) {
    helpMessage()
    exit 0
}

if (!params.manifest) {
    log.error "Missing required parameter --manifest (a sample manifest CSV/TSV, or a glob matching several)."
    helpMessage()
    exit 1
}

workflow {

    manifest_files = Channel.fromPath(params.manifest, checkIfExists: true)

    // The whole manifest DIRECTORY is staged (not just the file) so that
    // relative `grna_whitelist` / `hashtag_whitelist` columns -- resolved by
    // perturbseq_report relative to the manifest's own directory -- still
    // point at real files once staged into the task work dir. Staging as a
    // `path` (rather than passing a bare string) also gives correct `-resume`
    // caching: the task re-runs if the manifest's contents change, e.g. after
    // editing the threshold columns following an explore run.
    manifest_inputs = manifest_files
        .map { m -> tuple(m.baseName, m.getParent(), m.getName()) }
        .toList()
        .flatMap { list ->
            def ids = list.collect { it[0] }
            def dupes = ids.findAll { id -> ids.count(id) > 1 }.unique()
            if (dupes) {
                log.warn "Manifest basenames are not unique across the batch: ${dupes}. " +
                          "Their published output directories under --outdir will collide."
            }
            list
        }

    // Optional: stage a single .h5ad as an explicit pipeline input and pass
    // it via --h5ad, instead of trusting the manifest's own h5ad_path column.
    // Needed on ICA specifically -- ICA does not mount a project's data tree
    // into the task container, so a path typed into the manifest (relative
    // OR absolute) resolves to nothing once the container starts; only
    // files declared as actual pipeline inputs (like `manifest` itself) get
    // staged. `[]` is the standard Nextflow idiom for "no file" on a `path`
    // input; the module gates on isSet(params.h5ad) rather than inspecting
    // the staged value itself, so this never needs to be referenced when
    // absent.
    //
    // Uniform across every manifest in this run/batch: fine for the normal
    // case of one manifest per launch (the only shape ICA's single-file
    // picker supports anyway), wrong if you combine --h5ad with a
    // glob-based multi-manifest batch of genuinely different experiments.
    h5ad_ch = params.h5ad
        ? Channel.fromPath(params.h5ad, checkIfExists: true)
        : Channel.value([])

    // Same story as h5ad_ch, for the manifest's (per-row) dragen_path
    // columns instead of its (manifest-wide) h5ad_path: stage every
    // directory given here and the module passes them all as
    // --dragen-root, which tries each in turn against
    // <root>/<basename of that run's own dragen_path value> and uses
    // whichever exists. Without it, the "Sequencing QC" report section is
    // silently skipped on ICA -- non-fatal (transcriptome/guide/hashtag/
    // perturbation analysis are unaffected either way), but avoidable.
    //
    // One or several: several runs' DRAGEN output does not always share one
    // common parent directory that could be selected/staged as a single
    // unit, so ICA's dragen_root field allows selecting more than one
    // (maxValues > 1 in inputForm.json) -- params.dragen_root may then be a
    // real List rather than a single value, handled either way here. All
    // staged as one list-valued `path` input (rather than, say,
    // `.combine()`-ing a channel with one element per directory, which
    // would multiply out into extra tasks instead of giving one task
    // several directories to search).
    def dragen_roots = params.dragen_root
        ? (params.dragen_root instanceof List ? params.dragen_root : [params.dragen_root])
            .findAll { it }
            .collect { file(it, checkIfExists: true) }
        : []
    dragen_root_ch = Channel.value(dragen_roots)

    RUN_QC_REPORT(manifest_inputs.combine(h5ad_ch).combine(dragen_root_ch))

    RUN_QC_REPORT.out.report
        .ifEmpty { log.info "No manifest reached the report stage in this run (all explore-only, or all failed)." }

    RUN_QC_REPORT.out.analysis_outputs
        .subscribe { id, dir -> log.info "[${id}] analysis_outputs -> ${params.outdir}/${id}/analysis_outputs" }

    // On by default: every run that reaches artifacts.json (both explore and
    // full runs do) gets a .pptx slide deck built from it, the same registry
    // the HTML report itself renders from. --build_slides false skips this
    // (still leaves artifacts.json in place to build one from manually later
    // -- see build_slides.py's own docstring).
    if (params.build_slides) {
        // build_slides.py resolves every figure/table path in artifacts.json
        // relative to that file's OWN directory (see registry.py's Registry
        // paths and build_slides.py's `root = path.parent`) -- so the whole
        // analysis_outputs directory has to be staged into BUILD_SLIDES, not
        // just artifacts.json alone (that shipped a deck with every image
        // reference pointing at a file that was never staged into the task).
        qc_ready = RUN_QC_REPORT.out.analysis_outputs
            .filter { id, dir -> file("${dir}/artifacts.json").exists() }
        BUILD_SLIDES(qc_ready)
    }
}
