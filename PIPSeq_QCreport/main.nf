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

    RUN_QC_REPORT(manifest_inputs)

    RUN_QC_REPORT.out.report
        .ifEmpty { log.info "No manifest reached the report stage in this run (all explore-only, or all failed)." }

    RUN_QC_REPORT.out.analysis_outputs
        .subscribe { id, dir -> log.info "[${id}] analysis_outputs -> ${params.outdir}/${id}/analysis_outputs" }
}
