// Wraps the perturbseq_report CLI (perturbseq-report / run_perturbseq_report.py).
//
// The tool itself decides explore-vs-full: a manifest whose five threshold
// columns are all blank stops after QC ("explore"); once they are filled it
// runs end to end ("full"). `--explore` / `--auto-thresholds` (params.mode)
// override that decision the same way they do on the command line. See
// perturbseq_report/config.py::decide_run_mode for the exact precedence.
//
// IMPORTANT SIDE EFFECT: on a run that reaches the QC stage, the tool writes
// the thresholds it used back into the manifest CSV in place (with a
// timestamped .bak backup made first -- see manifest.py::Manifest.write_thresholds).
// Because the whole manifest DIRECTORY is staged as a Nextflow `path` input
// (so relative grna_whitelist/hashtag_whitelist columns resolve correctly),
// that write happens against the staged copy, which for the default local/HPC
// executors is a symlink back to your original file -- so your source manifest
// on disk really is modified. This is intentional upstream behaviour (the
// pipeline is explore-first: run once, review qc_explore.html, edit thresholds
// by hand or trust the auto-derived ones, run again) and is also exactly what
// makes `-resume` work correctly here: Nextflow hashes the staged directory's
// contents, so editing the manifest's thresholds invalidates the cached task.

process RUN_QC_REPORT {
    tag "$id"
    label 'process_high_memory'
    publishDir "${params.outdir}/${id}", mode: params.publish_mode, overwrite: true

    input:
    tuple val(id), path(manifest_dir), val(manifest_name)

    output:
    tuple val(id), path("analysis_outputs"), emit: analysis_outputs
    path "analysis_outputs/qc_*.html", optional: true, emit: report
    path "analysis_outputs/artifacts.json", optional: true, emit: artifacts
    path "run.log", emit: log
    path "${manifest_name}", emit: manifest_used
    path "${manifest_name}.bak-*", optional: true, emit: manifest_backup

    script:
    def flags = []

    if (params.mode == 'explore')      flags << '--explore'
    else if (params.mode == 'auto')    flags << '--auto-thresholds'
    // 'default' (the CLI's own explore-first decision): no flag added.

    if (params.min_genes   != null) flags << "--min-genes ${params.min_genes}"
    if (params.max_genes   != null) flags << "--max-genes ${params.max_genes}"
    if (params.min_counts  != null) flags << "--min-counts ${params.min_counts}"
    if (params.max_counts  != null) flags << "--max-counts ${params.max_counts}"
    if (params.max_mito    != null) flags << "--max-mito ${params.max_mito}"

    if (params.config)          flags << "--config ${params.config}"
    if (params.counts_layer)    flags << "--counts-layer ${params.counts_layer}"
    if (params.subsample_cells) flags << "--subsample-cells ${params.subsample_cells}"

    if (params.conditions)      flags << "--conditions ${params.conditions instanceof List ? params.conditions.join(' ') : params.conditions}"
    if (params.doublets)        flags << '--doublets'
    if (params.no_cell_cycle)   flags << '--no-cell-cycle'
    if (params.regress_qc)      flags << '--regress-qc'
    if (params.skip_input_check) flags << '--skip-input-check'
    if (params.force_recompute) flags << '--force-recompute'
    if (params.cluster_col)     flags << "--cluster-col ${params.cluster_col}"
    if (params.low_memory)      flags << '--low-memory'
    if (params.batch_correct)   flags << "--batch-correct ${params.batch_correct}"
    if (params.hvg_batch_key)   flags << "--hvg-batch-key ${params.hvg_batch_key}"
    if (params.resolution  != null) flags << "--resolution ${params.resolution}"
    if (params.n_top_genes != null) flags << "--n-top-genes ${params.n_top_genes}"

    if (params.guide_min_reads  != null) flags << "--guide-min-reads ${params.guide_min_reads}"
    if (params.guide_purity_min != null) flags << "--guide-purity-min ${params.guide_purity_min}"
    if (params.ntc_label)                flags << "--ntc-label ${params.ntc_label}"
    if (params.dual_guide == true)       flags << '--dual-guide'
    if (params.dual_guide == false)      flags << '--single-guide'

    if (params.hto_threshold_mode) flags << "--hto-threshold-mode ${params.hto_threshold_mode}"
    if (params.hto_quantile != null) flags << "--hto-quantile ${params.hto_quantile}"

    // Free-text field: single-quoted, with embedded single quotes escaped
    // (close quote, escaped quote, reopen quote) so an apostrophe in the
    // title can't break the generated command line.
    if (params.title)           flags << "--title '${params.title.toString().replace("'", "'\\''")}'"
    if (params.link_figures)    flags << '--link-figures'
    if (params.no_tables)       flags << '--no-tables'
    if (params.max_de_targets != null) flags << "--max-de-targets ${params.max_de_targets}"

    if (params.extra_args)      flags << params.extra_args

    def flag_str = flags.join(' \\\n        ')

    """
    set -euo pipefail
    python3 "${params.pipeline_dir}/run_perturbseq_report.py" \\
        --manifest "${manifest_dir}/${manifest_name}" \\
        --output-path . \\
        ${flag_str} \\
        2>&1 | tee run.log

    cp "${manifest_dir}/${manifest_name}" .
    cp ${manifest_dir}/${manifest_name}.bak-* . 2>/dev/null || true
    """
}
