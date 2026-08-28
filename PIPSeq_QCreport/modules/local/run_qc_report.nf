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

// Some launchers (ICA's inputForm.json among them) have no way to submit a
// truly absent value for a text/number field -- an untouched box comes
// through as an empty string, not Nextflow `null`. A bare `!= null` check
// (fine for CLI/-params-file use, where an omitted key really is null) would
// then treat "" as a real override and hand argparse an empty --flag value.
// isSet() treats blank-or-null uniformly as "not set". File-scoped (not
// defined inside `script:`) so the `container` directive below can use it too.
def isSet(v) { v != null && v.toString().trim() != '' }

process RUN_QC_REPORT {
    tag "$id"
    label 'process_high_memory'
    publishDir "${params.outdir}/${id}", mode: params.publish_mode, overwrite: true
    // Read directly from a plain parameter (defaulted in nextflow.config to
    // a public GCP image), not a profile -- ICA does not reliably apply
    // Nextflow config profiles, so a container wired up only under
    // `-profile docker` silently never activates there and the process runs
    // on the bare scheduler node instead (which is how we found this: it
    // failed with "python3: command not found"). This is the pattern this
    // lab's other ICA-deployed Nextflow pipelines already use (see
    // SingleCell/PIPseqDownsample's `params.qc_container`).
    container { isSet(params.qc_container) ? params.qc_container : null }

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

    if (isSet(params.min_genes))  flags << "--min-genes ${params.min_genes}"
    if (isSet(params.max_genes))  flags << "--max-genes ${params.max_genes}"
    if (isSet(params.min_counts)) flags << "--min-counts ${params.min_counts}"
    if (isSet(params.max_counts)) flags << "--max-counts ${params.max_counts}"
    if (isSet(params.max_mito))   flags << "--max-mito ${params.max_mito}"

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
    if (isSet(params.resolution))  flags << "--resolution ${params.resolution}"
    if (isSet(params.n_top_genes)) flags << "--n-top-genes ${params.n_top_genes}"

    if (isSet(params.guide_min_reads))  flags << "--guide-min-reads ${params.guide_min_reads}"
    if (isSet(params.guide_purity_min)) flags << "--guide-purity-min ${params.guide_purity_min}"
    if (params.ntc_label)                flags << "--ntc-label ${params.ntc_label}"
    // 'auto' (default): no flag, let the tool autodetect from guide-ID structure.
    // A plain true/false param can't survive a schema-driven checkbox, which
    // always submits false when left untouched -- silently forcing single-guide
    // on every UI-launched run instead of autodetecting. The three-way string
    // is what makes "untouched" distinguishable from "explicitly single".
    if (params.dual_guide == 'dual')     flags << '--dual-guide'
    if (params.dual_guide == 'single')   flags << '--single-guide'

    if (params.hto_threshold_mode) flags << "--hto-threshold-mode ${params.hto_threshold_mode}"
    if (isSet(params.hto_quantile)) flags << "--hto-quantile ${params.hto_quantile}"

    // Free-text field: single-quoted, with embedded single quotes escaped
    // (close quote, escaped quote, reopen quote) so an apostrophe in the
    // title can't break the generated command line.
    if (params.title)           flags << "--title '${params.title.toString().replace("'", "'\\''")}'"
    if (params.link_figures)    flags << '--link-figures'
    if (params.no_tables)       flags << '--no-tables'
    if (isSet(params.max_de_targets)) flags << "--max-de-targets ${params.max_de_targets}"

    if (params.extra_args)      flags << params.extra_args

    def flag_str = flags.join(' \\\n        ')

    // Same blank-vs-null issue as isSet() above, but higher stakes: if this
    // ever arrives blank the run doesn't misbehave on one option, it fails
    // outright (the script path resolves to nothing). Falling back here makes
    // it structurally impossible for a launcher to break this by submitting
    // an empty value for a field it thinks is just unset.
    //
    // The fallback itself now tracks qc_container rather than being a fixed
    // path: containers/Dockerfile unpacks the vendored package at
    // /opt/perturbseq_report_v1.3.5, so whenever a container is actually in
    // play that's where it lives regardless of this checkout's own location;
    // only a genuinely container-free run (conda/bare metal) should resolve
    // it relative to projectDir.
    def pipelineDir = isSet(params.pipeline_dir)
        ? params.pipeline_dir
        : (isSet(params.qc_container)
            ? '/opt/perturbseq_report_v1.3.5'
            : "${projectDir}/perturbseq_report_v1.3.5")

    """
    set -euo pipefail
    python3 "${pipelineDir}/run_perturbseq_report.py" \\
        --manifest "${manifest_dir}/${manifest_name}" \\
        --output-path . \\
        ${flag_str} \\
        2>&1 | tee run.log

    cp "${manifest_dir}/${manifest_name}" .
    cp ${manifest_dir}/${manifest_name}.bak-* . 2>/dev/null || true
    """
}
