// Wraps build_slides.py -- a standalone script in the vendored package
// (perturbseq_report_v1.3.5/build_slides.py, NOT called by
// run_perturbseq_report.py itself) that turns one run's artifacts.json into
// a Google-Slides-ready .pptx deck: title -> contents -> per-section slides
// with figures/tables/metrics, all inherited from the same registry the
// HTML report renders from. See that script's own module docstring for the
// full design rationale.
//
// Same isSet()/pipelineDir pattern as run_qc_report.nf -- duplicated rather
// than shared, since Nextflow module files don't share plain `def`s across
// files as freely as processes, and it's a handful of lines either way.
def isSet(v) { v != null && v.toString().trim() != '' }

process BUILD_SLIDES {
    tag "$id"
    label 'process_low'
    publishDir "${params.outdir}/${id}", mode: params.publish_mode, overwrite: true
    container { isSet(params.qc_container) ? params.qc_container : null }

    input:
    // The whole analysis_outputs directory, not just artifacts.json --
    // build_slides.py resolves every figure/table path in it relative to
    // artifacts.json's own directory, so figures/ and tables/ have to be
    // staged alongside it.
    tuple val(id), path(analysis_outputs_dir)

    output:
    tuple val(id), path("analysis_outputs/qc_deck.pptx"), emit: deck
    path "build_slides.log", emit: log

    script:
    def flags = []
    // Same free-text quoting as run_qc_report.nf's --title, and the same
    // title as the HTML report -- one experiment, one title, for both.
    if (isSet(params.title)) {
        flags << "--title '${params.title.toString().replace("'", "'\\''")}'"
    }
    if (params.slides_extra_args) flags << params.slides_extra_args
    def flag_str = flags.join(' \\\n        ')

    def pipelineDir = isSet(params.pipeline_dir)
        ? params.pipeline_dir
        : (isSet(params.qc_container)
            ? '/opt/perturbseq_report_v1.3.5'
            : "${projectDir}/perturbseq_report_v1.3.5")

    """
    set -euo pipefail
    mkdir -p analysis_outputs
    python3 "${pipelineDir}/build_slides.py" "${analysis_outputs_dir}/artifacts.json" \\
        --out analysis_outputs/qc_deck.pptx \\
        ${flag_str} \\
        2>&1 | tee build_slides.log
    """
}
