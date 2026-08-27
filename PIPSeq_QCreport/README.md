# PIPSeq_QCreport

A [Nextflow](https://www.nextflow.io/) wrapper around
[`perturbseq_report`](perturbseq_report_v1.3.5/), a Perturb-seq QC and
analysis pipeline that turns a sample manifest + `.h5ad` into a single
self-contained HTML report (QC gating, guide calling, hashtag demultiplexing,
clustering, and per-perturbation differential expression / E-distance).

The Python tool does all of the actual analysis and is vendored here
unmodified (see [`perturbseq_report_v1.3.5/`](perturbseq_report_v1.3.5/) for
its own docs and changelog). This directory only adds the Nextflow layer:
running one task per sample manifest in parallel, with proper resumability,
container/conda environments, and output publishing -- replacing the
bundled `run_batch.py`, which does the same fan-out with a serial `for` loop
and `subprocess.run`.

## Before you run anything: this pipeline is explore-first

`perturbseq_report` is deliberately **not** a fire-and-forget pipeline by
default. A manifest with no QC thresholds set stops after the QC stage,
writes `qc_explore.html`, and fills the manifest's five threshold columns
(`min_genes`, `max_genes`, `min_counts`, `max_counts`, `max_mito`) with
values derived from the data. The intended workflow is:

1. Run once. Look at `qc_explore.html`.
2. Edit the five threshold columns in the manifest if you disagree with the
   derived values (or leave them as-is).
3. Run the *same* command again. Because the manifest now carries all five
   thresholds, this pass runs the full pipeline and writes `qc_report.html`.

This pipeline exposes that as `--mode`:

| `--mode`   | behaviour                                                              |
|------------|-------------------------------------------------------------------------|
| `default`  | let the tool decide from the manifest's current state (the above)       |
| `explore`  | always stop after QC, even if thresholds are already set                |
| `auto`     | skip the review step; run end-to-end on thresholds derived from the data|

**Side effect to know about:** whenever a run reaches the QC stage (which is
every run), the tool writes the thresholds it used back into the manifest CSV
in place, after first saving a timestamped `.bak` copy next to it. That is
what step 3 above depends on, and it is also what makes `-resume` work
correctly here -- edit the manifest and Nextflow reruns the task; run it
again unchanged and Nextflow reuses the cached result. Point `--manifest` at
a manifest CSV that lives somewhere writable that you're happy to have
Nextflow edit, not a read-only or shared reference copy.

## Quick start

```bash
# 1. Generate a tiny synthetic dataset (no real data needed) and smoke-test
#    the plumbing:
python3 bin/make_test_data.py --outdir test_data
nextflow run . -profile test

# 2. On your own data, single manifest, first pass (QC review):
nextflow run . --manifest /path/to/sample_manifest.csv

#    ...look at results/<manifest-name>/analysis_outputs/qc_explore.html,
#    edit thresholds in the manifest if needed, then run the identical
#    command again for the full report:
nextflow run . --manifest /path/to/sample_manifest.csv

# 3. Skip the review step entirely (batch jobs / re-runs of a known experiment):
nextflow run . --manifest /path/to/sample_manifest.csv --mode auto

# 4. Several manifests at once, one Nextflow task each, run in parallel:
nextflow run . --manifest 'manifests/*.csv' --mode auto
```

Add `-profile docker` (after building the image, see below), `-profile
conda`, or `-resume` as needed. `-profile test` and real-data runs can be
combined with `docker`/`conda`, e.g. `-profile test,docker`.

## Inputs

- **`--manifest`** (required): a sample manifest CSV/TSV, or a glob matching
  several. See
  [`perturbseq_report_v1.3.5/examples/sample_manifest_example.csv`](perturbseq_report_v1.3.5/examples/sample_manifest_example.csv)
  and [`perturbseq_report_v1.3.5/examples/WHITELISTS.md`](perturbseq_report_v1.3.5/examples/WHITELISTS.md)
  for the manifest schema, the optional gRNA/hashtag whitelist CSVs, and what
  each column means. Every path column (`h5ad_path`, `grna_whitelist`,
  `hashtag_whitelist`, `dragen_path`) resolves relative to the manifest's own
  directory, exactly as it does when running the CLI directly -- this
  pipeline changes nothing about that.
- The manifest's own `output_path` column is **ignored** by this pipeline
  (overridden with `--output-path` so results land inside each task's Nextflow
  work directory and get published cleanly under `--outdir`). It still has to
  be present and non-blank for the manifest to validate, since the CLI
  requires it, but its value doesn't matter here.

## Outputs

Published under `<outdir>/<manifest-basename>/`:

```
analysis_outputs/
  qc_explore.html | qc_report.html   the report (see above for which one)
  figures/                            PNGs, also embedded in the HTML
  tables/                             every table as CSV
  checkpoints/                        intermediate state
  artifacts.json                      everything the report is built from
  config_used.json                    the fully-resolved config for this run
run.log                               full stdout/stderr of the run
<manifest-name>                       the manifest AS USED (thresholds written back)
<manifest-name>.bak-<timestamp>       backup of the manifest before it was rewritten
```

Pipeline-level execution reports (timeline, trace, DAG) land under
`<outdir>/pipeline_info/`.

## Parameters

Every `perturbseq-report` CLI flag has a matching `--param` here (see
`nextflow.config` for the full list and defaults, or `perturbseq_report_v1.3.5/perturbseq_report/cli.py`
for what each one does). The most common:

| param                 | CLI flag equivalent      | notes |
|-----------------------|---------------------------|-------|
| `--mode`              | `--explore` / `--auto-thresholds` | see above |
| `--min_genes` etc.    | `--min-genes` etc.        | the 5 QC thresholds |
| `--config`             | `--config`               | JSON/YAML config-override file |
| `--counts_layer`       | `--counts-layer`         | when `X` isn't raw counts |
| `--subsample_cells`    | `--subsample-cells`      | escape hatch for very large objects |
| `--conditions`         | `--conditions`           | space-separated column names |
| `--doublets`           | `--doublets`             | off by default upstream |
| `--batch_correct`      | `--batch-correct`        | `none` (default) or `harmony` |
| `--resolution`         | `--resolution`           | Leiden resolution |
| `--extra_args`         | (anything)               | passed through verbatim, for flags with no dedicated param |

## Environments

- **conda** (`-profile conda`): builds from [`envs/environment.yml`](envs/environment.yml).
- **Docker** (`-profile docker`): build the image first --
  `docker build -t pipseq-qcreport:1.3.5 -f containers/Dockerfile .`
  (run from this directory). Push it to a registry you control and update
  `process.container` in `nextflow.config` if you need it available to a
  remote executor.
- **Singularity** (`-profile singularity`): build a `.sif` from the Docker
  image (see comment in `nextflow.config`).
- **Bare metal**: install [`perturbseq_report_v1.3.5/requirements.txt`](perturbseq_report_v1.3.5/requirements.txt)
  yourself and run with no `-profile` (or `-profile local`).

## Resources

`conf/base.config` defaults to 4 CPUs / 32 GB (64 GB for the main analysis
process, retried up to 3x with escalating memory on an OOM-like exit code).
The vendored package's own docs quote **~48-64 GB RAM and 1-2 hours**
single-threaded for a reference 339k-cell x 38.6k-gene experiment; scale
`conf/base.config` to your executor and typical experiment size. `-profile
test` overrides this down to 2 CPUs / 4 GB for the synthetic smoke test.

## A note on what this wrapper does and doesn't validate

Nextflow's job here is orchestration, not analysis: it fans a batch of
manifests out into parallel, resumable, resource-managed tasks and runs the
existing, unmodified `perturbseq_report` CLI inside each one. It does not
re-implement or re-check anything about the QC/guide-calling/hashtag/
clustering/DE logic -- see `perturbseq_report_v1.3.5/tests/` (in particular
`test_end_to_end.py`, which checks the pipeline recovers ground truth planted
into a synthetic dataset) for that layer's own correctness testing, and its
CHANGELOG files for the history of what has already been found and fixed.
