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

# 2. On your own data, single manifest, first pass (QC review). qc_container
#    already defaults to a published, public image -- no setup needed:
nextflow run . -profile docker --manifest /path/to/sample_manifest.csv

#    ...look at results/<manifest-name>/analysis_outputs/qc_explore.html,
#    edit thresholds in the manifest if needed, then run the identical
#    command again for the full report:
nextflow run . -profile docker --manifest /path/to/sample_manifest.csv

# 3. Skip the review step entirely (batch jobs / re-runs of a known experiment):
nextflow run . -profile docker --manifest /path/to/sample_manifest.csv --mode auto

# 4. Several manifests at once, one Nextflow task each, run in parallel:
nextflow run . -profile docker --manifest 'manifests/*.csv' --mode auto
```

`-profile docker` is what makes Nextflow actually run each task inside
`docker run`; `--qc_container` (defaulted, see Environments below) is
*which* image it uses -- override it only if you've built and pushed your
own copy. Add `-resume` as needed. `-profile test` and real-data runs can be
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
  work directory and get published cleanly under `--outdir`) and doesn't need
  to be present in the manifest at all.
- **`--h5ad`** (for ICA specifically; optional on a local/HPC run): overrides
  the manifest's `h5ad_path` column with an explicitly staged file, and once
  given the manifest doesn't need a valid `h5ad_path` column either. **You
  need this on ICA.** ICA does not mount a project's data tree into the task
  container, so a path typed into the manifest -- relative or absolute,
  doesn't matter -- resolves to nothing once the container starts; only
  files declared as actual pipeline inputs (like `manifest` itself, or this
  one) get staged in. On a local/HPC run where the manifest's own
  `h5ad_path` is already reachable from wherever the task executes, leave
  this blank -- same for `grna_whitelist`/`hashtag_whitelist`, which don't
  have an equivalent override yet and are subject to the same limitation on
  ICA if you use them.
- **`--dragen_root`** (optional, for the report's Sequencing QC section):
  the same fix as `--h5ad`, for the manifest's per-run `prefix`/`dragen_path`
  columns instead of its (manifest-wide) `h5ad_path`. Point it at a directory
  containing every run's DRAGEN output subfolder, each named the same way
  the manifest's own `dragen_path` values already end (their basename) --
  this overrides every run's `dragen_path` to `<dragen_root>/<that
  basename>`. Unlike `h5ad`, this is genuinely optional: leaving it blank on
  ICA just means the "Sequencing QC" section (upstream DRAGEN read/mapping
  metrics) is skipped -- transcriptome, guide, hashtag and perturbation
  analysis are all unaffected either way.
- **`--qc_container`** (for `-profile docker`/`singularity`/ICA): the image
  every step runs in. Defaults to a published, public image (Google Artifact
  Registry) -- see Environments below; only set this yourself if you've
  rebuilt and pushed your own copy.

## Outputs

Published under `<outdir>/<manifest-basename>/`:

```
analysis_outputs/
  qc_explore.html | qc_report.html   the report (see above for which one)
  qc_deck.pptx                        slide deck built from artifacts.json (see below)
  figures/                            PNGs, also embedded in the HTML
  tables/                             every table as CSV
  checkpoints/                        intermediate state
  artifacts.json                      everything the report (and the deck) is built from
  config_used.json                    the fully-resolved config for this run
run.log                               full stdout/stderr of the run
build_slides.log                      stdout/stderr of the slide-deck build
<manifest-name>                       the manifest AS USED (thresholds written back)
<manifest-name>.bak-<timestamp>       backup of the manifest before it was rewritten
```

`qc_deck.pptx` is a Google-Slides-ready deck built from `artifacts.json` by
[`build_slides.py`](perturbseq_report_v1.3.5/build_slides.py) -- a standalone
script in the vendored package, not something `perturbseq-report` itself
produces; `--build_slides false` skips this step (`artifacts.json` is still
written either way, so you can build one manually later regardless). Drag it
into Drive and open with Google Slides, or use **File > Import slides** in an
existing deck.

Pipeline-level execution reports (timeline, trace, DAG) land under
`<outdir>/pipeline_info/`.

## Parameters

Every `perturbseq-report` CLI flag has a matching `--param` here (see
`nextflow.config` for the full list and defaults, or `perturbseq_report_v1.3.5/perturbseq_report/cli.py`
for what each one does). The most common:

| param                 | CLI flag equivalent      | notes |
|-----------------------|---------------------------|-------|
| `--h5ad`              | `--h5ad`                  | required on ICA -- see Inputs above |
| `--dragen_root`       | `--dragen-root`           | optional; enables the Sequencing QC section on ICA -- see Inputs above |
| `--qc_container`      | n/a (Nextflow `container` directive) | defaulted (public GCP image); see Environments |
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
| `--build_slides`       | n/a (runs build_slides.py) | on by default; produces analysis_outputs/qc_deck.pptx |
| `--slides_extra_args`  | (anything, to build_slides.py) | passed through verbatim, for its flags with no dedicated param |

## Environments

- **Docker** (`-profile docker`): `--qc_container` already defaults (see
  `nextflow.config`) to
  `us-central1-docker.pkg.dev/methods-dev-lab/pipseq-qcreport/pipseq-qcreport:1.3.5`
  -- a Google Artifact Registry image built from `containers/Dockerfile`,
  made deliberately **public** (verified with an actual unauthenticated
  `docker pull`) so ICA can pull it with zero credentials configured, the
  same way it would pull a public Docker Hub image. It is **not** hardcoded
  via a profile -- `qc_container` is a plain parameter that
  `RUN_QC_REPORT`'s `container` directive reads directly. That's
  deliberate: `-profile docker` only tells Nextflow to actually invoke
  `docker run`; it does not by itself say *which* image, and a hardcoded
  image tucked inside a profile block turned out to be exactly what
  silently never activated when this pipeline was first run on ICA (see
  "Running on ICA" below). Rebuild and republish with
  [`containers/build_and_push_gcp.sh`](containers/build_and_push_gcp.sh);
  see [`containers/SETUP.md`](containers/SETUP.md) for that and the AWS ECR
  alternative.
- **Singularity** (`-profile singularity`): build a `.sif` from the pushed
  Docker image (see comment in `nextflow.config`) and pass its path as
  `--qc_container`.
- **conda** (`-profile conda`): builds from [`envs/environment.yml`](envs/environment.yml).
  No container/`--qc_container` needed.
- **Bare metal**: install [`perturbseq_report_v1.3.5/requirements.txt`](perturbseq_report_v1.3.5/requirements.txt)
  yourself and run with no `-profile` (or `-profile local`). No
  `--qc_container` needed.

## Running on ICA

`inputForm.json` and `nextflow_schema.json` are both hand-authored (not
ICA's auto-generated form -- that guesses a widget type per parameter with
no real type information and gets several wrong, e.g. rendering string
parameters as checkboxes). Point ICA's pipeline import at
`PIPSeq_QCreport/inputForm.json` for the launch form and
`PIPSeq_QCreport/nextflow.config` / `PIPSeq_QCreport/main.nf` for the
pipeline itself.

**Three things that are easy to miss on ICA specifically:**

1. **`h5ad` needs to be filled in -- ICA does not mount your project's data
   into the task container.** Only files explicitly attached as pipeline
   inputs get staged; a path typed into the manifest's `h5ad_path` column,
   however it's written, resolves to nothing once the container starts. This
   surfaced as `pipeline error: h5ad file not found: <path>`, reproducibly,
   with both a relative and an absolute path in that column, and disappeared
   entirely once the file was attached through the `h5ad` field instead. The
   same limitation applies to `dragen_path` (the `dragen_root` field fixes
   it -- see Inputs above; it just means a skipped "Sequencing QC" section,
   not a failed run, if you skip it) and to `grna_whitelist`/
   `hashtag_whitelist` if you use them, which don't have an equivalent
   override field yet.
2. **`qc_container` is read directly by the process, not a profile**, for
   the reason above: ICA does not reliably apply `-profile docker`. It
   already defaults to a public GCP image (see Environments), so most
   launches don't need to touch this field at all. Only override it if
   you've rebuilt and pushed your own copy -- `containers/SETUP.md` covers
   rebuilding to GCP, an AWS ECR alternative, and (if neither cloud account
   is available to you) uploading a TAR directly into ICA's own Docker
   Repository.
3. **ICA's git-based pipeline import pins to a specific commit, not a
   branch, and the ICA UI doesn't let you edit that commit after the
   initial import.** Pushing a new commit to this branch does **not**
   update an already-imported ICA pipeline. [`ica_tools/`](ica_tools/)
   automates re-importing at the current commit (adapted from
   `SingleCell/PIPseqDownsample/ica_tools/export_pipeline_to_ica.py`
   elsewhere in this repo, which uses the same API the UI's "Import from
   Git" wizard does) -- see `ica_tools/README.md` for one-time setup, then
   `python3 ica_tools/export_pipeline_to_ica.py` after every commit you
   want ICA to pick up. Each run creates a new pipeline entry rather than
   updating one in place, so confirm you're launching the one that matches
   your latest commit.

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
