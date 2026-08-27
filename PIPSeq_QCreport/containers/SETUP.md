# Building and pushing the qc_container image

The pipeline reads its container image from `--qc_container` (see
`nextflow.config`) rather than a Nextflow profile -- ICA does not reliably
apply profiles, so a container only wired up under `-profile docker` can
silently never activate on it. You build the image once (and again after
any change to `perturbseq_report_v1.3.5/` or `envs/environment.yml`), get it
somewhere your executor can pull it from, and pass that reference to every
run. Two ways to do that -- pick whichever matches how your ICA project is
set up (ask a colleague already running an ICA pipeline here if unsure):

- **Your own AWS account's ECR** -- see "Build and push" below. Needs AWS
  credentials and IAM permissions, and ICA's compute environment must be
  authorized to pull from that account.
- **ICA's own built-in Docker Repository** (System Settings) -- see
  "Uploading directly to ICA" below. No AWS account needed at all: you
  build locally, export the image as a TAR, and upload it through ICA's own
  UI. This is the path to use if, like `SingleCell/PIPseqDownsample`'s
  author, you don't have (or don't want to set up) your own AWS account for
  this.

## Prerequisites

- Docker, and the AWS CLI authenticated (`aws sts get-caller-identity`
  should succeed).
- An IAM identity with permission to create/push to an ECR repository
  (`ecr:CreateRepository`, `ecr:GetAuthorizationToken`, `ecr:PutImage`, etc.)
  in the target account/region.
- Which account/region to use -- ask whoever manages the ECR registry your
  ICA project's compute environment can actually pull from. There is no
  hardcoded account ID in this repo; each deployment sets its own.

## Build and push

```bash
export AWS_REGION=us-east-1
export ECR_REGISTRY=<account-id>.dkr.ecr.us-east-1.amazonaws.com
cd PIPSeq_QCreport/containers
./build_and_push.sh            # tags the image 1.3.5
# ./build_and_push.sh v2       # or pass your own tag
```

This authenticates Docker to ECR, creates the `pipseq-qcreport` repository if
it doesn't already exist, builds from `Dockerfile`, and pushes. It prints the
full image reference at the end -- that's the value to paste into
`qc_container` (a Nextflow CLI flag, or the `qc_container` field on ICA's
launch form).

## Manual build (no push)

Useful for iterating on the Dockerfile locally before you're ready to push:

```bash
cd PIPSeq_QCreport
docker build -t pipseq-qcreport:1.3.5 -f containers/Dockerfile .
docker run --rm pipseq-qcreport:1.3.5 \
    python3 /opt/perturbseq_report_v1.3.5/run_perturbseq_report.py --version
```

## Uploading directly to ICA (no AWS account needed)

ICA does **not** build the image for you from a Dockerfile at run time --
despite how that sometimes gets described informally, what's actually
happening is a local build followed by a manual upload into ICA's own
registry. Confirmed against Illumina's own docs: "In order to use private
images in your tool, you must first upload them as a TAR file." There is no
on-demand build from source.

1. Build locally (same as "Manual build" above) and export it as a TAR:

   ```bash
   cd PIPSeq_QCreport
   docker build -t pipseq-qcreport:1.3.5 -f containers/Dockerfile .
   docker save pipseq-qcreport:1.3.5 -o pipseq-qcreport-1.3.5.tar
   ```

   This file is large (the built image is several hundred MB) -- it's
   git-ignored (`*.tar`) and should never be committed.

2. In the ICA UI: **Projects > your project > Data**, upload the TAR
   (drag-and-drop, CLI, or Connector).
3. Select the uploaded file, **Manage > Change Format**, set it to `DOCKER`,
   save.
4. **System Settings > Docker Repository > Create > Image.** Pick the TAR
   you just uploaded, give it a name and version (this auto-fills a global
   URL), choose region / type (tool or bench) / cluster compatibility /
   access method, save.
5. Wait for it to show **Available**, then copy that image's URL into
   `qc_container` on the launch form.

Note: creating entries under System Settings may require elevated
permissions on your ICA project -- if you don't see that option, whoever
manages your ICA project's settings can do steps 4-5 for you once you've
uploaded the TAR.

## Verifying an image end to end

Beyond the import check the Dockerfile already runs at build time, it's
worth actually running the pipeline once against synthetic data before
trusting a freshly built image:

```bash
docker run --rm pipseq-qcreport:1.3.5 bash -c '
  python3 -c "
import sys; sys.path.insert(0, \"/opt/perturbseq_report_v1.3.5\")
from perturbseq_report import synthetic
h5ad_path, _ = synthetic.make_h5ad(\"/tmp/test.h5ad\", n_cells=800, n_genes=300, seed=0)
synthetic.write_manifest(\"/tmp/manifest.csv\", h5ad_path, \"/tmp/out\")
"
  python3 /opt/perturbseq_report_v1.3.5/run_perturbseq_report.py \
      --manifest /tmp/manifest.csv --auto-thresholds
'
```

A successful run ends with `report written to /tmp/out/analysis_outputs/qc_report.html`.
