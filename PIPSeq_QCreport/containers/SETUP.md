# Building and pushing the qc_container image

The pipeline reads its container image from `--qc_container` (see
`nextflow.config`) rather than a Nextflow profile -- ICA does not reliably
apply profiles, so a container only wired up under `-profile docker` can
silently never activate on it (confirmed: this pipeline's first ICA run had
no container at all and failed on `python3: command not found`). You build
the image once (and again after any change to `perturbseq_report_v1.3.5/` or
`envs/environment.yml`), get it somewhere your executor can pull it from,
and pass that reference as `--qc_container`.

## Currently in use: Google Artifact Registry (public)

`nextflow.config`'s default `qc_container` points at:

```
us-central1-docker.pkg.dev/methods-dev-lab/pipseq-qcreport/pipseq-qcreport:1.3.5
```

This repository is deliberately **public** (`allUsers` granted
`roles/artifactregistry.reader`) -- ICA has no GCP credentials configured, so
a private image would be unpullable. Verified with an actual unauthenticated
`docker pull` (empty `DOCKER_CONFIG`, no `gcloud` credential helper) that
this works exactly like a public Docker Hub image. Only the container's
dependency list is exposed this way, not any data.

To rebuild and republish after a change:

```bash
cd PIPSeq_QCreport/containers
export GCP_PROJECT=methods-dev-lab      # defaults shown; override if needed
export GCP_REGION=us-central1
./build_and_push_gcp.sh                 # tags the image 1.3.5
# ./build_and_push_gcp.sh v2            # or pass your own tag
```

This authenticates Docker via `gcloud`, creates the Artifact Registry
repository if it doesn't already exist, builds, pushes, re-applies the
public IAM binding, and verifies an unauthenticated pull actually succeeds
before printing the final image reference. Needs the `gcloud` CLI
authenticated with permission to create Artifact Registry repositories and
set IAM policy in the target project.

If you rebuild with a different tag, update `qc_container`'s default in
`nextflow.config` (and in `inputForm.json` / `nextflow_schema.json`) to
match, or just pass `--qc_container` explicitly on each run.

## Alternative: your own AWS account's ECR

Useful if GCP isn't an option for you, or you'd rather keep the image
private. Mirrors `SingleCell/PIPseqDownsample/docker`'s convention elsewhere
in this repo.

### Prerequisites

- Docker, and the AWS CLI authenticated (`aws sts get-caller-identity`
  should succeed).
- An IAM identity with permission to create/push to an ECR repository
  (`ecr:CreateRepository`, `ecr:GetAuthorizationToken`, `ecr:PutImage`, etc.)
  in the target account/region.
- Which account/region to use -- ask whoever manages the ECR registry your
  ICA project's compute environment can actually pull from. There is no
  hardcoded account ID in this repo; each deployment sets its own. Unlike
  the GCP path above, a private ECR repository also needs ICA's compute
  environment to actually be authorized to pull from that account --
  confirm that's already true for your project before relying on this path.

### Build and push

```bash
export AWS_REGION=us-east-1
export ECR_REGISTRY=<account-id>.dkr.ecr.us-east-1.amazonaws.com
cd PIPSeq_QCreport/containers
./build_and_push.sh            # tags the image 1.3.5
# ./build_and_push.sh v2       # or pass your own tag
```

This authenticates Docker to ECR, creates the `pipseq-qcreport` repository if
it doesn't already exist, builds from `Dockerfile`, and pushes. It prints the
full image reference at the end -- that's the value to pass as
`--qc_container` (or paste into ICA's `qc_container` field, overriding the
GCP default).

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
