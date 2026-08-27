# Building and pushing the qc_container image

The pipeline reads its container image from `--qc_container` (see
`nextflow.config`) rather than a Nextflow profile -- ICA does not reliably
apply profiles, so a container only wired up under `-profile docker` can
silently never activate on it. You build and push the image once (and again
after any change to `perturbseq_report_v1.3.5/` or `envs/environment.yml`),
then pass the pushed reference to every run.

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
