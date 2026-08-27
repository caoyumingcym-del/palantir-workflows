#!/usr/bin/env bash
# Build the PIPSeq_QCreport image and push it to AWS ECR.
#
# Mirrors the convention used by SingleCell/PIPseqDownsample/docker
# elsewhere in this repo, so both pipelines' container images are built and
# published the same way in this lab.
#
# Usage:
#   export AWS_REGION=us-east-1
#   export ECR_REGISTRY=<account-id>.dkr.ecr.us-east-1.amazonaws.com
#   ./build_and_push.sh              # tags the image 1.3.5
#   ./build_and_push.sh v2           # or pick your own tag
#
# Requires the AWS CLI (authenticated: `aws sts get-caller-identity` should
# work) and Docker. See SETUP.md for the one-time IAM/repository setup.

set -euo pipefail

: "${AWS_REGION:?Set AWS_REGION, e.g. us-east-1}"
: "${ECR_REGISTRY:?Set ECR_REGISTRY, e.g. <account-id>.dkr.ecr.us-east-1.amazonaws.com}"

REPO_NAME="pipseq-qcreport"
TAG="${1:-1.3.5}"
IMAGE="${ECR_REGISTRY}/${REPO_NAME}:${TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_DIR="$(dirname "$SCRIPT_DIR")"   # PIPSeq_QCreport/, so the Dockerfile's
                                          # COPY perturbseq_report_v1.3.5/ resolves

echo "Authenticating docker to ${ECR_REGISTRY} ..."
aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "$ECR_REGISTRY"

if ! aws ecr describe-repositories --region "$AWS_REGION" \
        --repository-names "$REPO_NAME" >/dev/null 2>&1; then
    echo "Creating ECR repository '${REPO_NAME}' in ${AWS_REGION} ..."
    aws ecr create-repository --region "$AWS_REGION" \
        --repository-name "$REPO_NAME" >/dev/null
fi

echo "Building ${IMAGE} ..."
docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$CONTEXT_DIR"

echo "Pushing ${IMAGE} ..."
docker push "$IMAGE"

cat <<EOF

Done. Pass this image to the pipeline as --qc_container, e.g.:

  nextflow run . -profile docker \\
      --qc_container ${IMAGE} \\
      --manifest sample_manifest.csv

On ICA, paste "${IMAGE}" into the qc_container field of the launch form.
EOF
