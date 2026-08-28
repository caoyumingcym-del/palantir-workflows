#!/usr/bin/env bash
# Build the PIPSeq_QCreport image and push it to Google Artifact Registry,
# in the same project/region other tools in this lab already use
# (methods-dev-lab / us-central1 -- see `gcloud artifacts repositories list`
# for siblings like lrtools-qc, mas-tools, etc.).
#
# This is the path actually in use: ICA does not reliably apply Nextflow
# profiles and has no configured credentials for pulling from a private GCP
# registry, so the pushed repository is made PUBLIC (see the IAM step below)
# -- a plain unauthenticated pull, exactly like a public Docker Hub image.
# Only the container's own dependency list is exposed this way, not any
# data; make sure that's still the right call before rerunning this against
# a different project.
#
# Usage:
#   ./build_and_push_gcp.sh              # tags the image 1.3.5
#   ./build_and_push_gcp.sh v2           # or pick your own tag
#
# Requires the gcloud CLI, authenticated (`gcloud auth list` should show an
# active account) with permission to create Artifact Registry repositories
# and set IAM policy in the target project, and Docker.

set -euo pipefail

GCP_PROJECT="${GCP_PROJECT:-methods-dev-lab}"
GCP_REGION="${GCP_REGION:-us-central1}"
REPO_NAME="pipseq-qcreport"
TAG="${1:-1.3.5}"
REGISTRY_HOST="${GCP_REGION}-docker.pkg.dev"
IMAGE="${REGISTRY_HOST}/${GCP_PROJECT}/${REPO_NAME}/${REPO_NAME}:${TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_DIR="$(dirname "$SCRIPT_DIR")"   # PIPSeq_QCreport/, so the Dockerfile's
                                          # COPY perturbseq_report_v1.3.5/ resolves

echo "Authenticating docker to ${REGISTRY_HOST} ..."
gcloud auth configure-docker "$REGISTRY_HOST" --quiet --project="$GCP_PROJECT"

if ! gcloud artifacts repositories describe "$REPO_NAME" \
        --location="$GCP_REGION" --project="$GCP_PROJECT" >/dev/null 2>&1; then
    echo "Creating Artifact Registry repository '${REPO_NAME}' in ${GCP_PROJECT}/${GCP_REGION} ..."
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --description="Container for PIPSeq_QCreport (perturbseq_report_v1.3.5), for ICA pipeline execution"
fi

echo "Building ${IMAGE} ..."
docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$CONTEXT_DIR"

echo "Pushing ${IMAGE} ..."
docker push "$IMAGE"

echo "Making the '${REPO_NAME}' repository publicly readable (ICA has no GCP credentials to pull with otherwise) ..."
gcloud artifacts repositories add-iam-policy-binding "$REPO_NAME" \
    --location="$GCP_REGION" \
    --project="$GCP_PROJECT" \
    --member=allUsers \
    --role=roles/artifactregistry.reader >/dev/null

echo "Verifying an unauthenticated pull actually works ..."
docker rmi "$IMAGE" >/dev/null 2>&1 || true
tmp_docker_config="$(mktemp -d)"
echo '{}' > "$tmp_docker_config/config.json"
DOCKER_CONFIG="$tmp_docker_config" docker pull "$IMAGE"
rm -rf "$tmp_docker_config"

cat <<EOF

Done -- verified pullable with zero credentials. Pass this to the pipeline
as --qc_container (it's also now the default in nextflow.config, so most
runs won't need to set it explicitly at all):

  ${IMAGE}
EOF
