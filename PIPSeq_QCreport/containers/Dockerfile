# Container for PIPSeq_QCreport (perturbseq_report_v1.3.5).
#
# Build from the PIPSeq_QCreport/ directory (not this containers/ subfolder),
# so the COPY below can reach the vendored package:
#
#   cd PIPSeq_QCreport
#   docker build -t pipseq-qcreport:1.3.5 -f containers/Dockerfile .
#
# -profile docker in nextflow.config expects the image tagged exactly
# `pipseq-qcreport:1.3.5` and the package unpacked at /opt/perturbseq_report_v1.3.5
# (params.pipeline_dir) -- keep both in sync if you retag or upgrade the package.
#
# Relies on the base image's own /usr/local/bin/_entrypoint.sh (documented at
# https://github.com/mamba-org/micromamba-docker), which activates the target
# env for any CMD it is given -- so `docker run pipseq-qcreport:1.3.5 python3 ...`
# (which is how Nextflow invokes it) runs with the env already on PATH.

FROM mambaorg/micromamba:1.5-jammy

COPY --chown=$MAMBA_USER:$MAMBA_USER envs/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes \
    && rm /tmp/environment.yml

USER root
COPY perturbseq_report_v1.3.5/ /opt/perturbseq_report_v1.3.5/
USER $MAMBA_USER

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN python3 -c "import numpy, pandas, scanpy, anndata, leidenalg, igraph, harmonypy, scrublet, yaml; print('perturbseq_report dependency check OK')"

ENV PYTHONUNBUFFERED=1
