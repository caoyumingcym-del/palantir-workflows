# Container for PIPSeq_QCreport (perturbseq_report_v1.3.5).
#
# Build and push with ./build_and_push.sh in this directory (mirrors the
# convention SingleCell/PIPseqDownsample/docker uses elsewhere in this repo):
# it pushes to AWS ECR and prints the exact --qc_container value to pass to
# the pipeline. See build_and_push.sh for the manual build/push commands if
# you'd rather run them yourself, or SETUP.md for the one-time ECR repo setup.
#
# The package unpacked at /opt/perturbseq_report_v1.3.5 is where
# modules/local/run_qc_report.nf's pipelineDir fallback expects it whenever
# --qc_container is set -- keep that path in sync if you change this COPY.
#
# Relies on the base image's own /usr/local/bin/_entrypoint.sh (documented at
# https://github.com/mamba-org/micromamba-docker), which activates the target
# env for any CMD it is given -- so `docker run <image> python3 ...` (which is
# how Nextflow invokes it) runs with the env already on PATH.

FROM mambaorg/micromamba:1.5-jammy

COPY --chown=$MAMBA_USER:$MAMBA_USER envs/environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes \
    && rm /tmp/environment.yml

USER root
COPY perturbseq_report_v1.3.5/ /opt/perturbseq_report_v1.3.5/
# The vendored source carries owner-only (700/600) permissions from its
# original checkout, which the container's non-root $MAMBA_USER can't read
# at all -- `python3 run_perturbseq_report.py` fails with "Permission
# denied" before it ever gets to import anything. a+rX adds read for
# everyone and execute only where already executable or a directory
# (needed for traversal), without touching files that were already fine.
RUN chmod -R a+rX /opt/perturbseq_report_v1.3.5
USER $MAMBA_USER

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN python3 -c "import numpy, pandas, scanpy, anndata, leidenalg, igraph, harmonypy, scrublet, yaml; print('perturbseq_report dependency check OK')"

ENV PYTHONUNBUFFERED=1
