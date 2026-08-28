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
# Does NOT rely on the base image's own /usr/local/bin/_entrypoint.sh
# (documented at https://github.com/mamba-org/micromamba-docker) to put the
# conda env on PATH at runtime -- ICA's execution engine overrides the
# container ENTRYPOINT with /usr/bin/env (visible in its own task log:
# "The container ENTRYPOINT was overridden ... to ensure correct Nextflow
# task execution"), which skips that script entirely and produces
# "python3: command not found" despite the image being pulled and used
# correctly. Confirmed by reproducing locally:
#   docker run --entrypoint /usr/bin/env pipseq-qcreport:1.3.5 python3 --version
# PATH is set explicitly below via ENV instead, which is baked into the
# image's own metadata and applies unconditionally -- entrypoint or not.

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

# Baked into image metadata (unlike the entrypoint script's runtime PATH
# export), so python3 and every installed package stay reachable even when
# an orchestrator replaces or skips ENTRYPOINT entirely.
ENV PATH=/opt/conda/bin:/opt/conda/condabin:$PATH
# ICA runs the container as a UID with no corresponding /etc/passwd entry
# and no writable home directory, so anything that caches to $HOME (or
# XDG_CACHE_HOME/XDG_CONFIG_HOME, which default to under $HOME) fails --
# matplotlib hit this first ("mkdir -p failed for path /.config/matplotlib:
# Permission denied") and degraded gracefully to a /tmp fallback, but
# fontconfig's and numba's caches are less forgiving about it. /tmp is
# writable regardless of which UID ends up running the container, so
# pointing HOME there directly avoids the whole class of failure rather
# than patching each library's own cache-dir override one at a time.
ENV HOME=/tmp
ENV PYTHONUNBUFFERED=1
