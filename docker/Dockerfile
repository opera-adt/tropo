FROM condaforge/mambaforge:latest

LABEL description='OPERA_TROPO'

# For opencontainers label definitions, see:
#    https://github.com/opencontainers/image-spec/blob/master/annotations.md
LABEL org.opencontainers.image.title="opera_tropo"
LABEL org.opencontainers.image.description=""
LABEL org.opencontainers.image.authors="Marin Govorcin<marin.govorcin@jpl.nasa.gov>"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.url="https://github.com/opera-adt/tropo"
LABEL org.opencontainers.image.source="https://github.com/opera-adt/tropo"
LABEL org.opencontainers.image.documentation="https://github.com/opera-adt/tropo"

# Dynamic lables to define at build time via `docker build --label`
# LABEL org.opencontainers.image.created=""
# LABEL org.opencontainers.image.version=""
# LABEL org.opencontainers.image.revision=""

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=true

# Install dependencies in a single RUN command
RUN apt-get update && apt-get install -y --no-install-recommends \
    git unzip vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# run commands in a bash login shell
SHELL ["/bin/bash", "-l", "-c"]

# Create non-root user/group with default inputs
ARG CONDA_UID=1000
ARG CONDA_GID=1000

RUN groupadd -g "${CONDA_GID}" --system tropo_user && \
    useradd -l -u "${CONDA_UID}" -g "${CONDA_GID}" --system -d /home/tropo -m  -s /bin/bash tropo_user && \
    mkdir -p /home/ops && \
    chown -R tropo_user:tropo_user /home/ops /opt /home/tropo && \
    chmod -R 777 /home/ops /home/tropo

USER ${CONDA_UID}
WORKDIR /home/ops
ENV HOME="/home/tropo"
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV HERBIE_CONFIG_PATH=/tmp/herbie_config.toml

# Copy files while preserving ownership
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#leverage-build-cach
COPY --chown=tropo_user:tropo_user conda-env.yml /home/ops/opera_tropo/conda-env.yml
COPY --chown=tropo_user:tropo_user . /home/ops/opera_tropo

# Ensure all files are read/write by the user
RUN chown -R tropo_user:tropo_user /home/ops && chmod -R 777 /home/ops

# Create the environment with mamba
RUN mamba env create -f /home/ops/opera_tropo/conda-env.yml && \
    conda clean -afy

ENV CONDA_DEFAULT_ENV=opera_tropo
ENV PATH="/opt/conda/envs/${CONDA_DEFAULT_ENV}/bin:$PATH"

# Install repository with pip
RUN python -m pip install --no-cache-dir /home/ops/opera_tropo
