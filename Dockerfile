# Qwen3.8-Flash-Next (uncensored EXL3) — one DGX Spark, one Docker image
#
# Builds a self-contained serving image for GB10 / aarch64:
#   CUDA 13 devel base -> Python venv -> torch cu130 -> the sm_121 exllamav3
#   fork (vcruz305 @523ecd3) compiled from source -> TabbyAPI (OpenAI endpoint).
#
# The model is NOT baked in (it is 72 GB); mount it at /model. See README.
#
# Build:  docker build -t qwen38-flash-next-exl3:3bpw .
# Run:    see compose.yaml or the README one-liner.
#
# Pin these to reproduce the measured 79.9 tok/s configuration.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG EXLLAMAV3_REPO=https://github.com/vcruz305/exllamav3.git
ARG EXLLAMAV3_COMMIT=523ecd3
ARG TABBY_REPO=https://github.com/theroyallab/tabbyAPI.git
ARG TABBY_COMMIT=main
ARG TORCH_VERSION=2.14.0
ARG CUDA_INDEX=https://download.pytorch.org/whl/cu130
# GB10 / Blackwell consumer = sm_121
ARG TORCH_CUDA_ARCH_LIST=12.1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    # GB10 tuning for the EXL3 kernels (see docs/FINDINGS.md)
    EXL3_INT8_GEMV=0 \
    EXL3_MOE_COOP_WIDE=1 \
    EXL3_GR_INT8=1 \
    EXL3_MTP_HEAD_N=65536 \
    EXL3_NGRAM_STREAM=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev python3-pip \
      git build-essential ninja-build cmake pkg-config \
      ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/*

# ---- Python env: torch (cu130, aarch64 wheels) -----------------------------
RUN python3 -m venv "$VIRTUAL_ENV" \
    && pip install --upgrade pip wheel setuptools \
    && pip install "torch==${TORCH_VERSION}" --index-url "${CUDA_INDEX}"

# ---- exllamav3 (fork, compiled for sm_121) ---------------------------------
RUN git clone "${EXLLAMAV3_REPO}" /opt/exllamav3 \
    && git -C /opt/exllamav3 checkout "${EXLLAMAV3_COMMIT}" \
    && pip install -r /opt/exllamav3/requirements.txt \
    && pip install --no-build-isolation /opt/exllamav3 \
    && python -c "import torch, exllamav3_ext; print('exllamav3_ext OK', torch.__version__)"

# ---- TabbyAPI (OpenAI-compatible server) -----------------------------------
RUN git clone "${TABBY_REPO}" /opt/tabbyAPI \
    && git -C /opt/tabbyAPI checkout "${TABBY_COMMIT}" \
    && pip install --no-build-isolation /opt/tabbyAPI

# TabbyAPI's pyproject gates uvloop to x86_64 ("platform_machine == 'x86_64'"),
# but main.py imports it unconditionally, so an arm64 install dies at startup
# with ModuleNotFoundError. The aarch64 wheel exists; install it explicitly.
RUN pip install "uvloop>=0.21"

# ---- our serving glue ------------------------------------------------------
COPY config.yml /opt/tabbyAPI/config.yml
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY scripts/advise-model-files.py /usr/local/bin/advise-model-files.py
# spark-stats: the in-image metrics endpoint (see docs/STANDARD.md)
COPY stats/spark_stats.py /opt/spark-stats/spark_stats.py
COPY stats/container.json /opt/spark-stats.json
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/advise-model-files.py

WORKDIR /opt/tabbyAPI
# 5000 = OpenAI API, 8787 = spark-stats metrics
EXPOSE 5000 8787

# tini reaps the process group so SIGTERM stops the server cleanly.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
