FROM golang:1.24-alpine AS go-builder

ARG TARGETOS
ARG TARGETARCH

WORKDIR /src
COPY syl/parsers/parser.go .

RUN echo "Building Go parser for ${TARGETOS:-linux}/${TARGETARCH:-amd64}"
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -ldflags="-w -s" -o go-parser parser.go

FROM python:3.12-slim AS python-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --user \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.* torchvision==0.22.* torchaudio==2.7.*

# There aren't enough service-specific dependencies
# to warrant installing them in their respective
# Dockerfiles and skipping the distroless build.
#
# So install them all here and then wrap with different
# entrypoints per service.
COPY requirements/base.txt requirements/server.txt requirements/watcher.txt ./
RUN cat base.txt index.txt server.txt watcher.txt | sort -u > combined-requirements.txt

RUN pip install --no-cache-dir --user -r combined-requirements.txt


COPY pyproject.toml .
COPY syl/ syl/
RUN pip install --no-cache-dir --user .


RUN find /root/.local -name "*.pyc" -delete && \
    find /root/.local -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -name "*.pyo" -delete && \
    find /root/.local -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.local -name "test" -type d -exec rm -rf {} + 2>/dev/null || true

FROM python:3.12-slim

# TODO I'm sure there's a python SDK for this
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=go-builder /src/go-parser /usr/local/bin/go-parser
RUN chmod +x /usr/local/bin/go-parser

COPY --from=python-builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/root/.local/lib/python3.12/site-packages

WORKDIR /app

# DIND -- sorry not dealing with that perm nightmare
USER root
