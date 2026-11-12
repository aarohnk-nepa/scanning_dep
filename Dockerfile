# syntax=docker/dockerfile:1.7
# Outbound Vision API - built with uv only.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=automatic \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libzbar0 libgl1 libglib2.0-0 curl git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# e.g. UV_EXTRA_ARGS="--extra ocr" to bake the OCR engine into the image
ARG UV_EXTRA_ARGS=""

# 1) dependencies only (cached until pyproject.toml / uv.lock change)
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev ${UV_EXTRA_ARGS}

# 2) project source
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev ${UV_EXTRA_ARGS}

# 3) bake Grounding DINO weights into the image so startup needs no download
ARG GDINO_MODEL_ID=IDEA-Research/grounding-dino-tiny
RUN uv run --no-sync python -c "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection as M; AutoProcessor.from_pretrained('${GDINO_MODEL_ID}'); M.from_pretrained('${GDINO_MODEL_ID}')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uv", "run", "--no-sync", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
