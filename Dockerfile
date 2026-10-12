# ==========================================================================
# Optiseek RAG — AMD LabLab AI Academy Mini Challenge 3 (RAG)
# ==========================================================================
#
# MANDATORY BASE IMAGE (do NOT change):
#   rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
#
# The grader verifies base-image identity by LAYER HASH. Do NOT squash.
#
# Image size budget: 60 GiB uncompressed. We target ~50 GiB.
#   - ROCm base:       ~29 GiB
#   - Qwen2.5-7B LLM:  ~14 GiB (bf16)
#   - Qwen2.5-VL-3B:   ~6 GiB  (bf16, for image text extraction)
#   - BGE embeddings:  ~0.1 GiB
#   - Deps:            ~1 GiB
# ==========================================================================

FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

# -------------------------------------------------------------------------
# 1. Environment
# -------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/.hf_cache \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    OPTISEEK_RAG_INDEX_DIR=/app/index \
    OPTISEEK_RAG_SOCKET=/app/daemon.sock

WORKDIR /app

# -------------------------------------------------------------------------
# 2. Python dependencies
# -------------------------------------------------------------------------
COPY app/requirements.txt /app/requirements.txt

# Generate a constraints file that pins torch to the base image version,
# preventing pip from silently replacing ROCm torch with a CUDA build.
RUN torch_version=$(python3 -c "import torch; print(torch.__version__)") && \
    echo "torch==${torch_version}" > /tmp/constraints.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt -r /app/requirements.txt

# -------------------------------------------------------------------------
# 3. Application code
# -------------------------------------------------------------------------
COPY app/ /app/

RUN mkdir -p /app/output /app/corpus /app/index

# -------------------------------------------------------------------------
# 4. Model weights (baked into image — no network at eval time)
# -------------------------------------------------------------------------
# Ship three models:
#   - Qwen2.5-7B-Instruct:     LLM for answer generation + citation reasoning
#   - Qwen2.5-VL-3B-Instruct:  VLM for image text extraction during indexing
#   - BAAI/bge-small-en-v1.5:  Embedding model for retrieval
#
# Build modes:
#   docker build -t optiseek-rag:latest .
#   docker build --build-arg MODEL_DIR=/persistent/models -t optiseek-rag:latest .

ARG BAKE_WEIGHTS=true
ARG MODEL_DIR=""

RUN if [ "$BAKE_WEIGHTS" = "true" ]; then \
        mkdir -p /models && \
        python3 -c "from huggingface_hub import snapshot_download; \
                     snapshot_download('Qwen/Qwen2.5-7B-Instruct', \
                                       local_dir='/models/Qwen2.5-7B-Instruct'); \
                     snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', \
                                       local_dir='/models/Qwen2.5-VL-3B-Instruct'); \
                     snapshot_download('BAAI/bge-small-en-v1.5', \
                                       local_dir='/models/bge-small-en-v1.5')" && \
        rm -rf /models/.hf_cache; \
    else \
        echo "Skipping weight baking (BAKE_WEIGHTS=false)"; \
    fi

# -------------------------------------------------------------------------
# 5. Runtime
# -------------------------------------------------------------------------
# The daemon holds the LLM and listens on a unix socket.
# app.py is a thin client that connects to the daemon for --index and --query.
# CMD starts the daemon in the background, then keeps the container alive.

# Sanity check
RUN test -f /app/daemon.py || (echo "FATAL: /app/daemon.py missing" && exit 1)

CMD ["bash", "-c", "python3 /app/daemon.py & sleep infinity"]
