FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/.hf_cache \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    OPTISEEK_RAG_INDEX_DIR=/app/index \
    OPTISEEK_RAG_SOCKET=/app/daemon.sock

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt
RUN torch_version=$(python3 -c "import torch; print(torch.__version__)") && \
    echo "torch==${torch_version}" > /tmp/constraints.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt -r /app/requirements.txt

COPY app/ /app/
RUN mkdir -p /app/output /app/corpus /app/index

ARG BAKE_WEIGHTS=true
RUN if [ "$BAKE_WEIGHTS" = "true" ]; then \
        mkdir -p /models && \
        python3 -c "from huggingface_hub import snapshot_download; \
                     snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='/models/Qwen2.5-7B-Instruct'); \
                     snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', local_dir='/models/Qwen2.5-VL-3B-Instruct'); \
                     snapshot_download('BAAI/bge-small-en-v1.5', local_dir='/models/bge-small-en-v1.5')" && \
        rm -rf /models/.hf_cache; \
    fi

CMD ["bash", "-c", "python3 /app/daemon.py & sleep infinity"]
