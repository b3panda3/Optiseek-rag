# Optiseek RAG

Retrieval-Augmented Generation over mixed-document corpora, built for the AMD LabLab AI Academy Mini Challenge 3. Indexes a folder of PDFs, Word docs, spreadsheets, CSVs, logs, source code, and images, then answers questions with precise values and exact citations.

> **Hackathon**: [AMD LabLab AI Academy Challenge — Mini Challenge 3 (RAG)](https://lablab.ai/ai-hackathons/amd-lablab-ai-academy-challenge)

---

## What it does

Optiseek RAG is a containerized RAG system that:

1. **Indexes** a corpus of mixed file types (PDF, DOCX, XLSX, CSV, TXT, LOG, PY, PNG, JPG) — parsing each format with the appropriate extractor, including a vision-language model for images.
2. **Answers questions** about the corpus, returning the value only (not a sentence) and the exact set of source files necessary to derive that answer.
3. **Refuses honestly** — when the corpus doesn't contain the answer, it returns an empty answer with no citations, rather than guessing.

### Key capabilities

- **Multi-format parsing**: PDF (PyMuPDF), DOCX (python-docx), XLSX (openpyxl, multi-sheet), CSV, plain text, Python source, and images (via VLM)
- **Vision-language extraction**: Text printed inside images (diagrams, pinout charts, asset labels) is extracted at index time using Qwen2.5-VL-3B-Instruct
- **Semantic retrieval**: BGE-small-en-v1.5 embeddings with cosine similarity search
- **Citation verification**: Post-hoc verification removes unnecessary citations by testing whether the answer survives without each cited file
- **Defensive corpus walking**: Handles empty directories, unreadable files, encrypted PDFs, and unknown file types without crashing

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Container (CMD: daemon + sleep infinity)                            │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  DAEMON PROCESS (long-running, holds LLM)                    │    │
│  │  - Qwen2.5-7B-Instruct (bf16, ~14 GiB VRAM)                  │    │
│  │  - BGE-small embedder (CPU, ~130 MB)                         │    │
│  │  - Listens on /app/daemon.sock                               │    │
│  │  - Handles: INDEX, QUERY commands                            │    │
│  │  - During INDEX: loads VLM temporarily for image extraction  │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                          ↑ unix socket                                │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  app.py (thin client, invoked per --index/--query)           │    │
│  │  - Connects to daemon, sends command, writes output JSON     │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  /app/index/          ← persisted during --index                     │
│    chunks.json          (text chunks + file paths + metadata)        │
│    embeddings.npy       (BGE embedding vectors, 384-dim)             │
│  /models/              ← baked weights                               │
│    Qwen2.5-7B-Instruct/      (LLM for queries, ~14 GiB)              │
│    Qwen2.5-VL-3B-Instruct/   (VLM for image indexing, ~6 GiB)        │
│    bge-small-en-v1.5/        (embeddings, ~130 MB)                   │
│  /app/corpus/          ← grader copies corpus here                   │
│  /app/output/          ← query answers land here                     │
└──────────────────────────────────────────────────────────────────────┘
```

### Why a daemon?

The harness starts a new process for each of the 10 questions. Loading a 7B model takes ~30s — 10 questions × 30s = 300s, blowing the 30-second per-question budget. The daemon holds the model resident across all queries, so each query only pays the retrieval + generation cost (~5-10s).

### Indexing pipeline (during `--index`, charged to 10-min startup budget)

1. Walk corpus defensively (per-file try/except — one bad file doesn't stop the walk)
2. Parse each file by type:
   - `.pdf` → PyMuPDF (handles multi-page, detects WITHDRAWN/SUPERSEDED markers)
   - `.docx` → python-docx (extracts paragraphs + table cells)
   - `.xlsx` → openpyxl (multi-sheet, each sheet's rows serialized as tab-separated text)
   - `.csv` → csv module (rows as tab-separated lines)
   - `.txt` / `.log` / `.py` → plain text read
   - `.png` / `.jpg` → **VLM** (Qwen2.5-VL-3B extracts all printed text from the image)
   - Unknown / unreadable / encrypted → skip + log
3. Chunk text (paragraphs ≤500 chars, rows as individual chunks, images as single chunks)
4. Embed chunks with BGE-small (384-dim, L2-normalized)
5. Persist chunks + embeddings to `/app/index/`

### Query pipeline (in daemon, 30s budget)

1. Embed query with BGE-small (query-prefixed for retrieval)
2. Retrieve top-8 chunks by cosine similarity
3. Build prompt with retrieved chunks + file paths + withdrawal markers
4. LLM generates JSON: `{"answer": "...", "citations": [...], "confidence": ...}`
5. **Citation verification**: For each cited file, remove its chunks, re-run retrieval + generation. If the answer changes, the citation is necessary; if not, remove it.
6. Return final answer + verified citations

---

## Repository layout

```
optiseek-rag/
├── Dockerfile                    # ROCm base, bakes 3 models, CMD starts daemon
├── README.md
├── LICENSE
├── app/
│   ├── __init__.py
│   ├── app.py                    # Thin client (connects to daemon via socket)
│   ├── daemon.py                 # Long-running daemon (holds LLM, handles commands)
│   ├── requirements.txt
│   ├── parsers/
│   │   ├── registry.py           # Dispatch by extension, defensive walk
│   │   ├── pdf_parser.py         # PyMuPDF, detects superseded docs
│   │   ├── docx_parser.py        # python-docx, paragraphs + tables
│   │   ├── xlsx_parser.py        # openpyxl, multi-sheet
│   │   ├── csv_parser.py         # csv module
│   │   ├── text_parser.py        # .txt, .log, .py
│   │   └── image_parser.py       # VLM text extraction from images
│   ├── index/
│   │   ├── chunker.py            # Text chunking by type
│   │   ├── embedder.py           # BGE-small embeddings (CPU)
│   │   └── store.py              # Persist/load index to disk
│   ├── retrieval/
│   │   └── retriever.py          # Brute-force cosine similarity
│   ├── generation/
│   │   ├── llm.py                # Qwen2.5-7B-Instruct (bf16, GPU)
│   │   ├── vlm.py                # Qwen2.5-VL-3B-Instruct (bf16, GPU, indexing only)
│   │   └── prompts.py            # Answer + citation prompts
│   └── citations/
│       └── verifier.py           # Post-hoc citation necessity verification
├── test/
│   └── run_selfcheck.sh          # Run the official selfcheck.py
└── scripts/
    └── profile_vram.sh           # VRAM usage monitor
```

---

## Hardware and resource usage

| Resource | Used | Limit | Margin |
|---|---|---|---|
| GPU VRAM (query) | ~14 GiB (LLM only) | 48 GiB | 34 GiB headroom |
| GPU VRAM (index) | ~20 GiB (LLM + VLM) | 48 GiB | 28 GiB headroom |
| Docker image size | ~50 GiB | 60 GiB | 10 GiB headroom |
| Startup (incl. --index) | ~3-5 min | 10 min | 5-7 min margin |
| Per-query | ~8-15s (no model reload) | 30s | 15-22s margin |
| Total 10 queries | ~2-3 min | 10 min | 7-8 min margin |

### Image size breakdown

| Component | Size |
|---|---|
| ROCm base image | ~29 GiB |
| Qwen2.5-7B-Instruct (bf16) | ~14 GiB |
| Qwen2.5-VL-3B-Instruct (bf16) | ~6 GiB |
| BGE-small-en-v1.5 | ~0.1 GiB |
| Python deps | ~1 GiB |
| **Total** | **~50 GiB** |

---

## Building the Docker image

The image can be built on any machine with Docker. No GPU is needed for the build.

```bash
git clone https://github.com/b3panda3/optiseek-rag.git
cd optiseek-rag
docker build -t optiseek-rag:latest .
```

Build downloads three models from HuggingFace Hub (~20 GiB total). Build time: ~20-30 minutes.

### Verify the build

```bash
# Check image size (must be < 60 GiB)
docker image inspect optiseek-rag:latest --format '{{.Size}}' | awk '{printf "%.1f GiB\n", $1/1024/1024/1024}'

# Verify base layer
docker history optiseek-rag:latest --no-trunc | tail -5 | head -2

# Run the official selfcheck
python3 selfcheck.py optiseek-rag:latest mc3-corpus
```

---

## Running locally (without Docker)

### Prerequisites

- AMD GPU with ROCm 6.x+
- Python 3.10+

```bash
# Install deps (torch constraint prevents ROCm→CUDA replacement)
pip install -r app/requirements.txt

# Download models
python3 -c "
from huggingface_hub import snapshot_download
for model in ['Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-VL-3B-Instruct', 'BAAI/bge-small-en-v1.5']:
    snapshot_download(model, local_dir='./models/' + model.split('/')[-1])
"

# Start the daemon
export OPTISEEK_RAG_SOCKET=/tmp/daemon.sock
export OPTISEEK_RAG_INDEX_DIR=/tmp/index
python3 app/daemon.py &

# Index a corpus
python3 app/app.py --index /path/to/corpus

# Ask a question
python3 app/app.py --corpus /path/to/corpus --query-id q1 --query "What is the maximum junction temperature?"
cat /tmp/output/q1_output.json
```

---

## Key engineering decisions

### Why a daemon + unix socket?

The harness starts a new process for each of the 10 questions. Loading a 7B model takes ~30s. Without a daemon, each question pays the 30s load cost — blowing the 30s per-question budget on every question. The daemon holds the model resident; each query only pays retrieval + generation (~8-15s).

### Why Qwen2.5-7B instead of 14B?

The 14B model in bf16 is 28 GiB. Image math: 29 (base) + 28 (LLM) + 6 (VLM) = 63 GiB — over the 60 GiB limit. 4-bit quantization at load time still ships 28 GiB of bf16 weights. Pre-quantizing at build time requires a GPU (not available on GitHub Actions). AWQ/GPTQ pre-quantized models may not work on ROCm Py3.14. The 7B in bf16 (14 GiB) fits comfortably at ~50 GiB total and is strong enough for RAG reasoning.

### Why VLM during indexing only?

The VLM (Qwen2.5-VL-3B, ~6 GiB) is only needed to extract text from images. After indexing, all image content is text in the index. Loading the VLM only during indexing saves 6 GiB of VRAM for queries and reduces the model load time.

### Why BGE-small + numpy?

BGE-small-en-v1.5 is a 33M parameter model (~130 MB) that produces 384-dim embeddings. For our corpus size (~50-200 chunks), brute-force cosine similarity on numpy is faster than FAISS and has no dependencies. The embedder runs on CPU, freeing GPU for the LLM.

### Why post-hoc citation verification?

LLMs tend to over-cite — returning all retrieved files instead of only necessary ones. The challenge scores citations as an exact set: over-citing fails just as hard as under-citing. The verifier tests each citation by removing it and re-running the pipeline: if the answer survives, the citation is unnecessary and is removed. This costs N+1 LLM calls (1 original + N verifications) but catches over-citation reliably.

### Why defensive corpus walking?

The corpus deliberately contains four traps: empty directories, unreadable files (chmod 000), encrypted PDFs, and unknown file types. A walk that raises on the first bad file indexes nothing after it — and which files you lose depends on filename order. The walker catches per-file errors and continues, ensuring the entire readable corpus is indexed.

---

## The caching_allocator_warmup patch

Same fix as Optiseek MC2: transformers 5.x calls `caching_allocator_warmup()` which pre-allocates a buffer the size of the model before loading. On MI300X VF partitions, this fails with OOM. The daemon monkey-patches this to a no-op at import time, before any model loading occurs.

---

## License

MIT License. See `LICENSE`.

## Acknowledgments

- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) by Alibaba
- [BGE](https://github.com/FlagOpen/FlagEmbedding) by BAAI
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) for PDF parsing
- [AMD ROCm](https://github.com/ROCm/ROCm) for the GPU stack
- The LabLab + AMD AI Academy team for the challenge
