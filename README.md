# Optiseek RAG

Retrieval-Augmented Generation over mixed-document corpora, built for the AMD LabLab AI Academy Mini Challenge 3.

> **Hackathon**: [AMD LabLab AI Academy Challenge — Mini Challenge 3 (RAG)](https://lablab.ai/ai-hackathons/amd-lablab-ai-academy-challenge)

## What it does

Indexes a corpus of mixed file types (PDF, DOCX, XLSX, CSV, TXT, LOG, PY, PNG, JPG), then answers questions with precise values and exact citations.

## Architecture

- **Daemon + unix socket**: Holds Qwen2.5-7B-Instruct resident; app.py is a thin client
- **Multi-format parsers**: PyMuPDF, python-docx, openpyxl, csv, VLM for images
- **BGE-small embeddings**: 384-dim, CPU-based, brute-force cosine similarity
- **Citation verification**: Post-hoc test removes unnecessary citations

## Key files

- `app/app.py` — Thin client (connects to daemon via socket)
- `app/daemon.py` — Long-running daemon (holds LLM, handles INDEX + QUERY)
- `app/parsers/` — Format-specific extractors
- `app/index/` — Chunker, embedder, store
- `app/retrieval/` — Cosine similarity retriever
- `app/generation/` — LLM, VLM, prompts
- `app/citations/` — Post-hoc citation verifier

## License

MIT
