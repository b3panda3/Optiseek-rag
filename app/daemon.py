#!/usr/bin/env python3
"""daemon.py — Long-running model server for Optiseek RAG.

Started by the container's CMD. Loads the LLM once at startup, then listens
on a unix socket for INDEX and QUERY commands from app.py.

WHY A DAEMON:
    The harness starts a new process for each of the 10 questions. Loading a
    7B model takes ~30s. 10 questions × 30s = 300s — blows the 30s per-question
    budget on every question. The daemon holds the model resident across all
    queries, so each query only pays the retrieval + generation cost (~5-10s).

PROTOCOL:
    JSON over unix socket. One request per connection, one response per request.
    Commands:
        {"cmd": "index", "corpus": "/app/corpus"}
        {"cmd": "query", "corpus": "/app/corpus", "query": "..."}
    Responses:
        {"status": "ok", "files_indexed": N, "chunks_indexed": N}
        {"answer": "...", "citations": [...], "confidence": 0.9}
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path

# Add app to path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Patch out caching_allocator_warmup BEFORE any model loading
# (same fix as MC2 — transformers 5.x OOMs on VF partitions without this)
import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *args, **kwargs: None

from parsers.registry import parse_corpus
from index.chunker import chunk_documents
from index.embedder import Embedder
from index.store import IndexStore
from retrieval.retriever import Retriever
from generation.llm import LLMEngine
from generation.vlm import VLMEngine
from generation.prompts import build_answer_prompt
from citations.verifier import CitationVerifier

log = logging.getLogger("optiseek.rag.daemon")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

SOCKET_PATH = os.environ.get("OPTISEEK_RAG_SOCKET", "/app/daemon.sock")
INDEX_DIR = Path(os.environ.get("OPTISEEK_RAG_INDEX_DIR", "/app/index"))


class RAGDaemon:
    """Holds the LLM, embedder, and index. Handles INDEX and QUERY commands."""

    def __init__(self):
        self.llm: LLMEngine | None = None
        self.vlm: VLMEngine | None = None
        self.embedder: Embedder | None = None
        self.retriever: Retriever | None = None
        self.verifier: CitationVerifier | None = None
        self.indexed = False

    def start(self):
        """Load models and start listening on the socket."""
        log.info("Starting Optiseek RAG daemon")

        # Diagnostic: log GPU + weights status
        try:
            import torch
            log.info("GPU available: %s", torch.cuda.is_available())
            if torch.cuda.is_available():
                log.info("GPU device: %s", torch.cuda.get_device_name(0))
                free, total = torch.cuda.mem_get_info()
                log.info("VRAM: %.1f GiB free / %.1f GiB total", free/1e9, total/1e9)
            else:
                log.warning("NO GPU AVAILABLE — will fail VRAM floor gate")
        except Exception as e:
            log.warning("Could not check GPU status: %s", e)

        # Check baked-in weights
        for name in ["Qwen2.5-7B-Instruct", "Qwen2.5-VL-3B-Instruct", "bge-small-en-v1.5"]:
            p = Path(f"/models/{name}")
            if p.exists():
                log.info("Weights found: %s", p)
            else:
                log.error("Weights MISSING: %s — will fail at eval (no network)", p)

        # Load LLM (resident for all queries)
        self.llm = LLMEngine()
        self.llm.load()

        # Load embedder (resident — small and CPU-based)
        self.embedder = Embedder()
        self.embedder.load()

        # Start listening
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        server.listen(8)
        os.chmod(SOCKET_PATH, 0o666)
        log.info("Listening on %s", SOCKET_PATH)

        while True:
            try:
                conn, _ = server.accept()
                threading.Thread(target=self._handle, args=(conn,),
                                 daemon=True).start()
            except Exception as e:
                log.error("Accept error: %s", e)
                time.sleep(0.1)

    def _handle(self, conn: socket.socket):
        """Handle one client connection."""
        try:
            buf = b""
            while b"\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk

            command = json.loads(buf.decode("utf-8").strip())
            cmd = command.get("cmd")

            if cmd == "index":
                response = self._do_index(command)
            elif cmd == "query":
                response = self._do_query(command)
            else:
                response = {"status": "error", "error": f"unknown cmd: {cmd}"}

            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
        except Exception as e:
            log.exception("Handler error")
            try:
                err = {"status": "error", "error": str(e)}
                conn.sendall((json.dumps(err) + "\n").encode("utf-8"))
            except Exception:
                pass
        finally:
            conn.close()

    def _do_index(self, command: dict) -> dict:
        """Index the corpus: parse all files, chunk, embed, persist."""
        corpus = Path(command["corpus"])
        log.info("Indexing corpus at %s", corpus)

        # Load VLM temporarily for image text extraction
        self.vlm = VLMEngine()
        self.vlm.load()

        try:
            # Parse all files defensively
            documents = parse_corpus(corpus, vlm=self.vlm)
            log.info("Parsed %d documents", len(documents))

            # Chunk documents
            chunks = chunk_documents(documents)
            log.info("Created %d chunks", len(chunks))

            # Embed chunks
            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.embed(texts)
            log.info("Embedded %d chunks", len(embeddings))

            # Persist index
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            store = IndexStore(INDEX_DIR)
            store.save(chunks, embeddings)
            log.info("Index persisted to %s", INDEX_DIR)

            # Build retriever
            self.retriever = Retriever(chunks, embeddings)
            self.verifier = CitationVerifier(
                self.retriever, self.embedder, self.llm
            )
            self.indexed = True

            return {
                "status": "ok",
                "files_indexed": len(documents),
                "chunks_indexed": len(chunks),
            }
        finally:
            # Unload VLM to free VRAM for queries
            self.vlm.unload()
            self.vlm = None

    def _do_query(self, command: dict) -> dict:
        """Answer a single question."""
        query = command["query"]

        if not self.indexed or self.retriever is None:
            return {"answer": "", "citations": [], "confidence": 0.0,
                    "error": "not indexed"}

        # If index not loaded (e.g., daemon restarted), try to reload from disk
        if self.retriever is None:
            store = IndexStore(INDEX_DIR)
            if store.exists():
                chunks, embeddings = store.load()
                self.retriever = Retriever(chunks, embeddings)
                self.verifier = CitationVerifier(
                    self.retriever, self.embedder, self.llm
                )
            else:
                return {"answer": "", "citations": [], "confidence": 0.0,
                        "error": "no index"}

        # Retrieve top-k chunks
        query_emb = self.embedder.embed([query])[0]
        top_chunks = self.retriever.retrieve(query_emb, k=8)

        # Generate answer + citations
        prompt = build_answer_prompt(query, top_chunks)
        raw = self.llm.generate(prompt)

        # Parse LLM output
        answer, citations, confidence = _parse_llm_response(raw)

        if not answer:
            # Refusal — empty answer, empty citations
            return {"answer": "", "citations": [], "confidence": 0.0}

        # Verify citations (remove unnecessary ones)
        if citations and self.verifier:
            citations = self.verifier.verify(query, answer, citations)

        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
        }


def _parse_llm_response(raw: str) -> tuple[str, list[str], float]:
    """Parse the LLM's JSON response into (answer, citations, confidence)."""
    import re
    # Strip code fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    # Find JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        return "", [], 0.0

    try:
        data = json.loads(cleaned[start:end + 1])
        answer = str(data.get("answer", "")).strip()
        citations = [str(c) for c in data.get("citations", [])]
        confidence = float(data.get("confidence", 0.5))
        return answer, citations, confidence
    except (json.JSONDecodeError, ValueError):
        return "", [], 0.0


if __name__ == "__main__":
    daemon = RAGDaemon()
    daemon.start()