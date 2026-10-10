#!/usr/bin/env python3
"""Long-running daemon for Optiseek RAG. Holds LLM, handles INDEX + QUERY."""
from __future__ import annotations
import json, logging, os, socket, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

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
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)

SOCKET_PATH = os.environ.get("OPTISEEK_RAG_SOCKET", "/app/daemon.sock")
INDEX_DIR = Path(os.environ.get("OPTISEEK_RAG_INDEX_DIR", "/app/index"))

class RAGDaemon:
    def __init__(self):
        self.llm = None; self.vlm = None; self.embedder = None
        self.retriever = None; self.verifier = None; self.indexed = False

    def start(self):
        log.info("Starting Optiseek RAG daemon")
        self.llm = LLMEngine(); self.llm.load()
        self.embedder = Embedder(); self.embedder.load()
        if os.path.exists(SOCKET_PATH): os.unlink(SOCKET_PATH)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH); server.listen(8)
        os.chmod(SOCKET_PATH, 0o666)
        log.info("Listening on %s", SOCKET_PATH)
        while True:
            try:
                conn, _ = server.accept()
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
            except Exception as e:
                log.error("Accept error: %s", e); time.sleep(0.1)

    def _handle(self, conn):
        try:
            buf = b""
            while b"\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk: break
                buf += chunk
            command = json.loads(buf.decode("utf-8").strip())
            cmd = command.get("cmd")
            if cmd == "index": response = self._do_index(command)
            elif cmd == "query": response = self._do_query(command)
            else: response = {"status": "error", "error": f"unknown cmd: {cmd}"}
            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
        except Exception as e:
            log.exception("Handler error")
            try:
                conn.sendall((json.dumps({"status": "error", "error": str(e)}) + "\n").encode("utf-8"))
            except: pass
        finally:
            conn.close()

    def _do_index(self, command):
        corpus = Path(command["corpus"])
        log.info("Indexing corpus at %s", corpus)
        self.vlm = VLMEngine(); self.vlm.load()
        try:
            documents = parse_corpus(corpus, vlm=self.vlm)
            log.info("Parsed %d documents", len(documents))
            chunks = chunk_documents(documents)
            log.info("Created %d chunks", len(chunks))
            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.embed(texts)
            log.info("Embedded %d chunks", len(embeddings))
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            store = IndexStore(INDEX_DIR); store.save(chunks, embeddings)
            self.retriever = Retriever(chunks, embeddings)
            self.verifier = CitationVerifier(self.retriever, self.embedder, self.llm)
            self.indexed = True
            return {"status": "ok", "files_indexed": len(documents), "chunks_indexed": len(chunks)}
        finally:
            self.vlm.unload(); self.vlm = None

    def _do_query(self, command):
        query = command["query"]
        if not self.indexed or self.retriever is None:
            if self.retriever is None:
                store = IndexStore(INDEX_DIR)
                if store.exists():
                    chunks, embeddings = store.load()
                    self.retriever = Retriever(chunks, embeddings)
                    self.verifier = CitationVerifier(self.retriever, self.embedder, self.llm)
                else:
                    return {"answer": "", "citations": [], "confidence": 0.0, "error": "no index"}
        query_emb = self.embedder.embed_query(query)
        top_chunks = self.retriever.retrieve_hybrid(query_emb, query, k=8)
        prompt = build_answer_prompt(query, top_chunks)
        raw = self.llm.generate(prompt)
        answer, citations, confidence = _parse_llm_response(raw)
        if not answer:
            return {"answer": "", "citations": [], "confidence": 0.0}
        if citations and self.verifier:
            citations = self.verifier.verify(query, answer, citations)
        return {"answer": answer, "citations": citations, "confidence": confidence}

def _parse_llm_response(raw):
    import re
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    start = cleaned.find("{"); end = cleaned.rfind("}")
    if start == -1 or end == -1: return "", [], 0.0
    try:
        data = json.loads(cleaned[start:end+1])
        return str(data.get("answer","")).strip(), [str(c) for c in data.get("citations",[])], float(data.get("confidence",0.5))
    except (json.JSONDecodeError, ValueError):
        return "", [], 0.0

if __name__ == "__main__":
    RAGDaemon().start()
