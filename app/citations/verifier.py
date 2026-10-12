"""verifier.py — Post-hoc citation necessity verification.

After the LLM generates an answer + citations, this module verifies each
citation by testing: "If I remove this file's chunks from the context, can
I still derive the same answer?"

If yes → the citation is unnecessary → remove it.
If no (answer changes or becomes empty) → the citation is necessary → keep it.

This is critical because LLMs tend to over-cite (returning all retrieved
files instead of only necessary ones). The challenge scores citations as
an EXACT SET — over-citing fails just as hard as under-citing.

Cost: N+1 LLM calls per query (1 original + N verifications, where N is
the number of citations). With 7B at ~3s per call and 1-3 citations, this
is 6-12s total — within the 30s budget.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata

log = logging.getLogger("optiseek.rag.citations")

# Same normalization as the grader
_NORMALIZE_REMOVE_CHARS = set("-.\u00b7_")


def _normalize(text: str) -> str:
    """Apply grader normalization: uppercase, strip whitespace, strip - . · _"""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.upper()
    text = re.sub(r"\s+", "", text)
    text = "".join(c for c in text if c not in _NORMALIZE_REMOVE_CHARS)
    return text


class CitationVerifier:
    """Verify citation necessity by removing each cited file and re-running."""

    def __init__(self, retriever, embedder, llm):
        """
        Args:
            retriever: Retriever instance with the full index.
            embedder: Embedder instance for query embedding.
            llm: LLMEngine instance for re-generation.
        """
        self.retriever = retriever
        self.embedder = embedder
        self.llm = llm

    def verify(self, query: str, answer: str, citations: list[str]) -> list[str]:
        """Remove unnecessary citations. Returns the filtered citation list.

        For each cited file:
        1. Retrieve chunks EXCLUDING that file
        2. Re-run the LLM with the reduced context
        3. If the answer changes → citation is necessary (keep)
        4. If the answer is the same → citation is unnecessary (remove)
        """
        if not citations:
            return citations

        necessary: list[str] = []

        for cited_file in citations:
            # Get chunks excluding this file
            query_emb = self.embedder.embed_query(query)
            all_chunks = self.retriever.chunks
            reduced_chunks = [c for c in all_chunks if c["file"] != cited_file]

            if not reduced_chunks:
                # No chunks left — definitely necessary
                necessary.append(cited_file)
                continue

            # Retrieve top-k from reduced set
            # We need a temporary retriever — use the same embeddings but filter
            from retrieval.retriever import Retriever
            import numpy as np

            # Get embeddings for reduced chunks
            excluded_indices = set(
                self.retriever.get_chunks_for_file(cited_file)
            )
            kept_indices = [i for i in range(len(all_chunks))
                           if i not in excluded_indices]

            if not kept_indices:
                necessary.append(cited_file)
                continue

            reduced_embeddings = self.retriever.embeddings[kept_indices]
            reduced_chunks_list = [all_chunks[i] for i in kept_indices]

            temp_retriever = Retriever(reduced_chunks_list, reduced_embeddings)
            top_chunks = temp_retriever.retrieve(query_emb, k=8)

            # Re-run LLM
            from generation.prompts import build_answer_prompt
            prompt = build_answer_prompt(query, top_chunks)
            raw = self.llm.generate(prompt, max_new_tokens=128)

            # Parse the response
            new_answer = _parse_answer(raw)
            new_answer_norm = _normalize(new_answer)
            original_norm = _normalize(answer)

            if new_answer_norm != original_norm:
                # Answer changed → this file is necessary
                necessary.append(cited_file)
                log.info("Citation %s is NECESSARY (answer changed without it)",
                        cited_file)
            else:
                log.info("Citation %s is UNNECESSARY (answer unchanged without it)",
                        cited_file)

        return necessary


def _parse_answer(raw: str) -> str:
    """Extract the answer field from an LLM JSON response."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw,
                    flags=re.MULTILINE).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        return ""
    try:
        data = json.loads(cleaned[start:end + 1])
        return str(data.get("answer", "")).strip()
    except (json.JSONDecodeError, ValueError):
        return ""
