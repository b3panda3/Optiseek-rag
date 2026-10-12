"""retriever.py — Cosine similarity retrieval over chunks.

Uses brute-force cosine similarity (numpy dot product on L2-normalized
vectors). For our corpus size (~50-200 chunks), this is faster than
FAISS and has no dependencies.
"""

from __future__ import annotations

import numpy as np


class Retriever:
    """Brute-force cosine similarity retriever."""

    def __init__(self, chunks: list[dict], embeddings: np.ndarray):
        """Initialize with chunks and their embeddings.

        Args:
            chunks: List of chunk dicts with "file", "text", "metadata".
            embeddings: (N, D) numpy array, L2-normalized.
        """
        self.chunks = chunks
        self.embeddings = embeddings
        # Build a file → chunk indices map for citation verification
        self._file_to_indices: dict[str, list[int]] = {}
        for i, chunk in enumerate(chunks):
            f = chunk["file"]
            if f not in self._file_to_indices:
                self._file_to_indices[f] = []
            self._file_to_indices[f].append(i)

    def retrieve(self, query_embedding: np.ndarray, k: int = 8) -> list[dict]:
        """Find the top-k most similar chunks to the query.

        Args:
            query_embedding: (D,) numpy array, L2-normalized.
            k: Number of chunks to return.

        Returns:
            List of top-k chunk dicts, sorted by similarity (descending).
            Each chunk has an added "score" key.
        """
        # Cosine similarity = dot product (since both are L2-normalized)
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results

    def get_chunks_for_file(self, file_path: str) -> list[int]:
        """Return chunk indices for a given file path."""
        return self._file_to_indices.get(file_path, [])

    def get_chunks_excluding_file(self, file_path: str) -> list[dict]:
        """Return all chunks NOT from the given file. Used for citation verification."""
        excluded = set(self.get_chunks_for_file(file_path))
        return [self.chunks[i] for i in range(len(self.chunks)) if i not in excluded]
