"""Hybrid retrieval: semantic (BGE) + keyword (exact term match).

Semantic alone misses chunks where the wording differs slightly from the
query (e.g., 'thermal throttle engages' vs 'thermal throttle engaged').
Keyword retrieval catches these by matching exact terms.
"""

import re
import numpy as np

STOP_WORDS = {
    'what', 'is', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for',
    'which', 'when', 'how', 'are', 'was', 'were', 'been', 'has', 'have',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'must', 'can', 'this', 'that', 'these', 'those', 'it', 'its',
    'they', 'them', 'their', 'there', 'here', 'and', 'or', 'but', 'not',
    'no', 'yes', 'if', 'then', 'else', 'while', 'during', 'after', 'before',
}


class Retriever:
    """Hybrid cosine similarity + keyword retrieval."""

    def __init__(self, chunks, embeddings):
        self.chunks = chunks
        self.embeddings = embeddings
        self._file_to_indices = {}
        for i, chunk in enumerate(chunks):
            f = chunk["file"]
            if f not in self._file_to_indices:
                self._file_to_indices[f] = []
            self._file_to_indices[f].append(i)

    def retrieve(self, query_embedding, k=8):
        """Pure semantic retrieval."""
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results

    def retrieve_keyword(self, query, k=16):
        """Keyword retrieval. Returns chunks containing query terms."""
        terms = re.findall(r'\b[a-zA-Z][a-zA-Z0-9#_-]*\b', query.lower())
        terms = [t for t in terms if t not in STOP_WORDS and len(t) > 1]

        scored = []
        for i, chunk in enumerate(self.chunks):
            text_lower = chunk["text"].lower()
            score = sum(1 for term in terms if term in text_lower)
            if score > 0:
                scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, kw_score in scored[:k]:
            chunk = dict(self.chunks[idx])
            chunk["keyword_score"] = float(kw_score)
            results.append(chunk)
        return results

    def retrieve_hybrid(self, query_embedding, query_text, k=8):
        """Hybrid: semantic + keyword. Merges and re-ranks."""
        sem_results = self.retrieve(query_embedding, k=k * 2)
        kw_results = self.retrieve_keyword(query_text, k=k * 2)

        # Merge by (file, text[:100]) to deduplicate
        seen = {}
        merged = []

        for r in sem_results:
            key = (r["file"], r["text"][:100])
            if key not in seen:
                seen[key] = len(merged)
                merged.append(r)
            else:
                # Already present — shouldn't happen with semantic, but just in case
                pass

        for r in kw_results:
            key = (r["file"], r["text"][:100])
            if key in seen:
                # Boost existing entry's score
                idx = seen[key]
                merged[idx]["score"] = merged[idx].get("score", 0) + r.get("keyword_score", 0) * 0.15
            else:
                seen[key] = len(merged)
                r["score"] = r.get("keyword_score", 0) * 0.3  # keyword-only entries get lower base score
                merged.append(r)

        # Sort by combined score
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        return merged[:k]

    def get_chunks_for_file(self, file_path):
        return self._file_to_indices.get(file_path, [])

    def get_chunks_excluding_file(self, file_path):
        excluded = set(self.get_chunks_for_file(file_path))
        return [self.chunks[i] for i in range(len(self.chunks)) if i not in excluded]
