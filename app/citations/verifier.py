"""Post-hoc citation necessity verification.

Removes unnecessary citations by testing: if I remove this file's chunks,
can I still derive the same answer? If yes, the citation is unnecessary.
"""
import json, logging, re, unicodedata
log = logging.getLogger("optiseek.rag.citations")
_NORMALIZE_REMOVE_CHARS = set("-.\u00b7_")

def _normalize(text):
    if not text: return ""
    text = unicodedata.normalize("NFC", text)
    text = text.upper()
    text = re.sub(r"\s+", "", text)
    text = "".join(c for c in text if c not in _NORMALIZE_REMOVE_CHARS)
    return text

class CitationVerifier:
    def __init__(self, retriever, embedder, llm):
        self.retriever = retriever
        self.embedder = embedder
        self.llm = llm

    def verify(self, query, answer, citations):
        if not citations: return citations
        necessary = []
        for cited_file in citations:
            query_emb = self.embedder.embed_query(query)
            all_chunks = self.retriever.chunks
            excluded_indices = set(self.retriever.get_chunks_for_file(cited_file))
            kept_indices = [i for i in range(len(all_chunks)) if i not in excluded_indices]
            if not kept_indices:
                necessary.append(cited_file); continue
            reduced_embeddings = self.retriever.embeddings[kept_indices]
            reduced_chunks_list = [all_chunks[i] for i in kept_indices]
            from retrieval.retriever import Retriever
            temp_retriever = Retriever(reduced_chunks_list, reduced_embeddings)
            top_chunks = temp_retriever.retrieve_hybrid(query_emb, query, k=8)
            from generation.prompts import build_answer_prompt
            prompt = build_answer_prompt(query, top_chunks)
            raw = self.llm.generate(prompt, max_new_tokens=128)
            new_answer = _parse_answer(raw)
            new_answer_norm = _normalize(new_answer)
            original_norm = _normalize(answer)
            if new_answer_norm != original_norm:
                necessary.append(cited_file)
                log.info("Citation %s is NECESSARY (answer changed without it)", cited_file)
            else:
                log.info("Citation %s is UNNECESSARY (answer unchanged without it)", cited_file)
        return necessary

def _parse_answer(raw):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    start = cleaned.find("{"); end = cleaned.rfind("}")
    if start == -1 or end == -1: return ""
    try:
        data = json.loads(cleaned[start:end+1])
        return str(data.get("answer", "")).strip()
    except (json.JSONDecodeError, ValueError):
        return ""
