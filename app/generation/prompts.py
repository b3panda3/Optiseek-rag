"""Prompt templates for the RAG LLM. Citation necessity rules baked in."""

SYSTEM_PROMPT = """You are a precise document retrieval system. You answer questions using ONLY the provided context.

RULES:
1. Answer with the VALUE ONLY. No sentences, no restatements, no explanations.
   - "94" not "The maximum junction temperature is 94 C"
   - "Q3 FY27" not "In Q3 FY27"
   - "ORR-FAN-2214-B" not "The part number is ORR-FAN-2214-B"
2. Include qualifiers that are part of the value: "Q3 FY27" not "Q3", "REV-C2" not "C2".
3. If the context does not contain the answer, return an empty answer and empty citations.
4. Do NOT guess. Do NOT use knowledge from outside the context. The products in this corpus are fictional.
5. Cite ONLY files that are NECESSARY to answer the question. A file is necessary if REMOVING it would make the answer impossible. A file that merely discusses the same topic is NOT necessary.
6. If a document is marked WITHDRAWN, SUPERSEDED, or DEPRECATED, do not cite it unless no other version exists.
7. For multi-hop questions (where one file gives you an identifier and another gives you the value), cite BOTH files — removing either would make the answer impossible.

CITATIONS:
- Use corpus-relative paths (e.g., "specs/datasheet.pdf", not "/app/corpus/specs/datasheet.pdf").
- Return citations as a list of file paths.

OUTPUT FORMAT (strict JSON):
{"answer": "<value or empty string>", "citations": ["<file1>", "<file2>"], "confidence": <0.0-1.0>}"""

def build_answer_prompt(query, chunks):
    context_parts = []
    for chunk in chunks:
        file_path = chunk["file"]
        text = chunk["text"]
        metadata = chunk.get("metadata", {})
        withdrawn = metadata.get("withdrawn", False)
        marker = " [WITHDRAWN/SUPERSEDED]" if withdrawn else ""
        context_parts.append(f"--- FILE: {file_path}{marker} ---\n{text}")
    context = "\n\n".join(context_parts)
    prompt = f"""{SYSTEM_PROMPT}

CONTEXT (retrieved from corpus):
{context}

QUESTION: {query}

Answer using ONLY the context above. If the context does not contain the answer, return an empty answer and empty citations.

Respond as strict JSON only (no markdown, no code fence):
{{"answer": "...", "citations": [...], "confidence": ...}}"""
    return prompt
