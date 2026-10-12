"""image_parser.py — Extract text from images using a VLM.

Uses Qwen2.5-VL-3B-Instruct to read text printed inside images (diagrams,
photographs of labels, pinout charts). This is the only way to answer
questions like "Which backplane pin carries THERM_ALERT#?" — the value
exists only in the image, not in any text file.

The VLM is loaded by the daemon during indexing, then unloaded to free VRAM
for queries. Image text is stored as a text chunk in the index, so queries
don't need the VLM at all.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image

log = logging.getLogger("optiseek.rag.parsers.image")

# Prompt: extract ALL text from the image, verbatim.
EXTRACTION_PROMPT = (
    "Extract ALL text visible in this image. Return the text exactly as it "
    "appears, including labels, numbers, codes, pin names, revision numbers, "
    "and any printed text. Preserve the spatial layout if possible (e.g., "
    "'Pin B14: THERM_ALERT#'). Do not add descriptions, explanations, or "
    "text that is not present in the image. If there is no text, return an "
    "empty string."
)


def parse_image(path: Path, corpus_root: Path, vlm=None) -> list[dict] | None:
    """Extract text from an image using the VLM.

    Args:
        path: Path to the image file.
        corpus_root: Root of the corpus (for relative path calculation).
        vlm: VLMEngine instance. If None, returns None (can't extract text).
    """
    rel_path = str(path.relative_to(corpus_root))

    if vlm is None:
        log.warning("No VLM available for image: %s", rel_path)
        return None

    # Load image with PIL
    try:
        img = Image.open(path)
        img.load()
        # Convert to RGB (drop alpha, palette, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")
    except Exception as e:
        log.warning("Image open failed: %s: %s", rel_path, e)
        return None

    # Run VLM extraction
    try:
        extracted_text = vlm.extract_text(img)
    except Exception as e:
        log.warning("VLM extraction failed: %s: %s", rel_path, e)
        return None

    if not extracted_text or not extracted_text.strip():
        log.info("No text extracted from image: %s", rel_path)
        return None

    return [{
        "file": rel_path,
        "text": f"[Image text from {rel_path}]\n{extracted_text.strip()}",
        "metadata": {
            "type": "image",
            "ext": path.suffix.lower(),
            "size": img.size,
        },
    }]
