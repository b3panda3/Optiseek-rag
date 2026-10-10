"""Extract text from images using VLM. Only way to read text-in-image."""
import logging
from pathlib import Path
from PIL import Image
log = logging.getLogger("optiseek.rag.parsers.image")

def parse_image(path, corpus_root, vlm=None):
    rel_path = str(path.relative_to(corpus_root))
    if vlm is None:
        log.warning("No VLM available for image: %s", rel_path); return None
    try:
        img = Image.open(path); img.load()
        if img.mode != "RGB": img = img.convert("RGB")
    except Exception as e:
        log.warning("Image open failed: %s: %s", rel_path, e); return None
    try:
        extracted_text = vlm.extract_text(img)
    except Exception as e:
        log.warning("VLM extraction failed: %s: %s", rel_path, e); return None
    if not extracted_text or not extracted_text.strip():
        log.info("No text extracted from image: %s", rel_path); return None
    return [{"file": rel_path, "text": f"[Image text from {rel_path}]\n{extracted_text.strip()}", "metadata": {"type":"image","ext":path.suffix.lower(),"size":img.size}}]
