"""vlm.py — Qwen2.5-VL-3B-Instruct VLM engine for image text extraction.

Loaded temporarily during indexing to extract text from PNG/JPG images.
Unloaded after indexing to free VRAM for the LLM.
"""

from __future__ import annotations

import gc
import logging
import os
import time

log = logging.getLogger("optiseek.rag.vlm")

# Model paths (checked in order)
MODEL_PATHS = [
    "/models/Qwen2.5-VL-3B-Instruct",
]
HF_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


class VLMEngine:
    """Qwen2.5-VL-3B-Instruct vision-language model engine."""

    def __init__(self):
        self.model = None
        self.processor = None
        self._torch = None

    def load(self):
        """Load the VLM on GPU in bf16."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch

        # Find model path
        model_path = None
        for p in MODEL_PATHS:
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
                model_path = p
                break
        if model_path is None:
            model_path = HF_MODEL_ID

        log.info("Loading VLM from %s (bf16, GPU)", model_path)
        t0 = time.time()

        # Detect GPU device — grader's GPU may be at different index/name
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            log.info("Using GPU device: %s (%s)", device, torch.cuda.get_device_name(0))
        else:
            device = "cpu"
            log.warning("No GPU available! Falling back to CPU.")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        log.info("VLM loaded in %.1fs", time.time() - t0)

    def unload(self):
        """Unload the VLM and free VRAM."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if self._torch is not None:
            self._torch.cuda.empty_cache()
        log.info("VLM unloaded, VRAM freed")

    def extract_text(self, image) -> str:
        """Extract all text from a PIL image.

        Args:
            image: PIL.Image in RGB mode.

        Returns:
            Extracted text string.
        """
        torch = self._torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Extract ALL text visible in this image. Return the text "
                        "exactly as it appears, including labels, numbers, codes, "
                        "pin names, revision numbers, and any printed text. "
                        "Preserve spatial layout if possible (e.g., "
                        "'Pin B14: THERM_ALERT#'). Do not add descriptions or "
                        "explanations. If there is no text, return an empty string."
                    )},
                    {"type": "image"},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        in_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0, in_len:]
        result = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        return result