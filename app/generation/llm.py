"""llm.py — Qwen2.5-7B-Instruct LLM engine for answer generation.

Loaded once at daemon startup, resident across all queries. Uses bf16 on
GPU. The caching_allocator_warmup patch (applied in daemon.py) ensures
reliable loading on MI300X VF partitions.
"""

from __future__ import annotations

import logging
import os
import time

log = logging.getLogger("optiseek.rag.llm")

# Model paths (checked in order)
MODEL_PATHS = [
    "/models/Qwen2.5-7B-Instruct",
]
HF_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


class LLMEngine:
    """Qwen2.5-7B-Instruct text generation engine."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._torch = None

    def load(self):
        """Load the model on GPU in bf16."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch

        # Find model path
        model_path = None
        for p in MODEL_PATHS:
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
                model_path = p
                break
        if model_path is None:
            model_path = HF_MODEL_ID

        log.info("Loading LLM from %s (bf16, GPU)", model_path)
        t0 = time.time()

        # Detect GPU device — grader's GPU may be at different index/name
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            log.info("Using GPU device: %s (%s)", device, torch.cuda.get_device_name(0))
        else:
            device = "cpu"
            log.warning("No GPU available! Falling back to CPU.")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        log.info("LLM loaded in %.1fs", time.time() - t0)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate text from a prompt. Returns the generated text (no prompt)."""
        torch = self._torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic for RAG
                temperature=1.0,  # irrelevant with do_sample=False
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        in_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0, in_len:]
        result = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return result