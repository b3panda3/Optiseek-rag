"""Qwen2.5-7B-Instruct LLM engine. Loaded once, resident in daemon."""
import logging, os, time
log = logging.getLogger("optiseek.rag.llm")
MODEL_PATHS = ["/models/Qwen2.5-7B-Instruct", "/persistent/models/Qwen2.5-7B-Instruct"]
HF_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

class LLMEngine:
    def __init__(self):
        self.model = None; self.tokenizer = None; self._torch = None

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        model_path = None
        for p in MODEL_PATHS:
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
                model_path = p; break
        if model_path is None: model_path = HF_MODEL_ID
        log.info("Loading LLM from %s (bf16, GPU)", model_path)
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda:0",
            trust_remote_code=True, low_cpu_mem_usage=True)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        log.info("LLM loaded in %.1fs", time.time() - t0)

    def generate(self, prompt, max_new_tokens=256):
        torch = self._torch
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        in_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0, in_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
