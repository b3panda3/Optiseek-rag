"""Qwen2.5-VL-3B-Instruct VLM. Loaded during indexing, unloaded after."""
import gc, logging, os, time
log = logging.getLogger("optiseek.rag.vlm")
MODEL_PATHS = ["/models/Qwen2.5-VL-3B-Instruct", "/persistent/models/Qwen2.5-VL-3B-Instruct"]
HF_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

class VLMEngine:
    def __init__(self):
        self.model = None; self.processor = None; self._torch = None

    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self._torch = torch
        model_path = None
        for p in MODEL_PATHS:
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
                model_path = p; break
        if model_path is None: model_path = HF_MODEL_ID
        log.info("Loading VLM from %s (bf16, GPU)", model_path)
        t0 = time.time()
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda:0",
            trust_remote_code=True, low_cpu_mem_usage=True)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        log.info("VLM loaded in %.1fs", time.time() - t0)

    def unload(self):
        del self.model; del self.processor
        self.model = None; self.processor = None
        gc.collect()
        if self._torch is not None: self._torch.cuda.empty_cache()
        log.info("VLM unloaded, VRAM freed")

    def extract_text(self, image):
        torch = self._torch
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Extract ALL text visible in this image. Return the text exactly as it appears, including labels, numbers, codes, pin names, revision numbers, and any printed text. Preserve spatial layout if possible (e.g., 'Pin B14: THERM_ALERT#'). Do not add descriptions or explanations. If there is no text, return an empty string."},
            {"type": "image"}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.model.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        in_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[0, in_len:]
        return self.processor.decode(gen_ids, skip_special_tokens=True).strip()
