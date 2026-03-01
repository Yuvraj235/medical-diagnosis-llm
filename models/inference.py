"""
models/inference.py
Medical LLM inference — BioGPT base model or LoRA fine-tuned adapter.

Flow: question + retrieved context → prompt → BioGPT generate → answer + decision label
"""
import sys
import os
import re
import logging
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BASE_LLM_NAME, FALLBACK_LLM_NAME, MODEL_DIR,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, DO_SAMPLE, REPETITION_PENALTY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LORA_DIR = os.path.join(MODEL_DIR, "lora_best")


class MedicalLLM:
    """BioGPT-based medical QA model (base or LoRA-adapted)."""

    def __init__(self):
        self._model     = None
        self._tokenizer = None
        self._device    = None
        self._model_name: str = ""

    # ── Status ────────────────────────────────────────────────────────────

    def is_loaded(self) -> bool:
        return self._model is not None

    def _device_str(self) -> str:
        import torch
        if torch.backends.mps.is_available(): return "mps"
        if torch.cuda.is_available():         return "cuda"
        return "cpu"

    # ── Loading ───────────────────────────────────────────────────────────

    def load(self):
        if self.is_loaded():
            return
        import torch
        self._device = self._device_str()
        logger.info(f"Loading Medical LLM on {self._device}…")

        lora_ready = (
            os.path.isdir(LORA_DIR)
            and os.path.exists(os.path.join(LORA_DIR, "adapter_config.json"))
        )
        if lora_ready:
            logger.info(f"✅ LoRA adapter found → {LORA_DIR}")
            self._load_lora(torch)
        else:
            logger.info("No LoRA adapter found — loading base model")
            self._load_base(torch)

        self._model.eval()
        logger.info(f"✅ LLM ready: {self._model_name}")

    def _load_base(self, torch):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        for name in [BASE_LLM_NAME, FALLBACK_LLM_NAME]:
            try:
                logger.info(f"  Trying {name}…")
                dtype = torch.float16 if self._device in ("cuda", "mps") else torch.float32
                self._tokenizer = AutoTokenizer.from_pretrained(name)
                self._model     = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
                self._model     = self._model.to(self._device)
                self._model_name = name
                return
            except Exception as e:
                logger.warning(f"  Could not load {name}: {e}")
        raise RuntimeError("Could not load any LLM. Check internet connection.")

    def _load_lora(self, torch):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        base_file = os.path.join(LORA_DIR, "base_model.txt")
        base_name = open(base_file).read().strip() if os.path.exists(base_file) else BASE_LLM_NAME

        try:
            dtype           = torch.float16 if self._device in ("cuda", "mps") else torch.float32
            self._tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
            base            = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype)
            self._model     = PeftModel.from_pretrained(base, LORA_DIR)
            self._model     = self._model.to(self._device)
            self._model_name = f"{base_name} + LoRA"
        except Exception as e:
            logger.warning(f"LoRA load failed: {e}. Falling back to base model.")
            self._load_base(torch)

    # ── Generation ────────────────────────────────────────────────────────

    def generate(self, question: str, context_str: str) -> Tuple[str, str]:
        """
        Generate a medical answer.
        Returns: (raw_answer, prompt_used)
        """
        if not self.is_loaded():
            self.load()

        import torch
        prompt = self._build_prompt(question, context_str)

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self._device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE if DO_SAMPLE else 1.0,
                top_p=TOP_P if DO_SAMPLE else 1.0,
                do_sample=DO_SAMPLE,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        n_input    = inputs["input_ids"].shape[1]
        new_tokens = out[0][n_input:]
        raw_answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return raw_answer, prompt

    @staticmethod
    def _build_prompt(question: str, context_str: str) -> str:
        ctx = context_str[:2000] + "…" if len(context_str) > 2000 else context_str
        return (
            "### Instruction: Answer the medical question using the provided biomedical evidence. "
            "Begin your answer with 'yes', 'no', or 'maybe'.\n\n"
            f"### Evidence:\n{ctx}\n\n"
            f"### Question:\n{question}\n\n"
            "### Answer:"
        )

    @staticmethod
    def extract_decision(answer: str) -> str:
        """Extract yes / no / maybe from generated answer text."""
        if not answer:
            return "maybe"
        text = answer.lower().strip()[:150]
        if re.search(r'\b(yes|affirmative|positive|correct|indeed)\b', text):
            return "yes"
        if re.search(r'\b(no|negative|not|false|incorrect)\b', text):
            return "no"
        return "maybe"
