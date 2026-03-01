"""
models/lora_finetune.py
LoRA fine-tuning of BioGPT on PubMedQA expert-annotated data.

Run via:  python run.py finetune
"""
import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BASE_LLM_NAME, FALLBACK_LLM_NAME, MODEL_DIR, DATA_DIR,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, GRAD_ACCUM_STEPS,
    LEARNING_RATE, MAX_TRAIN_SAMPLES, MAX_SEQ_LENGTH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LORA_OUTPUT_DIR = os.path.join(MODEL_DIR, "lora_best")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device() -> str:
    import torch
    if torch.backends.mps.is_available():  return "mps"
    if torch.cuda.is_available():          return "cuda"
    return "cpu"


def load_training_data(max_samples: int = MAX_TRAIN_SAMPLES):
    train_path = os.path.join(DATA_DIR, "train.json")
    val_path   = os.path.join(DATA_DIR, "val.json")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found: {train_path}\n"
            "Run: python run.py setup"
        )
    with open(train_path) as f: train_data = json.load(f)
    val_data = []
    if os.path.exists(val_path):
        with open(val_path) as f: val_data = json.load(f)

    train_data = train_data[:max_samples]
    logger.info(f"Train: {len(train_data)}  Val: {len(val_data)}")
    return train_data, val_data


def format_prompt(record: Dict) -> str:
    ctx   = record.get("context", "")[:1500]
    q     = record.get("question", "")
    label = record.get("final_decision", "maybe")
    expl  = record.get("long_answer", "")[:500]
    return (
        "### Instruction: Answer the medical question based on the provided evidence.\n\n"
        f"### Evidence:\n{ctx}\n\n"
        f"### Question:\n{q}\n\n"
        f"### Answer: {label}\n"
        f"### Explanation: {expl}"
    )


def _find_target_modules(model) -> List[str]:
    """Return LoRA target modules that actually exist in the model."""
    import torch
    linear_names = {
        n.split(".")[-1]
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
    }
    targets = [m for m in LORA_TARGET_MODULES if m in linear_names]
    if not targets:
        # Common alternative names
        for alt in ["c_attn", "c_proj", "Wqkv", "query", "value", "key"]:
            if alt in linear_names:
                targets = [alt]
                break
    if not targets:
        targets = list(linear_names)[:4]
    logger.info(f"LoRA target modules: {targets}")
    return targets


# ── Main training function ────────────────────────────────────────────────────

def train():
    """LoRA fine-tune BioGPT on PubMedQA; saves adapter to models/checkpoints/lora_best/"""
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import Dataset

    os.makedirs(LORA_OUTPUT_DIR, exist_ok=True)
    device = get_device()
    logger.info(f"Training device: {device}")

    train_data, val_data = load_training_data()

    # ── Load base model ────────────────────────────────────────────────────
    model_name = None
    model, tokenizer = None, None
    for candidate in [BASE_LLM_NAME, FALLBACK_LLM_NAME]:
        try:
            logger.info(f"Loading base model: {candidate}")
            tokenizer = AutoTokenizer.from_pretrained(candidate)
            dtype     = torch.float16 if device in ("cuda", "mps") else torch.float32
            model     = AutoModelForCausalLM.from_pretrained(candidate, torch_dtype=dtype)
            model     = model.to(device)
            model_name = candidate
            logger.info(f"✅ Loaded {candidate}")
            break
        except Exception as e:
            logger.warning(f"Could not load {candidate}: {e}")

    if model is None:
        raise RuntimeError("Could not load any LLM. Check internet connection and model names.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Apply LoRA ─────────────────────────────────────────────────────────
    logger.info(f"Applying LoRA  r={LORA_R}  alpha={LORA_ALPHA}  dropout={LORA_DROPOUT}")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=_find_target_modules(model),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ────────────────────────────────────────────────────────────
    class PubMedDataset(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            prompt = format_prompt(self.records[idx])
            enc    = tokenizer(
                prompt,
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze()
            labels    = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            return {
                "input_ids":      input_ids,
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels":         labels,
            }

    train_ds = PubMedDataset(train_data)
    eval_ds  = PubMedDataset(val_data) if val_data else None

    # ── TrainingArguments ─────────────────────────────────────────────────
    use_fp16 = (device == "cuda")
    args = TrainingArguments(
        output_dir=LORA_OUTPUT_DIR,
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=use_fp16,
        logging_steps=10,
        eval_strategy="epoch" if eval_ds else "no",
        save_strategy="epoch",
        load_best_model_at_end=(eval_ds is not None),
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    logger.info(f"Starting training for {TRAIN_EPOCHS} epoch(s)…")
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    with open(os.path.join(LORA_OUTPUT_DIR, "base_model.txt"), "w") as f:
        f.write(model_name)

    logger.info(f"✅ LoRA fine-tuning complete! Adapter saved → {LORA_OUTPUT_DIR}")


if __name__ == "__main__":
    train()
