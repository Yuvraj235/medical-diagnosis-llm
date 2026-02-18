"""
LoRA Fine-tuning for BioMistral-7B
Uses QLoRA (4-bit quantization + Low-Rank Adaptation) for efficient fine-tuning.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd
import os

try:
    from src.data_loader import load_config, load_pubmedqa, create_train_val_test_split, format_for_training
except ImportError:
    from data_loader import load_config, load_pubmedqa, create_train_val_test_split, format_for_training


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_base_model(config: dict):
    """
    Load BioMistral-7B with 4-bit quantization (QLoRA).
    """
    llm_config = config["llm"]
    model_name = llm_config["base_model"]

    print(f"Loading base model: {model_name}")

    device = get_device()

    # Quantization config for QLoRA
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # For MPS/CPU: load in float16 without bitsandbytes quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded on device: {device}")
    print(f"Model parameters: {model.num_parameters():,}")

    return model, tokenizer


def apply_lora(model, config: dict):
    """Apply LoRA adapters to the model."""
    lora_config = config["lora"]

    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"],
        bias=lora_config["bias"],
        task_type=lora_config["task_type"],
    )

    # Prepare model for training
    if hasattr(model, "is_quantized") and model.is_quantized:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, peft_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"\nLoRA applied:")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable %:          {100 * trainable / total:.2f}%")

    return model


def prepare_dataset(df: pd.DataFrame, tokenizer, max_length: int = 1024) -> Dataset:
    """Convert DataFrame to HuggingFace Dataset for training."""
    texts = [format_for_training(row) for _, row in df.iterrows()]
    return Dataset.from_dict({"text": texts})


def train(config: dict = None):
    """
    Full fine-tuning pipeline:
    1. Load data
    2. Load model + apply LoRA
    3. Train with SFTTrainer
    4. Save adapter weights
    """
    if config is None:
        config = load_config()

    train_config = config["training"]

    # === Load Data ===
    print("=" * 60)
    print("Step 1: Loading data...")
    print("=" * 60)
    data = load_pubmedqa(config)

    # Use artificial data for training, labeled for validation
    train_df = data["artificial"]
    _, val_df, _ = create_train_val_test_split(data["labeled"], config)

    # Limit training size for faster iteration (remove for full training)
    max_train = min(len(train_df), 50000)
    train_df = train_df.sample(n=max_train, random_state=42)
    print(f"\nUsing {len(train_df)} training examples, {len(val_df)} validation examples")

    # === Load Model ===
    print("\n" + "=" * 60)
    print("Step 2: Loading model + LoRA...")
    print("=" * 60)
    model, tokenizer = load_base_model(config)
    model = apply_lora(model, config)

    # === Prepare Datasets ===
    print("\n" + "=" * 60)
    print("Step 3: Preparing datasets...")
    print("=" * 60)
    train_dataset = prepare_dataset(train_df, tokenizer, train_config["max_seq_length"])
    val_dataset = prepare_dataset(val_df, tokenizer, train_config["max_seq_length"])
    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Val dataset:   {len(val_dataset)} examples")

    # === Training Arguments ===
    output_dir = train_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_config["num_epochs"],
        per_device_train_batch_size=train_config["batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        warmup_ratio=train_config["warmup_ratio"],
        logging_steps=train_config["logging_steps"],
        save_steps=train_config["save_steps"],
        eval_steps=train_config["eval_steps"],
        eval_strategy="steps",
        save_total_limit=3,
        fp16=train_config["fp16"],
        bf16=train_config["bf16"],
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )

    # === SFT Trainer ===
    print("\n" + "=" * 60)
    print("Step 4: Starting training...")
    print("=" * 60)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        max_seq_length=train_config["max_seq_length"],
    )

    # Train
    trainer.train()

    # === Save ===
    print("\n" + "=" * 60)
    print("Step 5: Saving LoRA adapter...")
    print("=" * 60)

    adapter_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to: {adapter_path}")

    # Check adapter size
    adapter_size = sum(
        os.path.getsize(os.path.join(adapter_path, f))
        for f in os.listdir(adapter_path)
        if os.path.isfile(os.path.join(adapter_path, f))
    )
    print(f"Adapter size: {adapter_size / 1024 / 1024:.1f} MB")

    return model, tokenizer, trainer


def load_finetuned_model(config: dict = None):
    """Load the base model with fine-tuned LoRA adapter."""
    if config is None:
        config = load_config()

    model_name = config["llm"]["base_model"]
    adapter_path = os.path.join(config["training"]["output_dir"], "final_adapter")

    print(f"Loading base model: {model_name}")
    device = get_device()

    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("Fine-tuned model loaded successfully!")
    return model, tokenizer


# --- Main ---
if __name__ == "__main__":
    config = load_config()
    print("Starting LoRA fine-tuning of BioMistral-7B...")
    print("This may take several hours depending on hardware.\n")
    model, tokenizer, trainer = train(config)
    print("\nFine-tuning complete!")
