# -*- coding: utf-8 -*-
"""
models/fast_lora_finetune.py
Fast LoRA fine-tuning of BioGPT-Large-PubMedQA on CPU.

Teaches the model to output yes/no/maybe in our evaluation format.
Optimised for speed: max_len=256, batch=1, accum=2, 150 samples x 2 epochs.
~15-25 minutes on Apple Silicon CPU.

Run:
    python models/fast_lora_finetune.py
"""
import sys, os, json, time, logging, random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, MODEL_DIR, LABEL_NAMES

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LORA_OUT   = os.path.join(MODEL_DIR, "lora_best")
MODEL_NAME = "microsoft/BioGPT-Large-PubMedQA"

# ── Hypers tuned for CPU speed ──────────────────────────────────────────────
MAX_LEN       = 256   # SHORT sequences → fast backward pass
BATCH_SIZE    = 1
GRAD_ACCUM    = 2     # effective batch = 2
LEARNING_RATE = 2e-4  # safe LR for LoRA on large model
MAX_SAMPLES   = 150   # 150 training samples × 2 epochs = 300 grad steps→150 updates
N_EPOCHS      = 2
LOG_EVERY     = 10
EVAL_EVERY    = 30
N_EVAL        = 20


def build_prompt(rec, include_label=True):
    """Exact format used in real_eval.py evaluation."""
    ctx = rec.get("context", "")[:600]       # ~150 tokens; keeps label inside 256-token window
    q   = rec.get("question", "")
    base = f"{ctx}\n\nQuestion: {q}\n\nAnswer:"
    if include_label:
        return base + " " + rec.get("final_decision", "maybe")
    return base


def quick_eval(model, tokenizer, val_recs, label_tokens, device):
    import torch
    model.eval()
    correct = 0
    pred_dist = {l: 0 for l in LABEL_NAMES}
    for rec in val_recs:
        lbl    = rec.get("final_decision", "maybe")
        prompt = build_prompt(rec, include_label=False)
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=MAX_LEN).to(device)
        with torch.no_grad():
            lp = torch.log_softmax(model(**inputs).logits[0, -1], dim=-1)
        scores = {l: float(lp[tid]) for l, tid in label_tokens.items()}
        pred   = max(scores, key=scores.get)
        pred_dist[pred] += 1
        if pred == lbl:
            correct += 1
    model.train()
    acc = correct / len(val_recs)
    logger.info(f"    Quick eval: acc={acc:.3f}  pred_dist={pred_dist}")
    return acc


def run_finetune():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    device = "cpu"
    logger.info(f"Device: {device}  MAX_LEN={MAX_LEN}  samples={MAX_SAMPLES}x{N_EPOCHS}")

    with open(os.path.join(DATA_DIR, "train.json")) as f: train_all = json.load(f)
    with open(os.path.join(DATA_DIR, "val.json"))   as f: val_all   = json.load(f)

    # Stratified sample: equal yes/no/maybe
    random.seed(42)
    by_lbl = {l: [r for r in train_all if r["final_decision"]==l] for l in LABEL_NAMES}
    per_lbl = MAX_SAMPLES // 3
    train_data = []
    for l in LABEL_NAMES:
        random.shuffle(by_lbl[l])
        train_data += by_lbl[l][:per_lbl]
    random.shuffle(train_data)
    val_data = random.sample(val_all, min(N_EVAL, len(val_all)))
    logger.info(f"Train: {len(train_data)} ({per_lbl} per class)  Val: {len(val_data)}")

    # ── Load model ─────────────────────────────────────────────────────────
    logger.info(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32, low_cpu_mem_usage=True)
    model = model.to(device)

    # ── Label token IDs ────────────────────────────────────────────────────
    label_tokens = {}
    for lbl in LABEL_NAMES:
        ids = (tokenizer.encode(" " + lbl, add_special_tokens=False) or
               tokenizer.encode(lbl, add_special_tokens=False))
        label_tokens[lbl] = ids[0]
    logger.info(f"Label token IDs: {label_tokens}")

    # ── LoRA ───────────────────────────────────────────────────────────────
    linear_names = {n.split(".")[-1] for n, m in model.named_modules()
                    if isinstance(m, torch.nn.Linear)}
    targets = [m for m in ["q_proj", "v_proj"] if m in linear_names]  # minimal: 2 modules
    if not targets:
        targets = sorted(linear_names)[:2]
    logger.info(f"LoRA targets: {targets}")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,             # tiny r=4 for speed
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Baseline ───────────────────────────────────────────────────────────
    logger.info("Baseline eval:")
    base_acc = quick_eval(model, tokenizer, val_data, label_tokens, device)

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable  = [p for p in model.parameters() if p.requires_grad]
    optimizer  = AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    total_updates = (len(train_data) * N_EPOCHS) // GRAD_ACCUM
    scheduler  = CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-5)

    # ── Training loop ──────────────────────────────────────────────────────
    logger.info(f"\nTraining: {N_EPOCHS} epochs x {len(train_data)} samples"
                f" = ~{total_updates} gradient updates")
    history     = {"step": [], "loss": [], "val_acc": [base_acc]}
    best_acc    = base_acc
    global_step = 0
    grad_step   = 0
    run_loss    = 0.0
    t_start     = time.time()
    optimizer.zero_grad()
    os.makedirs(LORA_OUT, exist_ok=True)

    for epoch in range(N_EPOCHS):
        random.shuffle(train_data)
        logger.info(f"\n-- Epoch {epoch+1}/{N_EPOCHS} --")

        for rec in train_data:
            prompt = build_prompt(rec, include_label=True)
            # NO padding — keep variable length so label is always the last token
            enc    = tokenizer(prompt, max_length=MAX_LEN, truncation=True,
                               return_tensors="pt")
            input_ids = enc["input_ids"][0].to(device)   # shape: (seq_len,)
            attn_mask = enc["attention_mask"][0].to(device)

            # Loss only on the last token (the yes/no/maybe label)
            labels = torch.full_like(input_ids, -100)
            labels[-1] = input_ids[-1]   # only the label position contributes

            out  = model(input_ids=input_ids.unsqueeze(0),
                         attention_mask=attn_mask.unsqueeze(0),
                         labels=labels.unsqueeze(0))
            loss = out.loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"  Skipping NaN/Inf loss at step {global_step}")
                optimizer.zero_grad()
                global_step += 1
                continue
            loss = loss / GRAD_ACCUM
            loss.backward()
            run_loss    += loss.item() * GRAD_ACCUM
            global_step += 1

            if global_step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                grad_step += 1

                if grad_step % LOG_EVERY == 0:
                    avg = run_loss / LOG_EVERY
                    run_loss = 0.0
                    eta = (time.time() - t_start) / grad_step * (total_updates - grad_step) / 60
                    logger.info(f"  Step {grad_step}/{total_updates}"
                                f"  loss={avg:.4f}  ETA={eta:.1f}m")
                    history["step"].append(grad_step)
                    history["loss"].append(round(avg, 4))

                if grad_step % EVAL_EVERY == 0:
                    val_acc = quick_eval(model, tokenizer, val_data, label_tokens, device)
                    history["val_acc"].append(val_acc)
                    if val_acc > best_acc:
                        best_acc = val_acc
                        model.save_pretrained(LORA_OUT)
                        tokenizer.save_pretrained(LORA_OUT)
                        logger.info(f"  *** Saved best (acc={val_acc:.3f}) ***")

        # (timer already initialized above)

    # ── Final eval ────────────────────────────────────────────────────────
    final_acc = quick_eval(model, tokenizer, val_data, label_tokens, device)
    history["final_val_acc"] = final_acc
    if final_acc >= best_acc:
        model.save_pretrained(LORA_OUT)
        tokenizer.save_pretrained(LORA_OUT)
    with open(os.path.join(MODEL_DIR, "lora_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(LORA_OUT, "base_model.txt"), "w") as f:
        f.write(MODEL_NAME)

    logger.info(f"\n LoRA fine-tuning complete!")
    logger.info(f"  Baseline acc: {base_acc:.3f}   Final acc: {final_acc:.3f}")
    logger.info(f"  Improvement : +{(final_acc-base_acc)*100:.1f}%")
    logger.info(f"  Adapter saved to {LORA_OUT}")
    return history


if __name__ == "__main__":
    t_start = time.time()
    run_finetune()
