# -*- coding: utf-8 -*-
"""
models/lora_v2_continue.py
Warm-start continuation from lora_v2_best checkpoint (65.3% test acc).

Starts from the saved LoRA adapter (step 200, val_acc=63.3%) and continues
training to push accuracy from 65% -> 70%.

Key settings vs original v2:
  - Loads lora_v2_best as initial adapter weights
  - CPU float32 (reliable, no MPS complications)
  - use_cache=False in forward (avoids 400MB KV-cache memory per step)
  - LR = 5e-5  (warm-start needs lower LR to not destroy learned weights)
  - EVAL_EVERY = 30  (more frequent checkpointing)
  - PATIENCE = 3
  - MAX 3 epochs (should be enough for +7pp improvement)

Run:
    python models/lora_v2_continue.py
"""
import sys, os, json, time, logging, random, re
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, MODEL_DIR, LABEL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LORA_BASE_PATH = os.path.join(MODEL_DIR, "lora_v2_best")   # warm-start from here
LORA_SAVE_PATH = os.path.join(MODEL_DIR, "lora_v2_best")   # overwrite if better
MODEL_NAME     = "microsoft/BioGPT-Large-PubMedQA"

MAX_LEN       = 512
BATCH_SIZE    = 1
GRAD_ACCUM    = 4
LEARNING_RATE = 5e-5    # lower LR for warm-start
N_EPOCHS      = 3       # continue for 3 more epochs
LOG_EVERY     = 20
EVAL_EVERY    = 30      # more frequent evals
N_EVAL        = 30
PATIENCE      = 3
WARMUP_STEPS  = 5

CLASS_WEIGHTS = {"yes": 1.0, "no": 1.5, "maybe": 4.0}
TARGET_PREFIX = "the answer to the question given the context is "


def build_prompt(rec, include_label=True):
    ctx  = re.sub(r'\s+', ' ', rec.get("context", "").strip())[:1800]
    q    = rec.get("question", "").strip()
    base = "question: " + q + " context: " + ctx + " " + TARGET_PREFIX
    if include_label:
        return base + rec.get("final_decision", "maybe")
    return base


def quick_eval(model, tokenizer, val_recs, label_tokens, device):
    import torch
    model.eval()
    correct   = 0
    pred_dist = {l: 0 for l in LABEL_NAMES}
    per_class = {l: {"correct": 0, "total": 0} for l in LABEL_NAMES}
    for rec in val_recs:
        lbl    = rec.get("final_decision", "maybe")
        prompt = build_prompt(rec, include_label=False)
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=MAX_LEN).to(device)
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
            lp  = torch.log_softmax(out.logits[0, -1], dim=-1)
        scores = {l: float(lp[tid]) for l, tid in label_tokens.items()}
        pred   = max(scores, key=scores.get)
        pred_dist[pred] += 1
        per_class[lbl]["total"] += 1
        if pred == lbl:
            correct += 1
            per_class[lbl]["correct"] += 1
    model.train()
    acc = correct / len(val_recs)
    per_class_str = "  ".join(
        f"{l}={per_class[l]['correct']}/{per_class[l]['total']}"
        for l in LABEL_NAMES)
    logger.info(f"    eval: acc={acc:.3f}  preds={pred_dist}  [{per_class_str}]")
    return acc


def run_finetune():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    device = "cpu"
    logger.info(f"Warm-start continuation from: {LORA_BASE_PATH}")
    logger.info(f"Device: {device}  MAX_LEN={MAX_LEN}  LR={LEARNING_RATE}")
    logger.info(f"Class weights: {CLASS_WEIGHTS}")

    with open(os.path.join(DATA_DIR, "train.json")) as f: train_all = json.load(f)
    with open(os.path.join(DATA_DIR, "val.json"))   as f: val_all   = json.load(f)

    train_data = train_all[:]
    random.seed(42)
    by_lbl   = {l: [r for r in val_all if r["final_decision"] == l] for l in LABEL_NAMES}
    val_data = []
    for l in LABEL_NAMES:
        val_data += random.sample(by_lbl[l], min(N_EVAL // 3, len(by_lbl[l])))
    random.shuffle(val_data)

    label_dist = {l: sum(1 for r in train_data if r["final_decision"] == l) for l in LABEL_NAMES}
    logger.info(f"Train: {len(train_data)} samples  dist={label_dist}")
    logger.info(f"Val:   {len(val_data)} samples")

    # Load base model + existing LoRA adapter
    logger.info(f"Loading {MODEL_NAME} + adapter from {LORA_BASE_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base_model, LORA_BASE_PATH, is_trainable=True)
    model = model.to(device)
    model.train()
    model.print_trainable_parameters()
    logger.info("Warm-start model loaded")

    label_tokens = {}
    for lbl in LABEL_NAMES:
        ids = (tokenizer.encode(" " + lbl, add_special_tokens=False) or
               tokenizer.encode(lbl, add_special_tokens=False))
        label_tokens[lbl] = ids[0]
    logger.info(f"Label token IDs: {label_tokens}")

    # Baseline eval (= current checkpoint performance)
    logger.info("Baseline eval (warm-start checkpoint):")
    base_acc = quick_eval(model, tokenizer, val_data, label_tokens, device)

    trainable     = [p for p in model.parameters() if p.requires_grad]
    optimizer     = AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    total_updates = (len(train_data) * N_EPOCHS) // GRAD_ACCUM
    scheduler     = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, total_updates // 2))

    logger.info(f"\nContinuation: {N_EPOCHS} epochs x {len(train_data)} samples"
                f" = ~{total_updates} gradient updates")

    history      = {"step": [], "loss": [], "val_acc": [base_acc]}
    best_acc     = base_acc
    patience_cnt = 0
    global_step  = 0
    grad_step    = 0
    run_loss     = 0.0
    t_start      = time.time()
    optimizer.zero_grad()
    os.makedirs(LORA_SAVE_PATH, exist_ok=True)

    stop_training = False
    for epoch in range(N_EPOCHS):
        if stop_training:
            break
        random.shuffle(train_data)
        logger.info(f"\n-- Epoch {epoch+1}/{N_EPOCHS} --")

        for rec in train_data:
            if stop_training:
                break

            label = rec.get("final_decision", "maybe")
            w     = CLASS_WEIGHTS.get(label, 1.0)

            prompt    = build_prompt(rec, include_label=True)
            enc       = tokenizer(prompt, max_length=MAX_LEN, truncation=True,
                                  return_tensors="pt")
            input_ids = enc["input_ids"][0].to(device)
            attn_mask = enc["attention_mask"][0].to(device)

            expected_label_id = label_tokens[label]
            if input_ids[-1].item() != expected_label_id:
                global_step += 1
                continue

            labels     = torch.full_like(input_ids, -100)
            labels[-1] = input_ids[-1]

            out  = model(input_ids=input_ids.unsqueeze(0),
                         attention_mask=attn_mask.unsqueeze(0),
                         labels=labels.unsqueeze(0),
                         use_cache=False)    # key: no KV-cache memory
            loss = out.loss
            del out     # free memory immediately

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"  Skipping NaN/Inf at step {global_step}")
                optimizer.zero_grad()
                global_step += 1
                continue

            loss = loss * w / GRAD_ACCUM
            loss.backward()
            run_loss    += loss.item() * GRAD_ACCUM / w
            global_step += 1

            if grad_step < WARMUP_STEPS:
                for pg in optimizer.param_groups:
                    pg["lr"] = LEARNING_RATE * (grad_step + 1) / WARMUP_STEPS

            if global_step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 0.5)
                optimizer.step()
                if grad_step >= WARMUP_STEPS:
                    scheduler.step()
                optimizer.zero_grad()
                grad_step += 1

                if grad_step % LOG_EVERY == 0:
                    avg      = run_loss / LOG_EVERY
                    run_loss = 0.0
                    elapsed  = time.time() - t_start
                    eta      = elapsed / grad_step * (total_updates - grad_step) / 60
                    logger.info(f"  Step {grad_step}/{total_updates}"
                                f"  loss={avg:.4f}  ETA={eta:.1f}m")
                    history["step"].append(grad_step)
                    history["loss"].append(round(avg, 4))

                if grad_step % EVAL_EVERY == 0:
                    val_acc = quick_eval(model, tokenizer, val_data,
                                         label_tokens, device)
                    history["val_acc"].append(val_acc)
                    if val_acc > best_acc:
                        best_acc     = val_acc
                        patience_cnt = 0
                        model.save_pretrained(LORA_SAVE_PATH)
                        tokenizer.save_pretrained(LORA_SAVE_PATH)
                        logger.info(f"  *** New best acc={val_acc:.3f} -- saved to {LORA_SAVE_PATH} ***")
                    else:
                        patience_cnt += 1
                        logger.info(f"  No improvement ({patience_cnt}/{PATIENCE})")
                        if patience_cnt >= PATIENCE:
                            logger.info("  Early stopping triggered.")
                            stop_training = True
                            break

    # Final eval
    final_acc = quick_eval(model, tokenizer, val_data, label_tokens, device)
    history["final_val_acc"] = final_acc
    if final_acc >= best_acc:
        model.save_pretrained(LORA_SAVE_PATH)
        tokenizer.save_pretrained(LORA_SAVE_PATH)
        best_acc = final_acc

    with open(os.path.join(MODEL_DIR, "lora_v2_continue_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    total_min = (time.time() - t_start) / 60
    logger.info(f"\n Continuation complete!  ({total_min:.1f} min)")
    logger.info(f"  Warm-start acc : {base_acc:.3f}")
    logger.info(f"  Best val acc   : {best_acc:.3f}")
    logger.info(f"  Adapter saved  : {LORA_SAVE_PATH}")
    return history


if __name__ == "__main__":
    run_finetune()
