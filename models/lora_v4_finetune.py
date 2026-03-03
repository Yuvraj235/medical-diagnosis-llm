# -*- coding: utf-8 -*-
"""
models/lora_v4_finetune.py
LoRA v4 - MPS (Apple Silicon GPU) accelerated training.

Key change over v3: run on MPS device (Metal) instead of CPU.
MPS is 10-30x faster than CPU for transformer matrix ops.

Changes:
  1. device = "mps"  (Apple Silicon GPU via Metal Performance Shaders)
  2. dtype = float16  (MPS-friendly, halves VRAM vs float32)
  3. NO gradient checkpointing  (MPS is fast enough; GC was causing CPU slowdown)
  4. Saves to lora_v4_best

Everything else identical to v2/v3 gold-standard config.

Expected runtime:  1-3 hours  (vs 24h+ on CPU)
Expected accuracy: 70%+

Run:
    python models/lora_v4_finetune.py
"""
import sys, os, json, time, logging, random, re
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, MODEL_DIR, LABEL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LORA_OUT   = os.path.join(MODEL_DIR, "lora_v4_best")
MODEL_NAME = "microsoft/BioGPT-Large-PubMedQA"

# Hypers - identical to v2 gold standard
MAX_LEN       = 512
BATCH_SIZE    = 1
GRAD_ACCUM    = 4
LEARNING_RATE = 1e-4
N_EPOCHS      = 5
LOG_EVERY     = 20
EVAL_EVERY    = 50
N_EVAL        = 30
PATIENCE      = 2
WARMUP_STEPS  = 10

CLASS_WEIGHTS = {"yes": 1.0, "no": 1.5, "maybe": 4.0}
TARGET_PREFIX = "the answer to the question given the context is "


def build_prompt(rec, include_label=True):
    """Exact Microsoft BioGPT-PubMedQA training format."""
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
            lp = torch.log_softmax(
                    model(**inputs).logits[0, -1].float(), dim=-1)
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
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    # Use MPS (Apple Silicon GPU) for 10-30x speedup over CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype  = torch.float16
        logger.info("Using MPS (Apple Silicon GPU) with float16")
    else:
        device = torch.device("cpu")
        dtype  = torch.float32
        logger.info("MPS not available, falling back to CPU float32")

    logger.info(f"Device: {device}  dtype={dtype}  MAX_LEN={MAX_LEN}  LR={LEARNING_RATE}")
    logger.info(f"Class weights: {CLASS_WEIGHTS}")

    # Load data
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

    # Load model - first to CPU, then move to MPS
    logger.info(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = model.to(device)
    logger.info(f"Model loaded on {device}  param dtype={next(model.parameters()).dtype}")

    # Label token IDs
    label_tokens = {}
    for lbl in LABEL_NAMES:
        ids = (tokenizer.encode(" " + lbl, add_special_tokens=False) or
               tokenizer.encode(lbl, add_special_tokens=False))
        label_tokens[lbl] = ids[0]
    logger.info(f"Label token IDs: {label_tokens}")

    # LoRA config - identical to v2
    linear_names = {n.split(".")[-1] for n, m in model.named_modules()
                    if isinstance(m, torch.nn.Linear)}
    preferred = ["q_proj", "v_proj", "k_proj", "out_proj"]
    targets   = [m for m in preferred if m in linear_names]
    if len(targets) < 2:
        targets = sorted(linear_names)[:4]
    logger.info(f"LoRA v4 targets ({len(targets)} modules): {targets}")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Baseline eval
    logger.info("Baseline eval:")
    base_acc = quick_eval(model, tokenizer, val_data, label_tokens, device)

    # Optimizer + scheduler
    trainable     = [p for p in model.parameters() if p.requires_grad]
    optimizer     = AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    total_updates = (len(train_data) * N_EPOCHS) // GRAD_ACCUM
    scheduler     = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, total_updates // 2))

    logger.info(f"\nTraining: {N_EPOCHS} epochs x {len(train_data)} samples"
                f" = ~{total_updates} gradient updates")

    history      = {"step": [], "loss": [], "val_acc": [base_acc]}
    best_acc     = base_acc
    patience_cnt = 0
    global_step  = 0
    grad_step    = 0
    run_loss     = 0.0
    t_start      = time.time()
    optimizer.zero_grad()
    os.makedirs(LORA_OUT, exist_ok=True)

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

            # Safety check: skip if label token was truncated away
            expected_label_id = label_tokens[label]
            if input_ids[-1].item() != expected_label_id:
                global_step += 1
                continue

            labels     = torch.full_like(input_ids, -100)
            labels[-1] = input_ids[-1]

            out  = model(input_ids=input_ids.unsqueeze(0),
                         attention_mask=attn_mask.unsqueeze(0),
                         labels=labels.unsqueeze(0))
            loss = out.loss

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"  Skipping NaN/Inf at step {global_step}")
                optimizer.zero_grad()
                global_step += 1
                continue

            loss = loss * w / GRAD_ACCUM
            loss.backward()
            run_loss    += float(loss.item()) * GRAD_ACCUM / w
            global_step += 1

            # Warmup
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
                        # Save to CPU before writing
                        model.save_pretrained(LORA_OUT)
                        tokenizer.save_pretrained(LORA_OUT)
                        logger.info(f"  *** New best acc={val_acc:.3f} -- saved ***")
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
        model.save_pretrained(LORA_OUT)
        tokenizer.save_pretrained(LORA_OUT)
        best_acc = final_acc

    with open(os.path.join(MODEL_DIR, "lora_v4_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(LORA_OUT, "base_model.txt"), "w") as f:
        f.write(MODEL_NAME)

    total_min = (time.time() - t_start) / 60
    logger.info(f"\n LoRA v4 complete!  ({total_min:.1f} min)")
    logger.info(f"  Baseline acc : {base_acc:.3f}")
    logger.info(f"  Best val acc : {best_acc:.3f}")
    logger.info(f"  Adapter saved: {LORA_OUT}")
    return history


if __name__ == "__main__":
    run_finetune()
