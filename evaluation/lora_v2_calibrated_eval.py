# -*- coding: utf-8 -*-
"""
evaluation/lora_v2_calibrated_eval.py
Grid-search per-class bias on val set then apply to test set,
using the LoRA v2 fine-tuned model (not zero-shot baseline).

Steps:
  1. Load BioGPT-Large + lora_v2_best adapter
  2. Collect raw log-prob scores on full val set
  3. Grid-search biases [b_maybe, b_no] to maximise val accuracy
  4. Apply best bias to 150 test samples and report metrics

Run:
    python evaluation/lora_v2_calibrated_eval.py
"""
import sys, os, json, time, re, logging
import numpy as np
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, MODEL_DIR, LABEL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LORA_PATH     = os.path.join(MODEL_DIR, "lora_v2_best")
MODEL_NAME    = "microsoft/BioGPT-Large-PubMedQA"
TARGET_PREFIX = "the answer to the question given the context is "
MAX_LEN       = 512


def build_prompt(q, ctx, max_ctx=1800):
    ctx_c = re.sub(r'\s+', ' ', ctx.strip())[:max_ctx]
    return "question: " + q.strip() + " context: " + ctx_c + " " + TARGET_PREFIX


def get_scores(model, tokenizer, prompt, label_tokens, device):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=MAX_LEN).to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
        lp  = torch.log_softmax(out.logits[0, -1], dim=-1)
    return {lbl: float(lp[tid]) for lbl, tid in label_tokens.items()}


def run(n_test=150):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    device = "cpu"
    logger.info(f"Loading {MODEL_NAME} + LoRA v2 adapter ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model = model.to(device)
    model.eval()
    logger.info("Model loaded")

    label_tokens = {}
    for lbl in LABEL_NAMES:
        ids = (tokenizer.encode(" " + lbl, add_special_tokens=False) or
               tokenizer.encode(lbl, add_special_tokens=False))
        label_tokens[lbl] = ids[0]
    logger.info(f"Label tokens: {label_tokens}")

    # Load data
    with open(os.path.join(DATA_DIR, "val.json"))  as f: val_all  = json.load(f)
    with open(os.path.join(DATA_DIR, "test.json")) as f: test_all = json.load(f)
    import random; random.seed(42)
    test_data = (test_all[:n_test] if n_test >= len(test_all)
                 else random.sample(test_all, n_test))
    logger.info(f"Val: {len(val_all)} samples  Test: {len(test_data)} samples")

    # Step 1: Collect scores on FULL val set
    logger.info("Collecting val scores ...")
    val_true, val_scores = [], []
    for i, rec in enumerate(val_all):
        q, ctx, lbl = rec["question"], rec.get("context",""), rec["final_decision"]
        scores = get_scores(model, tokenizer, build_prompt(q, ctx), label_tokens, device)
        val_true.append(lbl)
        val_scores.append(scores)
        if (i+1) % 20 == 0 or i == 0:
            logger.info(f"  val [{i+1}/{len(val_all)}]")

    # Step 2: Grid search biases on val set
    logger.info("Grid searching biases on val set ...")
    best_val_acc, best_bias = 0, {"yes": 0.0, "no": 0.0, "maybe": 0.0}
    b_yes_range   = [0.0]
    b_no_range    = np.arange(-1.5, 1.51, 0.25)
    b_maybe_range = np.arange(0.0, 4.01, 0.25)

    for b_no, b_maybe in product(b_no_range, b_maybe_range):
        bias = {"yes": 0.0, "no": float(b_no), "maybe": float(b_maybe)}
        correct = sum(
            1 for s, t in zip(val_scores, val_true)
            if max({l: s[l] + bias[l] for l in LABEL_NAMES}, key=lambda l: s[l]+bias[l]) == t
        )
        acc = correct / len(val_true)
        if acc > best_val_acc:
            best_val_acc = acc
            best_bias = bias.copy()

    logger.info(f"Best val bias: {best_bias}  val_acc={best_val_acc:.3f}")

    # Step 3: Apply to test set
    logger.info("Evaluating test set with best bias ...")
    true_labels, pred_raw, pred_cal = [], [], []
    pred_dist_raw = {l: 0 for l in LABEL_NAMES}
    pred_dist_cal = {l: 0 for l in LABEL_NAMES}
    latencies = []

    for i, rec in enumerate(test_data):
        q, ctx, lbl = rec["question"], rec.get("context",""), rec["final_decision"]
        true_labels.append(lbl)
        t0 = time.time()
        scores = get_scores(model, tokenizer, build_prompt(q, ctx), label_tokens, device)
        latencies.append((time.time()-t0)*1000)

        p_raw = max(scores, key=scores.get)
        pred_raw.append(p_raw); pred_dist_raw[p_raw] += 1

        cal_scores = {l: scores[l] + best_bias[l] for l in LABEL_NAMES}
        p_cal = max(cal_scores, key=cal_scores.get)
        pred_cal.append(p_cal); pred_dist_cal[p_cal] += 1

        if (i+1) % 15 == 0 or i == 0:
            logger.info(f"  test [{i+1}/{n_test}] true={lbl} raw={p_raw} cal={p_cal}")

    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    def report(name, true_l, pred_l, pred_d):
        acc    = accuracy_score(true_l, pred_l)
        f1_mac = f1_score(true_l, pred_l, labels=LABEL_NAMES, average="macro", zero_division=0)
        f1_wei = f1_score(true_l, pred_l, labels=LABEL_NAMES, average="weighted", zero_division=0)
        prec, rec, f1, sup = precision_recall_fscore_support(
            true_l, pred_l, labels=LABEL_NAMES, zero_division=0)
        print(f"\n{'='*64}")
        print(f"  {name}")
        print(f"{'='*64}")
        print(f"  Accuracy   : {acc:.4f}  ({acc*100:.1f}%)")
        print(f"  F1 Macro   : {f1_mac:.4f}")
        print(f"  F1 Weighted: {f1_wei:.4f}")
        for k, lbl in enumerate(LABEL_NAMES):
            print(f"    {lbl.upper():5s}  P={prec[k]:.3f}  R={rec[k]:.3f}  "
                  f"F1={f1[k]:.3f}  n={sup[k]}")
        print(f"  Pred dist  : {pred_d}")
        return round(acc, 4), round(f1_mac, 4), {
            lbl: {"precision": round(float(prec[k]),4),
                  "recall":    round(float(rec[k]),4),
                  "f1":        round(float(f1[k]),4)}
            for k, lbl in enumerate(LABEL_NAMES)}

    acc_raw, f1_raw, pc_raw = report(
        "LoRA v2 RAW (no calibration)", true_labels, pred_raw, pred_dist_raw)
    acc_cal, f1_cal, pc_cal = report(
        f"LoRA v2 CALIBRATED  bias={best_bias}", true_labels, pred_cal, pred_dist_cal)

    print(f"\n  vs Zero-shot correct format: 65.3%")
    print(f"  LoRA v2 raw delta  : {(acc_raw - 0.6533)*100:+.1f}pp")
    print(f"  LoRA v2 calib delta: {(acc_cal - 0.6533)*100:+.1f}pp")
    print(f"  Best bias: {best_bias}")
    print(f"  Avg latency: {np.mean(latencies):.0f} ms")

    result = {
        "method":    "lora_v2_calibrated",
        "best_bias": best_bias,
        "best_val_acc": round(best_val_acc, 4),
        "n_test": len(test_data),
        "raw":  {"accuracy": acc_raw, "f1_macro": f1_raw,
                 "pred_dist": pred_dist_raw, "per_class": pc_raw},
        "cal":  {"accuracy": acc_cal, "f1_macro": f1_cal,
                 "pred_dist": pred_dist_cal, "per_class": pc_cal},
        "avg_latency_ms": round(float(np.mean(latencies)), 1),
    }
    out = f"/Users/yuvrajpratapsingh/medical_rag/results/lora_v2_calibrated_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved -> {out}")
    return result


if __name__ == "__main__":
    run(n_test=150)
