# -*- coding: utf-8 -*-
"""
evaluation/threshold_calibration.py
Threshold calibration for BioGPT-Large-PubMedQA.

Approach (standard ML calibration):
  1. Run model on full val set (200 samples) → collect raw logit scores
  2. Grid search per-class bias vector [b_yes, b_no, b_maybe] on val set
     to maximise accuracy (equivalent to Platt scaling / label bias)
  3. Apply best bias to test set (150 samples)
  4. Report calibrated accuracy — fully reproducible, no data leakage

This fixes the MAYBE=0% recall problem without any training.
Tuning on val set and evaluating on test set is standard calibration practice.

Run:
    python evaluation/threshold_calibration.py
"""
import sys, os, json, time, re, logging
import numpy as np
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, LABEL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_PREFIX = "the answer to the question given the context is "


def build_prompt(q, ctx, max_ctx=1800):
    ctx_c = re.sub(r'\s+', ' ', ctx.strip())[:max_ctx]
    return f"question: {q.strip()} context: {ctx_c} {TARGET_PREFIX}"


def get_scores(model, tokenizer, prompt, label_tokens, device):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    lp = torch.log_softmax(logits[0, -1, :], dim=-1)
    return {lbl: float(lp[tid]) for lbl, tid in label_tokens.items()}


def collect_scores(model, tokenizer, dataset, label_tokens, device, name):
    """Collect raw log-prob scores for all samples in dataset."""
    logger.info(f"Collecting scores for {name} ({len(dataset)} samples)...")
    records = []
    for i, rec in enumerate(dataset):
        q   = rec["question"]
        ctx = rec.get("context", "")
        lbl = rec["final_decision"]
        t0  = time.time()
        prompt = build_prompt(q, ctx)
        scores = get_scores(model, tokenizer, prompt, label_tokens, device)
        lat = (time.time() - t0) * 1000
        records.append({"true": lbl, "scores": scores})
        if (i+1) % 20 == 0 or i == 0:
            logger.info(f"  [{i+1}/{len(dataset)}] lat={lat:.0f}ms "
                        f"true={lbl:5s} raw_pred={max(scores, key=scores.get):5s}")
    return records


def calibrated_acc(records, bias):
    """Accuracy with per-class bias vector."""
    correct = 0
    pred_dist = {l: 0 for l in LABEL_NAMES}
    for rec in records:
        cal_scores = {lbl: rec["scores"][lbl] + bias[lbl] for lbl in LABEL_NAMES}
        pred = max(cal_scores, key=cal_scores.get)
        pred_dist[pred] += 1
        if pred == rec["true"]:
            correct += 1
    return correct / len(records), pred_dist


def calibrated_metrics(records, bias):
    """Full metrics with per-class bias vector."""
    true_labels, pred_labels = [], []
    for rec in records:
        cal_scores = {lbl: rec["scores"][lbl] + bias[lbl] for lbl in LABEL_NAMES}
        pred = max(cal_scores, key=cal_scores.get)
        true_labels.append(rec["true"])
        pred_labels.append(pred)
    return true_labels, pred_labels


def run(n_val=200, n_test=150):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cpu"
    model_name = "microsoft/BioGPT-Large-PubMedQA"
    logger.info(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    logger.info("Model loaded")

    label_tokens = {}
    for lbl in LABEL_NAMES:
        ids = (tokenizer.encode(" " + lbl, add_special_tokens=False) or
               tokenizer.encode(lbl, add_special_tokens=False))
        label_tokens[lbl] = ids[0]
    logger.info(f"Label tokens: {label_tokens}")

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(os.path.join(DATA_DIR, "val.json"))  as f: all_val  = json.load(f)
    with open(os.path.join(DATA_DIR, "test.json")) as f: all_test = json.load(f)

    import random; random.seed(42)
    # Val: use up to n_val samples, stratified by class
    by_lbl_val = {l: [r for r in all_val if r["final_decision"] == l] for l in LABEL_NAMES}
    n_each_val = min(n_val // 3, min(len(v) for v in by_lbl_val.values()))
    val_data = []
    for l in LABEL_NAMES:
        val_data += random.sample(by_lbl_val[l], n_each_val)
    random.shuffle(val_data)
    logger.info(f"Val set: {len(val_data)} samples ({n_each_val} per class)")

    # Test: fixed 150 random samples
    test_data = all_test[:n_test] if n_test >= len(all_test) \
                else random.sample(all_test, n_test)
    logger.info(f"Test set: {len(test_data)} samples")

    # ── Step 1: Collect scores ─────────────────────────────────────────────────
    val_recs  = collect_scores(model, tokenizer, val_data,  label_tokens, device, "val")
    test_recs = collect_scores(model, tokenizer, test_data, label_tokens, device, "test")

    # ── Step 2: Baseline (no calibration) ─────────────────────────────────────
    zero_bias = {l: 0.0 for l in LABEL_NAMES}
    base_val_acc,  base_val_pd  = calibrated_acc(val_recs,  zero_bias)
    base_test_acc, base_test_pd = calibrated_acc(test_recs, zero_bias)
    logger.info(f"Baseline   val acc={base_val_acc:.3f}  pred_dist={base_val_pd}")
    logger.info(f"Baseline  test acc={base_test_acc:.3f}  pred_dist={base_test_pd}")

    # ── Step 3: Grid search on val set ────────────────────────────────────────
    # Tune b_maybe and b_no (fix b_yes=0); search over 25×15=375 combinations
    logger.info("Grid searching per-class biases on val set...")
    b_maybe_grid = np.arange(0.0, 5.1, 0.25)   # 0 to 5 in 0.25 steps
    b_no_grid    = np.arange(-1.0, 1.1, 0.25)  # -1 to +1 in 0.25 steps
    best_val_acc = base_val_acc
    best_bias    = zero_bias.copy()

    for b_maybe, b_no in product(b_maybe_grid, b_no_grid):
        bias = {"yes": 0.0, "no": b_no, "maybe": b_maybe}
        acc, _ = calibrated_acc(val_recs, bias)
        if acc > best_val_acc:
            best_val_acc = acc
            best_bias = bias.copy()

    logger.info(f"Best val acc: {best_val_acc:.3f}  bias={best_bias}")

    # ── Step 4: Apply best bias to test set ───────────────────────────────────
    cal_test_acc, cal_test_pd = calibrated_acc(test_recs, best_bias)
    true_labels, pred_labels  = calibrated_metrics(test_recs, best_bias)

    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    acc     = accuracy_score(true_labels, pred_labels)
    f1_mac  = f1_score(true_labels, pred_labels, labels=LABEL_NAMES, average="macro", zero_division=0)
    f1_wei  = f1_score(true_labels, pred_labels, labels=LABEL_NAMES, average="weighted", zero_division=0)
    prec, rec, f1, sup = precision_recall_fscore_support(
        true_labels, pred_labels, labels=LABEL_NAMES, zero_division=0)

    print("\n" + "="*64)
    print(f"  CALIBRATED BioGPT-Large (bias tuned on val set)")
    print(f"  Bias: yes={best_bias['yes']:.2f}  no={best_bias['no']:.2f}  "
          f"maybe={best_bias['maybe']:.2f}")
    print("="*64)
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1 Macro   : {f1_mac:.4f}")
    print(f"  F1 Weighted: {f1_wei:.4f}")
    for k, lbl in enumerate(LABEL_NAMES):
        print(f"    {lbl.upper():5s}  P={prec[k]:.3f}  R={rec[k]:.3f}  "
              f"F1={f1[k]:.3f}  n={sup[k]}")
    print(f"  Pred dist  : {cal_test_pd}")
    print("="*64)
    print(f"\n  vs Raw (no calibration): {base_test_acc*100:.1f}%")
    print(f"  Calibration delta: {(acc-base_test_acc)*100:+.1f}pp")

    result = {
        "method": "threshold_calibration",
        "bias": best_bias,
        "best_val_acc": round(best_val_acc, 4),
        "baseline_test_acc": round(base_test_acc, 4),
        "n_val":  len(val_data),
        "n_test": len(test_data),
        "accuracy":    round(acc, 4),
        "f1_macro":    round(f1_mac, 4),
        "f1_weighted": round(f1_wei, 4),
        "per_class": {
            lbl: {"precision": round(float(prec[k]),4),
                  "recall":    round(float(rec[k]),4),
                  "f1":        round(float(f1[k]),4)}
            for k, lbl in enumerate(LABEL_NAMES)
        },
        "pred_dist": cal_test_pd,
    }
    out = f"/Users/yuvrajpratapsingh/medical_rag/results/calibrated_eval_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved -> {out}")
    return result


if __name__ == "__main__":
    run(n_val=200, n_test=150)
