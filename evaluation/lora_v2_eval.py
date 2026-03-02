# -*- coding: utf-8 -*-
"""
evaluation/lora_v2_eval.py
Evaluate LoRA v2 fine-tuned BioGPT-Large-PubMedQA on 150 PubMedQA test samples.

Uses the CORRECT Microsoft training format:
  question: {q} context: {ctx} the answer to the question given the context is

Run:
    python evaluation/lora_v2_eval.py
"""
import sys, os, json, time, re, logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, MODEL_DIR, LABEL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LORA_V2_PATH = os.path.join(MODEL_DIR, "lora_v2_best")
BASE_MODEL    = "microsoft/BioGPT-Large-PubMedQA"
TARGET_PREFIX = "the answer to the question given the context is "


def build_prompt(q, ctx, max_ctx=1800):
    ctx_c = re.sub(r'\s+', ' ', ctx.strip())[:max_ctx]
    return f"question: {q.strip()} context: {ctx_c} {TARGET_PREFIX}"


def score_labels(model, tokenizer, prompt, label_tokens, device):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    lp = torch.log_softmax(logits[0, -1, :], dim=-1)
    return {lbl: float(lp[tid]) for lbl, tid in label_tokens.items()}


def run(n_samples=150):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    device = "cpu"

    # Check adapter exists
    if not os.path.isdir(LORA_V2_PATH):
        logger.error(f"LoRA v2 adapter not found at {LORA_V2_PATH} — is training complete?")
        return None

    logger.info(f"Loading base model {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float32, low_cpu_mem_usage=True)

    logger.info(f"Loading LoRA v2 adapter from {LORA_V2_PATH} ...")
    model = PeftModel.from_pretrained(base_model, LORA_V2_PATH)
    model.eval()
    logger.info("Model + LoRA v2 adapter loaded")

    label_tokens = {}
    for lbl in LABEL_NAMES:
        ids = (tokenizer.encode(" " + lbl, add_special_tokens=False) or
               tokenizer.encode(lbl, add_special_tokens=False))
        label_tokens[lbl] = ids[0]
    logger.info(f"Label tokens: {label_tokens}")

    with open(os.path.join(DATA_DIR, "test.json")) as f:
        all_test = json.load(f)
    import random; random.seed(42)
    test_data = all_test[:n_samples] if n_samples >= len(all_test) \
                else random.sample(all_test, n_samples)
    logger.info(f"Evaluating on {len(test_data)} test samples")

    true_labels, pred_labels = [], []
    pred_dist = {l: 0 for l in LABEL_NAMES}
    latencies = []

    for i, rec in enumerate(test_data):
        q   = rec["question"]
        ctx = rec.get("context", "")
        lbl = rec["final_decision"]
        true_labels.append(lbl)

        t0 = time.time()
        prompt = build_prompt(q, ctx)
        scores = score_labels(model, tokenizer, prompt, label_tokens, device)
        pred   = max(scores, key=scores.get)
        pred_labels.append(pred)
        pred_dist[pred] += 1
        lat = (time.time() - t0) * 1000
        latencies.append(lat)

        if (i+1) % 10 == 0 or i == 0:
            logger.info(f"  [{i+1}/{len(test_data)}] true={lbl:5s} pred={pred:5s} "
                        f"lat={lat:.0f}ms scores={{{', '.join(f'{k}:{v:.2f}' for k,v in scores.items())}}}")

    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    acc    = accuracy_score(true_labels, pred_labels)
    f1_mac = f1_score(true_labels, pred_labels, labels=LABEL_NAMES, average="macro", zero_division=0)
    f1_wei = f1_score(true_labels, pred_labels, labels=LABEL_NAMES, average="weighted", zero_division=0)
    prec, rec, f1, sup = precision_recall_fscore_support(
        true_labels, pred_labels, labels=LABEL_NAMES, zero_division=0)

    print("\n" + "="*64)
    print("  LoRA v2 + Correct Microsoft Format")
    print("="*64)
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1 Macro   : {f1_mac:.4f}")
    print(f"  F1 Weighted: {f1_wei:.4f}")
    for k, lbl in enumerate(LABEL_NAMES):
        print(f"    {lbl.upper():5s}  P={prec[k]:.3f}  R={rec[k]:.3f}  "
              f"F1={f1[k]:.3f}  n={sup[k]}")
    print(f"  Pred dist  : {pred_dist}")
    print(f"  Avg latency: {np.mean(latencies):.0f} ms")
    print("="*64)

    # Compare vs format-only baseline
    print(f"\n  vs Correct Format (no LoRA): 65.3%  F1=0.480")
    delta = acc - 0.6533
    print(f"  LoRA v2 delta: {delta:+.1%} ({'+' if delta>=0 else ''}{delta*100:.1f}pp)")

    result = {
        "adapter": LORA_V2_PATH,
        "base_model": BASE_MODEL,
        "format": "microsoft_correct_format",
        "n_samples": len(test_data),
        "accuracy":    round(acc, 4),
        "f1_macro":    round(f1_mac, 4),
        "f1_weighted": round(f1_wei, 4),
        "per_class": {
            lbl: {"precision": round(float(prec[k]),4),
                  "recall":    round(float(rec[k]),4),
                  "f1":        round(float(f1[k]),4)}
            for k, lbl in enumerate(LABEL_NAMES)
        },
        "pred_dist": pred_dist,
        "avg_latency_ms": round(float(np.mean(latencies)), 1),
    }
    out = f"/Users/yuvrajpratapsingh/medical_rag/results/lora_v2_eval_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved -> {out}")
    return result


if __name__ == "__main__":
    run(n_samples=150)
