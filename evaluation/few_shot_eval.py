# -*- coding: utf-8 -*-
"""
evaluation/few_shot_eval.py
Few-shot in-context learning with BioGPT-Large-PubMedQA.

Adds 3 labeled examples (one per class: yes / no / maybe) before each test question.
This "primes" the model to predict all three classes, including MAYBE.

Zero training required — runs in ~30 min on CPU.

Run:
    python evaluation/few_shot_eval.py
"""
import sys, os, json, time, re, logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, LABEL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_PREFIX = "the answer to the question given the context is "


def build_few_shot_prompt(q, ctx, examples, max_ctx=1200):
    """
    Build a few-shot prompt with 3 labeled examples (one per class) followed by test question.
    Examples are SHORT (400 chars) to keep total sequence manageable.
    Test question uses full context (1200 chars).
    """
    prompt = ""
    for ex in examples:
        ex_ctx = re.sub(r'\s+', ' ', ex["context"].strip())[:400]
        ex_q   = ex["question"].strip()
        ex_lbl = ex["final_decision"]
        prompt += f"question: {ex_q} context: {ex_ctx} {TARGET_PREFIX}{ex_lbl}.\n\n"

    # Test question (no label — we score it)
    test_ctx = re.sub(r'\s+', ' ', ctx.strip())[:max_ctx]
    prompt += f"question: {q.strip()} context: {test_ctx} {TARGET_PREFIX}"
    return prompt


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

    # Load data
    with open(os.path.join(DATA_DIR, "train.json")) as f: train_all = json.load(f)
    with open(os.path.join(DATA_DIR, "test.json"))  as f: all_test  = json.load(f)

    import random; random.seed(42)

    # Select diverse, clear few-shot examples (one per class)
    # Pick examples where the correct answer has a clear, decisive score
    by_lbl = {l: [r for r in train_all if r["final_decision"] == l] for l in LABEL_NAMES}
    # Use first example of each class (they're consistent across runs with seed=42)
    few_shot_examples = [by_lbl["yes"][0], by_lbl["no"][0], by_lbl["maybe"][0]]
    logger.info(f"Few-shot examples: yes/no/maybe — one each from train set")

    test_data = all_test[:n_samples] if n_samples >= len(all_test) \
                else random.sample(all_test, n_samples)
    logger.info(f"Evaluating {len(test_data)} test samples with 3-shot prompting")

    true_labels, pred_labels = [], []
    pred_dist = {l: 0 for l in LABEL_NAMES}
    latencies = []

    for i, rec in enumerate(test_data):
        q   = rec["question"]
        ctx = rec.get("context", "")
        lbl = rec["final_decision"]
        true_labels.append(lbl)

        # Exclude the test sample's question from few-shot examples
        examples = [e for e in few_shot_examples if e["question"] != q][:3]
        if len(examples) < 3:
            # Fallback: use different examples
            examples = [by_lbl["yes"][1], by_lbl["no"][1], by_lbl["maybe"][1]]

        t0 = time.time()
        prompt = build_few_shot_prompt(q, ctx, examples)
        scores = score_labels(model, tokenizer, prompt, label_tokens, device)
        pred   = max(scores, key=scores.get)
        pred_labels.append(pred)
        pred_dist[pred] += 1
        lat = (time.time() - t0) * 1000
        latencies.append(lat)

        if (i+1) % 10 == 0 or i == 0:
            logger.info(f"  [{i+1}/{len(test_data)}] true={lbl:5s} pred={pred:5s} "
                        f"lat={lat:.0f}ms "
                        f"scores={{{', '.join(f'{k}:{v:.2f}' for k,v in scores.items())}}}")

    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    acc    = accuracy_score(true_labels, pred_labels)
    f1_mac = f1_score(true_labels, pred_labels, labels=LABEL_NAMES, average="macro", zero_division=0)
    f1_wei = f1_score(true_labels, pred_labels, labels=LABEL_NAMES, average="weighted", zero_division=0)
    prec, rec, f1, sup = precision_recall_fscore_support(
        true_labels, pred_labels, labels=LABEL_NAMES, zero_division=0)

    print("\n" + "="*64)
    print("  3-Shot In-Context Learning — BioGPT-Large-PubMedQA")
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
    print(f"\n  vs Zero-shot (correct format): 65.3%")
    print(f"  Few-shot delta: {(acc - 0.6533)*100:+.1f}pp")

    result = {
        "method": "3_shot_in_context_learning",
        "n_shots": 3,
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
    out = f"/Users/yuvrajpratapsingh/medical_rag/results/few_shot_eval_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved -> {out}")
    return result


if __name__ == "__main__":
    run(n_samples=150)
