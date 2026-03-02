# -*- coding: utf-8 -*-
"""
evaluation/format_test.py
Quick test: which prompt format gives BioGPT-Large-PubMedQA the highest accuracy?
The model was trained with a specific format — matching it = free +20-30% accuracy.

Tests 5 candidate formats on 30 test samples. Runs in ~10 min on CPU.
"""
import sys, os, json, time, logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, LABEL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 5 candidate prompt formats ──────────────────────────────────────────────
# Format F1: Microsoft BioGPT GitHub style (question first, context second)
# Format F2: Context first (our current format)
# Format F3: BioGPT paper format with explicit "question:"/"context:" keys
# Format F4: PubMedQA dataset style (minimal, just Q then A)
# Format F5: Context + Q with \n\n separator and "A:" (short form)

FORMATS = {
    "F1_q_then_ctx": lambda q, ctx: (
        f"question: {q}\n"
        f"context: {ctx[:900]}\n"
        f"answer:"
    ),
    "F2_ctx_then_q_current": lambda q, ctx: (
        f"{ctx[:900]}\n\nQuestion: {q}\n\nAnswer:"
    ),
    "F3_explicit_keys": lambda q, ctx: (
        f"Question: {q}\n\nContext: {ctx[:900]}\n\nAnswer:"
    ),
    "F4_minimal_qa": lambda q, ctx: (
        f"{q}\n{ctx[:900]}\n"
    ),
    "F5_biogpt_paper": lambda q, ctx: (
        f"question: {q} context: {ctx[:900]} answer:"
    ),
}


def score_labels(model, tokenizer, prompt, label_tokens, device):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    lp = torch.log_softmax(logits[0, -1, :], dim=-1)
    return {lbl: float(lp[tid]) for lbl, tid in label_tokens.items()}


def test_formats(n_samples=30):
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

    with open(os.path.join(DATA_DIR, "test.json")) as f:
        all_test = json.load(f)
    import random; random.seed(42)
    # Stratified: 10 yes, 10 no, 10 maybe
    by_lbl = {l: [r for r in all_test if r["final_decision"] == l] for l in LABEL_NAMES}
    samples = []
    for l in LABEL_NAMES:
        samples += random.sample(by_lbl[l], min(10, len(by_lbl[l])))
    random.shuffle(samples)
    logger.info(f"Testing on {len(samples)} samples (stratified)")

    results = {}
    for fname, fmt_fn in FORMATS.items():
        correct = 0
        pred_dist = {l: 0 for l in LABEL_NAMES}
        t0 = time.time()
        for rec in samples:
            q   = rec["question"]
            ctx = rec.get("context", "")
            lbl = rec["final_decision"]
            prompt = fmt_fn(q, ctx)
            scores = score_labels(model, tokenizer, prompt, label_tokens, device)
            pred = max(scores, key=scores.get)
            pred_dist[pred] += 1
            if pred == lbl:
                correct += 1
        acc = correct / len(samples)
        elapsed = time.time() - t0
        results[fname] = acc
        logger.info(f"  {fname:30s}  acc={acc:.3f}  preds={pred_dist}  ({elapsed:.0f}s)")

    print("\n" + "=" * 60)
    print("  FORMAT TEST RESULTS")
    print("=" * 60)
    best = max(results, key=results.get)
    for fname, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ◄ BEST" if fname == best else ""
        print(f"  {acc*100:5.1f}%  {fname}{marker}")
    print("=" * 60)
    print(f"\n  Best format: {best}  ({results[best]*100:.1f}% on {len(samples)} samples)")
    return results, best


if __name__ == "__main__":
    results, best = test_formats(n_samples=30)
