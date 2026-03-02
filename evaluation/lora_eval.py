# -*- coding: utf-8 -*-
"""
evaluation/lora_eval.py
Evaluate BioGPT-Large-PubMedQA WITH the LoRA adapter applied.

Runs the same constrained next-token scoring as real_eval.py, but
loads the PEFT adapter from models/checkpoints/lora_best/ first.

Run:
    python evaluation/lora_eval.py
"""
import sys, os, json, time, random, logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, RESULTS_DIR, MODEL_DIR, LABEL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.makedirs(RESULTS_DIR, exist_ok=True)

LORA_PATH  = os.path.join(MODEL_DIR, "lora_best")
MODEL_NAME = "microsoft/BioGPT-Large-PubMedQA"


def build_prompt(question: str, context: str) -> str:
    """Same format used during LoRA fine-tuning (fast_lora_finetune.py)."""
    ctx = context[:600]
    return f"{ctx}\n\nQuestion: {question}\n\nAnswer:"


def score_labels(model, tokenizer, prompt: str, label_tokens: dict, device: str):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    last_logits = logits[0, -1, :]
    log_probs   = torch.log_softmax(last_logits, dim=-1)
    return {lbl: float(log_probs[tid]) for lbl, tid in label_tokens.items()}


def evaluate_split(model, tokenizer, label_tokens, device,
                   test_data, mode, retriever=None):
    true_labels, pred_labels = [], []
    latencies, retrieval_scores = [], []

    for i, rec in enumerate(test_data):
        q   = rec["question"]
        lbl = rec.get("final_decision", "maybe")
        true_labels.append(lbl)

        t0 = time.time()

        if mode == "orig_ctx":
            ctx_str = rec.get("context", "")
            retrieval_scores.append(1.0)
        else:
            ctx_str, chunks = retriever.retrieve_and_format(q, top_k=5)
            if chunks:
                retrieval_scores.append(chunks[0].get("score", 0.0))

        prompt = build_prompt(q, ctx_str)
        scores = score_labels(model, tokenizer, prompt, label_tokens, device)
        pred   = max(scores, key=scores.get)
        pred_labels.append(pred)

        lat_ms = (time.time() - t0) * 1000
        latencies.append(lat_ms)

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"  [{mode}] [{i+1}/{len(test_data)}] "
                        f"true={lbl:5s}  pred={pred:5s}  "
                        f"lat={lat_ms:.0f}ms  {scores}")

    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_recall_fscore_support)
    accuracy    = accuracy_score(true_labels, pred_labels)
    f1_macro    = f1_score(true_labels, pred_labels, labels=LABEL_NAMES,
                           average="macro", zero_division=0)
    f1_weighted = f1_score(true_labels, pred_labels, labels=LABEL_NAMES,
                           average="weighted", zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=LABEL_NAMES, zero_division=0)

    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    avg_ret = float(np.mean(retrieval_scores)) if retrieval_scores else 0.0
    hit1    = float(np.mean([1 if s > 0.3 else 0 for s in retrieval_scores]))
    mrr     = float(np.mean([1.0 if s > 0.3 else 0.0 for s in retrieval_scores]))

    return {
        "mode":        mode,
        "accuracy":    round(accuracy, 4),
        "f1_macro":    round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "per_class": {
            lbl: {"precision": round(float(prec[k]), 4),
                  "recall":    round(float(rec[k]),  4),
                  "f1":        round(float(f1[k]),   4),
                  "support":   int(support[k])}
            for k, lbl in enumerate(LABEL_NAMES)
        },
        "retrieval": {
            "hit_rate_at_1":       round(hit1, 4),
            "avg_retrieval_score": round(avg_ret, 4),
            "mrr":                 round(mrr, 4),
        },
        "latency_ms": {
            "mean": round(avg_lat, 1),
            "p95":  round(p95_lat, 1),
        },
        "label_distribution": {
            "true": {l: true_labels.count(l) for l in LABEL_NAMES},
            "pred": {l: pred_labels.count(l) for l in LABEL_NAMES},
        },
    }


def _print_results(title, r):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)
    print(f"  Accuracy   : {r['accuracy']:.4f}  ({r['accuracy']*100:.1f}%)")
    print(f"  F1 Macro   : {r['f1_macro']:.4f}")
    print(f"  F1 Weighted: {r['f1_weighted']:.4f}")
    for lbl in LABEL_NAMES:
        pc = r["per_class"][lbl]
        print(f"    {lbl.upper():5s}  P={pc['precision']:.3f}  "
              f"R={pc['recall']:.3f}  F1={pc['f1']:.3f}  n={pc['support']}")
    rt = r["retrieval"]
    print(f"  Hit Rate@1 : {rt['hit_rate_at_1']:.4f}")
    print(f"  Avg Retr.  : {rt['avg_retrieval_score']:.4f}")
    lt = r["latency_ms"]
    print(f"  Avg Latency: {lt['mean']:.0f} ms")
    print(f"  P95 Latency: {lt['p95']:.0f} ms")
    print("=" * 64)


def run_lora_evaluation(n_samples=150):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    device = "cpu"

    # ── Check LoRA adapter exists ──────────────────────────────────────────────
    if not os.path.isdir(LORA_PATH):
        logger.error(f"LoRA adapter not found at {LORA_PATH}. "
                     f"Run: python models/fast_lora_finetune.py first.")
        sys.exit(1)

    logger.info(f"Device: {device}")
    logger.info(f"LoRA adapter: {LORA_PATH}")

    # ── Load test data ─────────────────────────────────────────────────────────
    test_path = os.path.join(DATA_DIR, "test.json")
    with open(test_path) as f:
        all_test = json.load(f)
    random.seed(42)
    test_data = all_test[:n_samples] if n_samples >= len(all_test) \
                else random.sample(all_test, n_samples)
    logger.info(f"Test samples: {len(test_data)}")

    # ── Load base model + LoRA adapter ─────────────────────────────────────────
    logger.info(f"Loading base model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32, low_cpu_mem_usage=True)
    logger.info(f"Loading LoRA adapter from {LORA_PATH} ...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.to(device)
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model+LoRA loaded: {params_m:.0f}M params total")

    # ── Label token IDs ────────────────────────────────────────────────────────
    label_tokens = {}
    for lbl in LABEL_NAMES:
        candidates = (tokenizer.encode(" " + lbl, add_special_tokens=False) or
                      tokenizer.encode(lbl,       add_special_tokens=False))
        label_tokens[lbl] = candidates[0]
    logger.info(f"Label token IDs: {label_tokens}")

    # ── Load FAISS retriever ───────────────────────────────────────────────────
    logger.info("Loading FAISS retriever ...")
    from retrieval.retriever import MedicalRetriever
    retriever = MedicalRetriever()
    retriever.initialize()
    logger.info("Retriever ready")

    # ── Evaluate: original context ─────────────────────────────────────────────
    logger.info("\n=== EVALUATING (LoRA): original context ===")
    res_orig = evaluate_split(model, tokenizer, label_tokens, device,
                              test_data, mode="orig_ctx")
    _print_results("BioGPT-Large + LoRA + Original Context", res_orig)

    # ── Evaluate: RAG ──────────────────────────────────────────────────────────
    logger.info("\n=== EVALUATING (LoRA): RAG pipeline ===")
    res_rag  = evaluate_split(model, tokenizer, label_tokens, device,
                              test_data, mode="rag", retriever=retriever)
    _print_results("BioGPT-Large + LoRA + RAG (FAISS)", res_rag)

    # ── Save ───────────────────────────────────────────────────────────────────
    results = {
        "model":      MODEL_NAME,
        "lora_path":  LORA_PATH,
        "n_samples":  len(test_data),
        "device":     device,
        "orig_ctx":   res_orig,
        "rag":        res_rag,
    }
    ts  = int(time.time())
    out = os.path.join(RESULTS_DIR, f"lora_eval_{ts}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved -> {out}")
    return results


if __name__ == "__main__":
    run_lora_evaluation(n_samples=150)
