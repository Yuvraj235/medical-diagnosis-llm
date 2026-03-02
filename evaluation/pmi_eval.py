# -*- coding: utf-8 -*-
"""
PMI (Pointwise Mutual Information) calibration with the correct BioGPT format.
Adjusts for the model's class prior bias so MAYBE can be predicted.

PMI formula:  calibrated(label|ctx) = log P(label|ctx) - log P(label|null)
Where null = empty/neutral context.
"""
import sys, os, json, time, re, logging
import numpy as np
from pathlib import Path

sys.path.insert(0, '/Users/yuvrajpratapsingh/medical_rag')
from config import DATA_DIR, LABEL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_PREFIX = "the answer to the question given the context is "


def build_prompt(q, ctx, max_ctx=1800):
    ctx_c = re.sub(r'\s+', ' ', ctx.strip())[:max_ctx]
    return f"question: {q.strip()} context: {ctx_c} {TARGET_PREFIX}"


def build_null_prompt(q):
    """Neutral context — reveals the model's class prior."""
    return f"question: {q.strip()} context: N/A {TARGET_PREFIX}"


def get_logprobs(model, tokenizer, prompt, label_tokens, device):
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

    # ── Step 1: Estimate class priors via null-context prompts ────────────────
    logger.info("Estimating class priors (null context)...")
    prior_samples = random.sample(test_data, min(20, len(test_data)))
    prior_acc = {lbl: [] for lbl in LABEL_NAMES}
    for rec in prior_samples:
        null_p = build_null_prompt(rec["question"])
        lp = get_logprobs(model, tokenizer, null_p, label_tokens, device)
        for lbl in LABEL_NAMES:
            prior_acc[lbl].append(lp[lbl])
    prior = {lbl: float(np.mean(prior_acc[lbl])) for lbl in LABEL_NAMES}
    logger.info(f"  Class priors: yes={prior['yes']:.3f}  "
                f"no={prior['no']:.3f}  maybe={prior['maybe']:.3f}")

    # ── Step 2: PMI-calibrated evaluation ────────────────────────────────────
    logger.info("PMI-calibrated evaluation...")
    true_labels, pred_labels_raw, pred_labels_pmi = [], [], []
    pred_dist_raw  = {l: 0 for l in LABEL_NAMES}
    pred_dist_pmi  = {l: 0 for l in LABEL_NAMES}

    for i, rec in enumerate(test_data):
        q, ctx, lbl = rec["question"], rec.get("context",""), rec["final_decision"]
        true_labels.append(lbl)

        prompt = build_prompt(q, ctx)
        scores = get_logprobs(model, tokenizer, prompt, label_tokens, device)

        # Raw prediction
        pred_raw = max(scores, key=scores.get)
        pred_labels_raw.append(pred_raw)
        pred_dist_raw[pred_raw] += 1

        # PMI prediction: subtract prior
        pmi_scores = {lbl: scores[lbl] - prior[lbl] for lbl in LABEL_NAMES}
        pred_pmi   = max(pmi_scores, key=pmi_scores.get)
        pred_labels_pmi.append(pred_pmi)
        pred_dist_pmi[pred_pmi] += 1

        if (i+1) % 10 == 0 or i == 0:
            logger.info(f"  [{i+1}/{n_samples}] true={lbl:5s} "
                        f"raw={pred_raw:5s} pmi={pred_pmi:5s} "
                        f"raw_scores={{{','.join(f'{k}:{v:.2f}' for k,v in scores.items())}}} "
                        f"pmi_scores={{{','.join(f'{k}:{v:.2f}' for k,v in pmi_scores.items())}}}")

    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    def report(name, true_l, pred_l, pred_d):
        acc    = accuracy_score(true_l, pred_l)
        f1_mac = f1_score(true_l, pred_l, labels=LABEL_NAMES, average="macro", zero_division=0)
        f1_wei = f1_score(true_l, pred_l, labels=LABEL_NAMES, average="weighted", zero_division=0)
        prec, rec, f1, sup = precision_recall_fscore_support(
            true_l, pred_l, labels=LABEL_NAMES, zero_division=0)
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print(f"  Accuracy   : {acc:.4f}  ({acc*100:.1f}%)")
        print(f"  F1 Macro   : {f1_mac:.4f}")
        print(f"  F1 Weighted: {f1_wei:.4f}")
        for k, lbl in enumerate(LABEL_NAMES):
            print(f"    {lbl.upper():5s}  P={prec[k]:.3f}  R={rec[k]:.3f}  "
                  f"F1={f1[k]:.3f}  n={sup[k]}")
        print(f"  Pred dist  : {pred_d}")
        return acc, f1_mac, {"per_class": {lbl: {"p": round(float(prec[k]),4),
                                                  "r": round(float(rec[k]),4),
                                                  "f1": round(float(f1[k]),4)}
                                           for k, lbl in enumerate(LABEL_NAMES)}}

    acc_raw, f1_raw, pc_raw = report(
        "RAW (correct format, no calibration)", true_labels, pred_labels_raw, pred_dist_raw)
    acc_pmi, f1_pmi, pc_pmi = report(
        "PMI CALIBRATED (correct format + PMI)", true_labels, pred_labels_pmi, pred_dist_pmi)

    # Save
    result = {
        "prior": prior,
        "raw":   {"accuracy": round(acc_raw,4), "f1_macro": round(f1_raw,4),
                  "pred_dist": pred_dist_raw, **pc_raw},
        "pmi":   {"accuracy": round(acc_pmi,4), "f1_macro": round(f1_pmi,4),
                  "pred_dist": pred_dist_pmi, **pc_pmi},
    }
    out = f"/Users/yuvrajpratapsingh/medical_rag/results/pmi_eval_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved -> {out}")
    return result


if __name__ == "__main__":
    run(n_samples=150)
