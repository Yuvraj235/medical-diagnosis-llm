# -*- coding: utf-8 -*-
"""
models/pubmedbert_classifier.py
Fine-tune PubMedBERT (110M) as a 3-class sequence classifier for PubMedQA.

Unlike the generative BioGPT approach (next-token probability scoring),
this trains a direct classification head → typically 72–78% accuracy
on PubMedQA vs ~65% for generative next-token approaches.

Model: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
Task:  [CLS] question [SEP] context [SEP] → {yes, no, maybe}
Train: 700 samples, ~25 min on Apple Silicon CPU
Goal:  70%+ accuracy on 150 test samples

Run:
    python models/pubmedbert_classifier.py
"""
import sys, os, json, time, logging, random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, MODEL_DIR, LABEL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BERT_MODEL  = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
SAVE_PATH   = os.path.join(MODEL_DIR, "pubmedbert_classifier")
MAX_LEN     = 256   # 4x less attention memory than 512 (O(n²)) → no swap
BATCH_SIZE  = 8
GRAD_ACCUM  = 1
LR          = 3e-5
N_EPOCHS    = 10
WARMUP_FRAC = 0.1
PATIENCE    = 4
LOG_EVERY   = 10
SEED        = 42

LABEL2ID = {l: i for i, l in enumerate(LABEL_NAMES)}   # yes=0, no=1, maybe=2
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
CLASS_WEIGHTS = [1.0, 2.0, 6.0]   # yes, no, maybe — stronger penalty for rare classes


def build_input(rec):
    """Build (text_a, text_b) for BERT [CLS] q [SEP] ctx [SEP] format."""
    q   = rec.get("question", "").strip()
    ctx = rec.get("context", "").strip()
    ctx = " ".join(ctx.split())[:2000]      # normalise whitespace, cap length
    return q, ctx


def encode_batch(tokenizer, records):
    qs, ctxs, labels = [], [], []
    for rec in records:
        q, ctx = build_input(rec)
        qs.append(q)
        ctxs.append(ctx)
        labels.append(LABEL2ID[rec["final_decision"]])
    enc = tokenizer(
        qs, ctxs,
        max_length=MAX_LEN, truncation=True, padding=True,
        return_tensors="pt"
    )
    return enc, labels


def quick_eval(model, tokenizer, val_recs, device):
    import torch
    from sklearn.metrics import f1_score, accuracy_score
    model.eval()
    trues, preds = [], []
    pred_dist    = {l: 0 for l in LABEL_NAMES}
    per_class    = {l: {"correct": 0, "total": 0} for l in LABEL_NAMES}

    for rec in val_recs:
        q, ctx = build_input(rec)
        enc    = tokenizer(q, ctx, max_length=MAX_LEN, truncation=True,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits[0]
        pred = ID2LABEL[int(logits.argmax())]
        lbl  = rec["final_decision"]
        trues.append(lbl); preds.append(pred)
        pred_dist[pred] += 1
        per_class[lbl]["total"] += 1
        if pred == lbl:
            per_class[lbl]["correct"] += 1

    model.train()
    acc    = accuracy_score(trues, preds)
    f1_mac = f1_score(trues, preds, labels=LABEL_NAMES, average="macro",
                      zero_division=0)
    pc_str = "  ".join(
        f"{l}={per_class[l]['correct']}/{per_class[l]['total']}"
        for l in LABEL_NAMES)
    logger.info(f"  eval: acc={acc:.3f}  f1_mac={f1_mac:.3f}  "
                f"preds={pred_dist}  [{pc_str}]")
    return f1_mac   # F1-macro for checkpoint selection (handles class imbalance)


def run():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}  LR={LR}  epochs={N_EPOCHS}  batch={BATCH_SIZE}")

    # ── Load data ────────────────────────────────────────────────────────────
    with open(os.path.join(DATA_DIR, "train.json")) as f: train_all = json.load(f)
    with open(os.path.join(DATA_DIR, "val.json"))   as f: val_all   = json.load(f)
    with open(os.path.join(DATA_DIR, "test.json"))  as f: test_all  = json.load(f)

    label_dist = {l: sum(1 for r in train_all if r["final_decision"]==l)
                  for l in LABEL_NAMES}
    logger.info(f"Train: {len(train_all)}  dist={label_dist}")
    logger.info(f"Val:   {len(val_all)}  Test: {len(test_all)}")

    # ── Load model ───────────────────────────────────────────────────────────
    logger.info(f"Loading {BERT_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_params/1e6:.1f}M")

    # Baseline (random init classifier head) — expect ~33%
    logger.info("Baseline eval (before fine-tuning):")
    _ = quick_eval(model, tokenizer, val_all[:30], device)
    logger.info("  (Note: checkpoint selection uses F1-macro, not accuracy)")

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = (len(train_all) // BATCH_SIZE) * N_EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    cw_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(device)
    loss_fn   = torch.nn.CrossEntropyLoss(weight=cw_tensor)

    # ── Training loop ────────────────────────────────────────────────────────
    best_acc     = 0.0
    patience_cnt = 0
    history      = {"epoch": [], "loss": [], "val_f1mac": []}
    t_start      = time.time()
    os.makedirs(SAVE_PATH, exist_ok=True)

    for epoch in range(N_EPOCHS):
        random.shuffle(train_all)
        run_loss  = 0.0
        n_batches = 0
        logger.info(f"\n── Epoch {epoch+1}/{N_EPOCHS} ──")

        for b_start in range(0, len(train_all), BATCH_SIZE):
            batch = train_all[b_start: b_start + BATCH_SIZE]
            enc, label_ids = encode_batch(tokenizer, batch)
            enc = {k: v.to(device) for k, v in enc.items()}
            label_tensor = torch.tensor(label_ids, dtype=torch.long).to(device)

            logits = model(**enc).logits
            loss   = loss_fn(logits, label_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            run_loss  += loss.item()
            n_batches += 1

            if n_batches % LOG_EVERY == 0:
                logger.info(f"  batch {n_batches}/{len(train_all)//BATCH_SIZE}"
                            f"  loss={run_loss/n_batches:.4f}")

        avg_loss = run_loss / max(n_batches, 1)
        val_f1   = quick_eval(model, tokenizer, val_all, device)  # returns F1-macro
        elapsed  = (time.time() - t_start) / 60
        logger.info(f"  Epoch {epoch+1}: loss={avg_loss:.4f}  val_f1mac={val_f1:.3f}"
                    f"  ({elapsed:.1f} min elapsed)")

        history["epoch"].append(epoch + 1)
        history["loss"].append(round(avg_loss, 4))
        history["val_f1mac"].append(round(val_f1, 4))

        if val_f1 > best_acc:    # best_acc now tracks best F1-macro
            best_acc     = val_f1
            patience_cnt = 0
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            logger.info(f"  *** New best val_f1mac={val_f1:.3f} — saved ***")
        else:
            patience_cnt += 1
            logger.info(f"  No improvement ({patience_cnt}/{PATIENCE})")
            if patience_cnt >= PATIENCE:
                logger.info("  Early stopping.")
                break

    # ── Final test evaluation ────────────────────────────────────────────────
    logger.info("\nLoading best checkpoint for test eval ...")
    model_best = AutoModelForSequenceClassification.from_pretrained(SAVE_PATH).to(device)
    model_best.eval()

    true_labels, pred_labels = [], []
    pred_dist  = {l: 0 for l in LABEL_NAMES}
    latencies  = []

    for i, rec in enumerate(test_all):
        q, ctx  = build_input(rec)
        lbl     = rec["final_decision"]
        true_labels.append(lbl)
        t0      = time.time()
        enc     = tokenizer(q, ctx, max_length=MAX_LEN, truncation=True,
                            return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model_best(**enc).logits[0]
        pred    = ID2LABEL[int(logits.argmax())]
        pred_labels.append(pred)
        pred_dist[pred] += 1
        latencies.append((time.time() - t0) * 1000)
        if (i + 1) % 15 == 0 or i == 0:
            logger.info(f"  test [{i+1}/{len(test_all)}] true={lbl} pred={pred}")

    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_recall_fscore_support)
    acc    = accuracy_score(true_labels, pred_labels)
    f1_mac = f1_score(true_labels, pred_labels, labels=LABEL_NAMES,
                      average="macro", zero_division=0)
    f1_wei = f1_score(true_labels, pred_labels, labels=LABEL_NAMES,
                      average="weighted", zero_division=0)
    prec, rec, f1, sup = precision_recall_fscore_support(
        true_labels, pred_labels, labels=LABEL_NAMES, zero_division=0)

    print("\n" + "=" * 64)
    print("  PubMedBERT Fine-tuned Classifier (sequence classification)")
    print("=" * 64)
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1 Macro   : {f1_mac:.4f}")
    print(f"  F1 Weighted: {f1_wei:.4f}")
    for k, lbl in enumerate(LABEL_NAMES):
        print(f"    {lbl.upper():5s}  P={prec[k]:.3f}  R={rec[k]:.3f}"
              f"  F1={f1[k]:.3f}  n={sup[k]}")
    print(f"  Pred dist  : {pred_dist}")
    print(f"  Avg latency: {np.mean(latencies):.0f} ms")
    print("=" * 64)
    print(f"\n  vs Zero-shot BioGPT correct format: 65.3%")
    print(f"  PubMedBERT classifier delta: {(acc - 0.6533)*100:+.1f}pp")
    print(f"  Best val F1-macro (checkpoint selection): {best_acc:.3f}")
    print(f"  Total training time: {(time.time()-t_start)/60:.1f} min")

    result = {
        "method": "pubmedbert_sequence_classifier",
        "model":  BERT_MODEL,
        "n_train": len(train_all),
        "n_test":  len(test_all),
        "best_val_acc": round(best_acc, 4),
        "accuracy":    round(acc, 4),
        "f1_macro":    round(f1_mac, 4),
        "f1_weighted": round(f1_wei, 4),
        "per_class": {
            lbl: {"precision": round(float(prec[k]), 4),
                  "recall":    round(float(rec[k]),  4),
                  "f1":        round(float(f1[k]),   4)}
            for k, lbl in enumerate(LABEL_NAMES)
        },
        "pred_dist":      pred_dist,
        "avg_latency_ms": round(float(np.mean(latencies)), 1),
        "history":        history,
    }
    ts  = int(time.time())
    out = f"/Users/yuvrajpratapsingh/medical_rag/results/pubmedbert_clf_{ts}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved -> {out}")
    with open(os.path.join(MODEL_DIR, "pubmedbert_clf_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return result


if __name__ == "__main__":
    run()
