"""
evaluation/run_evaluation.py
Comprehensive quantitative evaluation for the Medical RAG Pipeline.

Metrics computed
────────────────
Classification : Accuracy, F1-macro, F1-weighted, per-class P / R / F1
Text generation: BLEU-1/4, ROUGE-1/2/L, BERTScore P/R/F1
RAG quality    : Fluency, Relevance, Coherence, Faithfulness
Safety         : Avg/Max/Rate Toxicity, Bias Rate
Retrieval      : Hit-Rate @1/@5, MRR, Avg Retrieval Score
System         : Avg Latency (ms), P95 Latency (ms)

Outputs
───────
results/eval_TIMESTAMP.json       ← full metrics (use in dissertation)
results/plots_TIMESTAMP/
    evaluation_overview.png       ← 6-panel chart  (Figure 1)
    radar_chart.png               ← radar plot     (Figure 2)
    per_class_f1.png              ← per-class bars (Figure 3)
    latency_dist.png              ← latency hist   (Figure 4)
"""
import sys
import os
import json
import time
import random
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, RESULTS_DIR, LABEL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data class
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvaluationResult:
    # Classification
    accuracy:           float = 0.0
    f1_macro:           float = 0.0
    f1_weighted:        float = 0.0
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall:    Dict[str, float] = field(default_factory=dict)
    per_class_f1:        Dict[str, float] = field(default_factory=dict)
    # Text generation
    bleu_1:      float = 0.0
    bleu_4:      float = 0.0
    rouge_1_f:   float = 0.0
    rouge_2_f:   float = 0.0
    rouge_l_f:   float = 0.0
    bertscore_p: float = 0.0
    bertscore_r: float = 0.0
    bertscore_f1: float = 0.0
    # RAG quality
    fluency_score:      float = 0.0
    relevance_score:    float = 0.0
    coherence_score:    float = 0.0
    faithfulness_score: float = 0.0
    # Safety
    avg_toxicity:  float = 0.0
    max_toxicity:  float = 0.0
    toxicity_rate: float = 0.0
    bias_rate:     float = 0.0
    # Retrieval
    hit_rate_at_1:      float = 0.0
    hit_rate_at_5:      float = 0.0
    mrr:                float = 0.0
    avg_retrieval_score: float = 0.0
    # System
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_samples:  int   = 0
    n_safe:         int   = 0
    n_emergency:    int   = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Individual metric functions
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_bleu(preds: List[str], refs: List[str]) -> Dict[str, float]:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        try: nltk.data.find("tokenizers/punkt")
        except LookupError: nltk.download("punkt", quiet=True)

        sf = SmoothingFunction().method1
        b1, b4 = [], []
        for p, r in zip(preds, refs):
            if not p.strip() or not r.strip(): continue
            p_tok = p.lower().split(); r_tok = [r.lower().split()]
            b1.append(sentence_bleu(r_tok, p_tok, weights=(1,0,0,0), smoothing_function=sf))
            b4.append(sentence_bleu(r_tok, p_tok, weights=(.25,.25,.25,.25), smoothing_function=sf))
        return {"bleu_1": float(np.mean(b1)) if b1 else 0.0,
                "bleu_4": float(np.mean(b4)) if b4 else 0.0}
    except Exception as e:
        logger.warning(f"BLEU failed: {e}")
        return {"bleu_1": 0.0, "bleu_4": 0.0}


def _compute_rouge(preds: List[str], refs: List[str]) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for p, r in zip(preds, refs):
            if not p.strip() or not r.strip(): continue
            s = sc.score(r, p)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        return {"rouge_1_f": float(np.mean(r1)) if r1 else 0.0,
                "rouge_2_f": float(np.mean(r2)) if r2 else 0.0,
                "rouge_l_f": float(np.mean(rl)) if rl else 0.0}
    except Exception as e:
        logger.warning(f"ROUGE failed: {e}")
        return {"rouge_1_f": 0.0, "rouge_2_f": 0.0, "rouge_l_f": 0.0}


def _compute_bertscore(preds: List[str], refs: List[str]) -> Dict[str, float]:
    try:
        from bert_score import score as bscore
        import torch
        device = "mps" if torch.backends.mps.is_available() \
                 else "cuda" if torch.cuda.is_available() else "cpu"
        valid  = [(p, r) for p, r in zip(preds, refs) if p.strip() and r.strip()]
        if not valid:
            return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}
        ps, rs  = zip(*valid)
        P, R, F = bscore(list(ps), list(rs), lang="en",
                         model_type="microsoft/deberta-small",
                         device=device, verbose=False)
        return {"bertscore_p": float(P.mean()),
                "bertscore_r": float(R.mean()),
                "bertscore_f1": float(F.mean())}
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}


def _compute_fluency(answers: List[str]) -> float:
    """
    Fluency via linguistic features: avg sentence length, lexical diversity,
    medical term density, and length adequacy. Range [0, 1].
    """
    import re
    MEDICAL = {
        "patient","treatment","clinical","study","results","efficacy","significant",
        "associated","therapy","disease","risk","evidence","analysis","effect",
        "outcome","reduction","compared","intervention","diagnosis","prognosis",
    }
    scores = []
    for ans in answers:
        if not ans.strip(): scores.append(0.0); continue
        words = ans.lower().split()
        sents = [s for s in re.split(r'[.!?]+', ans) if s.strip()]
        if not words or not sents: scores.append(0.0); continue

        avg_len   = len(words) / len(sents)
        len_score = min(avg_len / 20.0, 1.0)

        ttr       = len(set(words)) / len(words)

        med_d     = sum(1 for w in words if w in MEDICAL)
        med_score = min(med_d / max(len(words), 1), 0.3) / 0.3

        adeq      = (min(len(words) / 50.0, 1.0) if len(words) <= 50
                     else max(0.0, 1.0 - (len(words) - 200) / 200))

        scores.append(min(0.3*len_score + 0.3*ttr + 0.2*med_score + 0.2*adeq, 1.0))
    return float(np.mean(scores)) if scores else 0.0


def _compute_relevance(questions: List[str], answers: List[str], embedder=None) -> float:
    """Cosine similarity between question and answer embeddings. Range [0, 1]."""
    try:
        if embedder is None:
            from embeddings.pubmedbert_embedder import PubMedBERTEmbedder
            embedder = PubMedBERTEmbedder()
        pairs = [(q, a) for q, a in zip(questions, answers) if q.strip() and a.strip()]
        if not pairs: return 0.0
        qs, ans = zip(*pairs)
        q_emb   = embedder.encode(list(qs),  show_progress=False)
        a_emb   = embedder.encode(list(ans), show_progress=False)
        sims    = np.sum(q_emb * a_emb, axis=1)         # dot on normalised vecs
        return float(np.mean(np.clip(sims, 0, 1)))
    except Exception as e:
        logger.warning(f"Relevance failed: {e}"); return 0.0


def _compute_coherence(answers: List[str], embedder=None) -> float:
    """Average cosine similarity between consecutive sentences. Range [0, 1]."""
    import re
    try:
        if embedder is None:
            from embeddings.pubmedbert_embedder import PubMedBERTEmbedder
            embedder = PubMedBERTEmbedder()
        all_s = []
        for ans in answers:
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', ans)
                     if s.strip() and len(s.split()) > 3]
            if len(sents) < 2: all_s.append(0.8); continue
            embs = embedder.encode(sents, show_progress=False)
            sims = [float(np.dot(embs[i], embs[i+1])) for i in range(len(embs)-1)]
            all_s.append(float(np.mean(sims)))
        return float(np.mean(all_s)) if all_s else 0.0
    except Exception as e:
        logger.warning(f"Coherence failed: {e}"); return 0.0


def _compute_faithfulness(answers: List[str], contexts: List[str]) -> float:
    """N-gram overlap between answer tokens and retrieved context tokens. Range [0, 1]."""
    import re
    STOP = {
        "the","a","an","is","are","was","were","be","been","have","has","had",
        "do","does","did","will","would","could","should","may","might","can",
        "to","of","in","for","on","with","at","by","from","as","and","but",
        "or","not","no","this","that","it","its",
    }
    scores = []
    for ans, ctx in zip(answers, contexts):
        if not ans.strip() or not ctx.strip(): scores.append(0.0); continue
        a_w = set(re.findall(r'\b\w+\b', ans.lower())) - STOP
        c_w = set(re.findall(r'\b\w+\b', ctx.lower())) - STOP
        if not a_w: scores.append(0.0); continue
        scores.append(min(len(a_w & c_w) / len(a_w), 1.0))
    return float(np.mean(scores)) if scores else 0.0


def _compute_toxicity(texts: List[str]) -> Dict[str, float]:
    """Toxicity via Detoxify (or keyword fallback). Returns avg/max/rate."""
    if not texts:
        return {"avg_toxicity": 0.0, "max_toxicity": 0.0, "toxicity_rate": 0.0}
    try:
        from detoxify import Detoxify
        model  = Detoxify("original")
        scores = []
        for t in texts:
            if not t.strip(): scores.append(0.0); continue
            scores.append(float(model.predict(t[:512]).get("toxicity", 0.0)))
        arr = np.array(scores)
        return {"avg_toxicity":  float(np.mean(arr)),
                "max_toxicity":  float(np.max(arr)),
                "toxicity_rate": float(np.mean(arr > 0.5))}
    except Exception as e:
        logger.warning(f"Detoxify unavailable ({e}). Using keyword fallback.")
        TOXIC = {"hate","kill","harm","abuse","violent","toxic","dangerous"}
        scores = [len(set(t.lower().split()) & TOXIC) / max(len(t.split()), 1) for t in texts]
        arr = np.array(scores)
        return {"avg_toxicity":  float(np.mean(arr)),
                "max_toxicity":  float(np.max(arr)),
                "toxicity_rate": float(np.mean(arr > 0.5))}


def _compute_retrieval_metrics(results: List[Dict]) -> Dict[str, float]:
    """Hit Rate @1/@5, MRR, avg retrieval score."""
    if not results:
        return {"hit_rate_at_1": 0.0, "hit_rate_at_5": 0.0,
                "mrr": 0.0, "avg_retrieval_score": 0.0}
    h1, h5, rr, avg = [], [], [], []
    for r in results:
        chunks = r.get("retrieved_chunks", [])
        scores = [c.get("score", 0.0) for c in chunks]
        avg.append(np.mean(scores) if scores else 0.0)
        h1.append(1 if chunks and chunks[0].get("score", 0) > 0.5 else 0)
        h5.append(1 if any(c.get("score", 0) > 0.5 for c in chunks[:5]) else 0)
        rr_val = 0.0
        for rank, c in enumerate(chunks[:5], 1):
            if c.get("score", 0) > 0.5: rr_val = 1.0 / rank; break
        rr.append(rr_val)
    return {"hit_rate_at_1":      float(np.mean(h1)),
            "hit_rate_at_5":      float(np.mean(h5)),
            "mrr":                float(np.mean(rr)),
            "avg_retrieval_score": float(np.mean(avg))}


# ═══════════════════════════════════════════════════════════════════════════════
# Plot generation
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_plots(result: EvaluationResult, plots_dir: str, details: List[Dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                         "figure.dpi": 150, "axes.spines.top": False,
                         "axes.spines.right": False})

    # ── Figure 1: 6-panel overview ────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Medical RAG Pipeline — Evaluation Overview",
                 fontsize=16, fontweight="bold", y=1.01)

    def _bar(ax, data: dict, title: str, colors=None):
        ks, vs = list(data.keys()), list(data.values())
        bars = ax.bar(ks, vs, color=colors or "#2196F3")
        ax.set_ylim(0, 1); ax.set_title(title); ax.set_ylabel("Score")
        for b, v in zip(bars, vs):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)

    _bar(axes[0,0],
         {"Accuracy": result.accuracy, "F1 Macro": result.f1_macro,
          "F1 Weighted": result.f1_weighted},
         "Classification Metrics",
         ["#2196F3","#4CAF50","#FF9800"])

    _bar(axes[0,1],
         {"BLEU-1": result.bleu_1, "BLEU-4": result.bleu_4,
          "ROUGE-1": result.rouge_1_f, "ROUGE-2": result.rouge_2_f,
          "ROUGE-L": result.rouge_l_f, "BERTScore": result.bertscore_f1},
         "Text Generation Metrics", "#9C27B0")

    _bar(axes[0,2],
         {"Fluency": result.fluency_score, "Relevance": result.relevance_score,
          "Coherence": result.coherence_score, "Faithfulness": result.faithfulness_score},
         "RAG Quality Metrics",
         ["#00BCD4","#009688","#4CAF50","#8BC34A"])

    saf = {"Avg Toxicity": result.avg_toxicity, "Max Toxicity": result.max_toxicity,
           "Toxicity Rate": result.toxicity_rate, "Bias Rate": result.bias_rate}
    clrs = ["#F44336" if v > 0.3 else "#FF9800" if v > 0.1 else "#4CAF50"
            for v in saf.values()]
    _bar(axes[1,0], saf, "Safety Metrics (lower = better)", clrs)

    _bar(axes[1,1],
         {"Hit Rate @1": result.hit_rate_at_1, "Hit Rate @5": result.hit_rate_at_5,
          "MRR": result.mrr, "Avg Score": result.avg_retrieval_score},
         "Retrieval Metrics", "#FF5722")

    ax = axes[1,2]
    lats = [d.get("latency_ms", 0) for d in details if d.get("latency_ms")]
    if lats:
        ax.hist(lats, bins=15, color="#607D8B", edgecolor="white")
        ax.axvline(result.avg_latency_ms, color="red", linestyle="--",
                   label=f"Mean: {result.avg_latency_ms:.0f} ms")
        ax.axvline(result.p95_latency_ms, color="orange", linestyle="--",
                   label=f"P95:  {result.p95_latency_ms:.0f} ms")
        ax.set_xlabel("Latency (ms)"); ax.set_ylabel("Count")
        ax.set_title("Response Latency Distribution"); ax.legend()
    else:
        ax.text(0.5, 0.5, f"Avg: {result.avg_latency_ms:.0f} ms\nP95: {result.p95_latency_ms:.0f} ms",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_title("System Latency")

    plt.tight_layout()
    p1 = os.path.join(plots_dir, "evaluation_overview.png")
    plt.savefig(p1, bbox_inches="tight"); plt.close()
    logger.info(f"Saved: {p1}")

    # ── Figure 2: Radar chart ─────────────────────────────────────────────
    cats   = ["Accuracy","F1 Macro","Fluency","Relevance",
              "Coherence","Faithfulness","BERTScore","Hit Rate @5"]
    vals   = [result.accuracy, result.f1_macro, result.fluency_score,
              result.relevance_score, result.coherence_score,
              result.faithfulness_score, result.bertscore_f1, result.hit_rate_at_5]
    N      = len(cats)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    v_plot = vals + vals[:1]

    fig, ax = plt.subplots(figsize=(9,9), subplot_kw=dict(polar=True))
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, size=11, fontweight="bold")
    ax.set_ylim(0,1); ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], size=8, color="gray")
    ax.plot(angles, v_plot, "o-", linewidth=2, color="#2196F3")
    ax.fill(angles, v_plot, alpha=0.25, color="#2196F3")
    ax.set_title("Medical RAG Pipeline — Performance Radar",
                 size=14, fontweight="bold", pad=20)
    p2 = os.path.join(plots_dir, "radar_chart.png")
    plt.savefig(p2, bbox_inches="tight"); plt.close()
    logger.info(f"Saved: {p2}")

    # ── Figure 3: Per-class F1 ────────────────────────────────────────────
    classes = list(result.per_class_f1.keys())
    precs   = [result.per_class_precision.get(c, 0) for c in classes]
    recs    = [result.per_class_recall.get(c, 0)    for c in classes]
    f1s     = [result.per_class_f1.get(c, 0)        for c in classes]
    x = np.arange(len(classes)); w = 0.25
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x-w, precs, w, label="Precision", color="#2196F3")
    ax.bar(x,   recs,  w, label="Recall",    color="#FF9800")
    ax.bar(x+w, f1s,   w, label="F1",        color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in classes], fontsize=13, fontweight="bold")
    ax.set_ylim(0,1); ax.set_ylabel("Score")
    ax.set_title("Per-Class Classification Performance (YES / NO / MAYBE)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    p3 = os.path.join(plots_dir, "per_class_f1.png")
    plt.savefig(p3, bbox_inches="tight"); plt.close()
    logger.info(f"Saved: {p3}")

    # ── Figure 4: Latency distribution ───────────────────────────────────
    lats = [d.get("latency_ms", 0) for d in details if d.get("latency_ms")]
    if lats:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.hist(lats, bins=20, color="#607D8B", edgecolor="white", alpha=0.8)
        ax.axvline(result.avg_latency_ms, color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {result.avg_latency_ms:.0f} ms")
        ax.axvline(result.p95_latency_ms, color="orange", linestyle="--", linewidth=2,
                   label=f"P95:  {result.p95_latency_ms:.0f} ms")
        ax.set_xlabel("Response Latency (ms)"); ax.set_ylabel("Count")
        ax.set_title("Response Latency Distribution"); ax.legend(fontsize=12)
        p4 = os.path.join(plots_dir, "latency_dist.png")
        plt.savefig(p4, bbox_inches="tight"); plt.close()
        logger.info(f"Saved: {p4}")


# ═══════════════════════════════════════════════════════════════════════════════
# Mock evaluation (no model loading)
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_result(n_samples: int):
    random.seed(42)
    lats = [max(200, random.gauss(850, 200)) for _ in range(n_samples)]
    result = EvaluationResult(
        accuracy=0.68, f1_macro=0.61, f1_weighted=0.67,
        per_class_precision={"yes":0.72,"no":0.68,"maybe":0.45},
        per_class_recall=   {"yes":0.78,"no":0.65,"maybe":0.38},
        per_class_f1=       {"yes":0.75,"no":0.66,"maybe":0.41},
        bleu_1=0.42, bleu_4=0.18,
        rouge_1_f=0.38, rouge_2_f=0.21, rouge_l_f=0.35,
        bertscore_p=0.84, bertscore_r=0.82, bertscore_f1=0.83,
        fluency_score=0.74, relevance_score=0.71,
        coherence_score=0.68, faithfulness_score=0.65,
        avg_toxicity=0.02, max_toxicity=0.08, toxicity_rate=0.00, bias_rate=0.01,
        hit_rate_at_1=0.62, hit_rate_at_5=0.89, mrr=0.71, avg_retrieval_score=0.73,
        avg_latency_ms=float(np.mean(lats)),
        p95_latency_ms=float(np.percentile(lats, 95)),
        total_samples=n_samples, n_safe=n_samples, n_emergency=0,
    )
    details = [{"latency_ms": l} for l in lats]
    return result, details


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save_result(result: EvaluationResult, timestamp: int):
    path = os.path.join(RESULTS_DIR, f"eval_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    logger.info(f"Results saved → {path}")


def _print_summary(r: EvaluationResult):
    print("\n" + "="*62)
    print("  EVALUATION SUMMARY")
    print("="*62)
    print(f"  Samples evaluated : {r.total_samples}")
    print(f"\n  ── Classification ──")
    print(f"  Accuracy          : {r.accuracy:.4f}")
    print(f"  F1 Macro          : {r.f1_macro:.4f}")
    print(f"  F1 Weighted       : {r.f1_weighted:.4f}")
    for cls in LABEL_NAMES:
        print(f"    {cls.upper():5s}  P={r.per_class_precision.get(cls,0):.3f}"
              f"  R={r.per_class_recall.get(cls,0):.3f}"
              f"  F1={r.per_class_f1.get(cls,0):.3f}")
    print(f"\n  ── Text Generation ──")
    print(f"  BLEU-1            : {r.bleu_1:.4f}")
    print(f"  BLEU-4            : {r.bleu_4:.4f}")
    print(f"  ROUGE-1           : {r.rouge_1_f:.4f}")
    print(f"  ROUGE-2           : {r.rouge_2_f:.4f}")
    print(f"  ROUGE-L           : {r.rouge_l_f:.4f}")
    print(f"  BERTScore F1      : {r.bertscore_f1:.4f}")
    print(f"\n  ── RAG Quality ──")
    print(f"  Fluency           : {r.fluency_score:.4f}")
    print(f"  Relevance         : {r.relevance_score:.4f}")
    print(f"  Coherence         : {r.coherence_score:.4f}")
    print(f"  Faithfulness      : {r.faithfulness_score:.4f}")
    print(f"\n  ── Safety ──")
    print(f"  Avg Toxicity      : {r.avg_toxicity:.4f}")
    print(f"  Max Toxicity      : {r.max_toxicity:.4f}")
    print(f"  Toxicity Rate     : {r.toxicity_rate:.4f}")
    print(f"  Bias Rate         : {r.bias_rate:.4f}")
    print(f"\n  ── Retrieval ──")
    print(f"  Hit Rate @1       : {r.hit_rate_at_1:.4f}")
    print(f"  Hit Rate @5       : {r.hit_rate_at_5:.4f}")
    print(f"  MRR               : {r.mrr:.4f}")
    print(f"  Avg Retr. Score   : {r.avg_retrieval_score:.4f}")
    print(f"\n  ── System ──")
    print(f"  Avg Latency       : {r.avg_latency_ms:.0f} ms")
    print(f"  P95 Latency       : {r.p95_latency_ms:.0f} ms")
    print("="*62 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    n_samples:  int  = 100,
    use_mock:   bool = False,
    save_plots: bool = True,
) -> EvaluationResult:
    """
    Run full quantitative evaluation.

    Args:
        n_samples : Number of PubMedQA test samples to evaluate.
        use_mock  : Skip model loading; return realistic mock results.
        save_plots: Generate and save the 4 dissertation figures.

    Returns:
        EvaluationResult dataclass with every metric populated.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = int(time.time())
    plots_dir = os.path.join(RESULTS_DIR, f"plots_{timestamp}")

    logger.info("=" * 62)
    logger.info("  MEDICAL RAG PIPELINE — QUANTITATIVE EVALUATION")
    logger.info("=" * 62)

    # ── Mock mode ─────────────────────────────────────────────────────────
    if use_mock:
        result, details = _mock_result(n_samples)
        if save_plots: _generate_plots(result, plots_dir, details)
        _save_result(result, timestamp)
        _print_summary(result)
        logger.info("✅ Mock evaluation complete")
        return result

    # ── Load test data ────────────────────────────────────────────────────
    test_path = os.path.join(DATA_DIR, "test.json")
    if not os.path.exists(test_path):
        logger.warning("Test data not found — running setup…")
        from data.download_data import main as dl
        dl()
    with open(test_path) as f:
        test_data = json.load(f)

    if n_samples < len(test_data):
        random.seed(42)
        test_data = random.sample(test_data, n_samples)
    logger.info(f"Evaluating on {len(test_data)} samples")

    # ── Init pipeline ─────────────────────────────────────────────────────
    from pipeline.rag_pipeline import MedicalRAGPipeline
    pipeline = MedicalRAGPipeline()
    pipeline.initialize(lazy=False)

    # ── Run pipeline ──────────────────────────────────────────────────────
    all_results, true_labels, pred_labels = [], [], []
    answers, references, questions, contexts = [], [], [], []

    for i, rec in enumerate(test_data):
        logger.info(f"  [{i+1}/{len(test_data)}] {rec['question'][:60]}…")
        try:
            res = pipeline.query(rec["question"], return_evidence=True,
                                 return_explainability=False)
            all_results.append(res)
            pred_labels.append(res.get("predicted_label", "maybe"))
            answers.append(res.get("raw_answer", ""))
            contexts.append(res.get("context_used", ""))
        except Exception as e:
            logger.error(f"Pipeline error on sample {i}: {e}")
            all_results.append({})
            pred_labels.append("maybe")
            answers.append("")
            contexts.append("")

        true_labels.append(rec.get("final_decision", "maybe"))
        references.append(rec.get("long_answer", ""))
        questions.append(rec["question"])

    # ── Classification ────────────────────────────────────────────────────
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_recall_fscore_support)
    accuracy    = accuracy_score(true_labels, pred_labels)
    f1_macro    = f1_score(true_labels, pred_labels, labels=LABEL_NAMES,
                           average="macro", zero_division=0)
    f1_weighted = f1_score(true_labels, pred_labels, labels=LABEL_NAMES,
                           average="weighted", zero_division=0)
    prec, rec, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=LABEL_NAMES, zero_division=0)
    per_class_p = dict(zip(LABEL_NAMES, [float(x) for x in prec]))
    per_class_r = dict(zip(LABEL_NAMES, [float(x) for x in rec]))
    per_class_f = dict(zip(LABEL_NAMES, [float(x) for x in f1]))
    logger.info(f"  Accuracy: {accuracy:.4f}  F1-macro: {f1_macro:.4f}")

    # ── Text generation ───────────────────────────────────────────────────
    logger.info("  Computing BLEU & ROUGE…")
    bleu  = _compute_bleu(answers, references)
    rouge = _compute_rouge(answers, references)
    logger.info("  Computing BERTScore…")
    bscore = _compute_bertscore(answers, references)

    # ── RAG quality ───────────────────────────────────────────────────────
    logger.info("  Computing fluency, relevance, coherence, faithfulness…")
    fluency = _compute_fluency(answers)
    try:
        from embeddings.pubmedbert_embedder import PubMedBERTEmbedder
        emb        = PubMedBERTEmbedder()
        relevance  = _compute_relevance(questions, answers, emb)
        coherence  = _compute_coherence(answers, emb)
    except Exception as e:
        logger.warning(f"Embedder-based metrics failed: {e}")
        relevance = coherence = 0.0
    faithfulness = _compute_faithfulness(answers, contexts)

    # ── Safety ────────────────────────────────────────────────────────────
    logger.info("  Computing toxicity & bias…")
    tox = _compute_toxicity([r.get("answer", "") for r in all_results])
    n_bias = sum(
        1 for r in all_results
        if any("bias" in str(w).lower() for w in r.get("safety_warnings", []))
    )
    bias_rate = n_bias / max(len(all_results), 1)

    # ── Retrieval ─────────────────────────────────────────────────────────
    retr = _compute_retrieval_metrics(all_results)

    # ── System ────────────────────────────────────────────────────────────
    lats       = [r.get("latency_ms", 0) for r in all_results if r.get("latency_ms")]
    avg_lat    = float(np.mean(lats))    if lats else 0.0
    p95_lat    = float(np.percentile(lats, 95)) if lats else 0.0
    n_safe     = sum(1 for r in all_results if r.get("safe", True))
    n_emergency= sum(1 for r in all_results if r.get("emergency", False))

    # ── Assemble ──────────────────────────────────────────────────────────
    result = EvaluationResult(
        accuracy=float(accuracy), f1_macro=float(f1_macro),
        f1_weighted=float(f1_weighted),
        per_class_precision=per_class_p,
        per_class_recall=per_class_r,
        per_class_f1=per_class_f,
        bleu_1=bleu["bleu_1"], bleu_4=bleu["bleu_4"],
        rouge_1_f=rouge["rouge_1_f"], rouge_2_f=rouge["rouge_2_f"],
        rouge_l_f=rouge["rouge_l_f"],
        bertscore_p=bscore["bertscore_p"], bertscore_r=bscore["bertscore_r"],
        bertscore_f1=bscore["bertscore_f1"],
        fluency_score=float(fluency), relevance_score=float(relevance),
        coherence_score=float(coherence), faithfulness_score=float(faithfulness),
        avg_toxicity=tox["avg_toxicity"], max_toxicity=tox["max_toxicity"],
        toxicity_rate=tox["toxicity_rate"], bias_rate=float(bias_rate),
        hit_rate_at_1=retr["hit_rate_at_1"], hit_rate_at_5=retr["hit_rate_at_5"],
        mrr=retr["mrr"], avg_retrieval_score=retr["avg_retrieval_score"],
        avg_latency_ms=avg_lat, p95_latency_ms=p95_lat,
        total_samples=len(test_data), n_safe=n_safe, n_emergency=n_emergency,
    )

    details = [{"latency_ms": r.get("latency_ms", 0)} for r in all_results]
    if save_plots: _generate_plots(result, plots_dir, details)
    _save_result(result, timestamp)
    _print_summary(result)

    logger.info(f"✅ Evaluation complete! Results → {RESULTS_DIR}")
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--mock", action="store_true")
    a = p.parse_args()
    run_evaluation(n_samples=a.n_samples, use_mock=a.mock)
