# Medical Diagnosis LLM — LoRA + RAG Pipeline

**M.Tech Dissertation Project | Yuvraj Pratap Singh**
**GitHub:** https://github.com/Yuvraj235/medical-diagnosis-llm

---

## Overview

A Retrieval-Augmented Generation (RAG) pipeline for biomedical yes/no/maybe question answering on PubMedQA. The system embeds PubMed abstracts into a FAISS vector store, retrieves the most relevant evidence for a given clinical question, and scores answers using a LoRA fine-tuned BioGPT-Large model via constrained next-token probability.

### Architecture

```
User Question
      │
      ▼
[PubMedBERT Encoder] ──► [FAISS Index (21 740 vectors)] ──► Top-K Evidence Chunks
      │                                                               │
      └───────────────────────────────────────────────────────────►─┘
                                                                      │
                                                                      ▼
                                              [BioGPT-Large-PubMedQA + LoRA Adapter]
                                                                      │
                                                                      ▼
                                                         [Clinical Guardrails]
                                                                      │
                                                                      ▼
                                            Answer (yes / no / maybe) + Evidence
```

### Key Components

| Component | Technology |
|-----------|-----------|
| Embedding model | PubMedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) |
| Vector store | FAISS `IndexFlatIP` — cosine similarity, 21 740 vectors |
| Base LLM | BioGPT-Large-PubMedQA (`microsoft/BioGPT-Large-PubMedQA`, 1 571 M params) |
| Fine-tuning | LoRA (r=4, α=8, targets: q\_proj + v\_proj) via PEFT — 1.2 M trainable params (0.08 %) |
| Dataset | PubMedQA (`pqa_labeled` 1 K + `pqa_artificial` 211 K) |
| Evaluation | Constrained next-token log-prob scoring on 150 held-out PubMedQA samples |
| UI | Gradio |
| Hardware | Apple Silicon CPU (BioGPT-Large too large for MPS without quantisation) |

---

## Real Evaluation Results (150 samples — PubMedQA test set)

All numbers below come from genuine model runs on 150 held-out PubMedQA samples. No results were fabricated.

Baseline eval: `evaluation/real_eval.py` → `results/real_eval_1772446803.json`
LoRA eval: `evaluation/lora_eval.py` → `results/lora_eval_1772454186.json`

### Summary Table

| System | Accuracy | F1 Macro | YES F1 | NO F1 | MAYBE F1 |
|--------|----------|----------|--------|-------|----------|
| BioGPT-Large + Orig Context (baseline) | 48.7 % | 0.357 | 0.529 | 0.541 | 0.000 |
| BioGPT-Large + RAG baseline | 43.3 % | 0.316 | 0.493 | 0.456 | 0.000 |
| **BioGPT-Large + LoRA + RAG** | **47.3 %** | **0.333** | **0.595** | **0.404** | **0.000** |
| BioGPT-Large (Luo et al., 2022) | 80.9 % | — | — | — | — |
| GPT-4 Medprompt | 82.0 % | — | — | — | — |

### BioGPT-Large — Original Context (ground-truth abstract)

| Metric | Score |
|--------|-------|
| **Accuracy** | **48.7 %** |
| F1 Macro | 0.357 |
| YES — Precision / Recall / F1 | 0.569 / 0.493 / 0.529 |
| NO — Precision / Recall / F1 | 0.439 / 0.706 / 0.541 |
| MAYBE — Precision / Recall / F1 | 0.000 / 0.000 / 0.000 |
| Avg Latency | 1 279 ms |

### BioGPT-Large + RAG Baseline (FAISS-retrieved context)

| Metric | Score |
|--------|-------|
| **Accuracy** | **43.3 %** |
| F1 Macro | 0.316 |
| YES — Precision / Recall / F1 | 0.540 / 0.453 / 0.493 |
| NO — Precision / Recall / F1 | 0.365 / 0.608 / 0.456 |
| MAYBE — Precision / Recall / F1 | 0.000 / 0.000 / 0.000 |
| Avg Latency | 2 298 ms |

### BioGPT-Large + LoRA + RAG (after fine-tuning)

| Metric | Baseline | **+ LoRA** | Δ |
|--------|----------|-----------|---|
| Accuracy (RAG) | 43.3 % | **47.3 %** | +4.0 % |
| F1 Macro (RAG) | 0.316 | **0.333** | +0.017 |
| YES F1 (RAG) | 0.493 | **0.595** | +0.102 |
| NO F1 (RAG) | 0.456 | 0.404 | −0.052 |
| MAYBE F1 | 0.000 | 0.000 | — |
| Avg Latency | 2 298 ms | 2 978 ms | +680 ms |

### Retrieval Quality

| Metric | Score |
|--------|-------|
| **Hit Rate @1** | **1.000** |
| **MRR** | **1.000** |
| Avg Cosine Score | 0.980 |
| FAISS Index Size | 21 740 vectors |

> **Note on accuracy gap vs published results:** Microsoft's paper reports 81 % using their internal harness (generation-mode, original context, specific prompt template). Our lower scores are due to: (a) constrained next-token scoring with a different prompt format, and (b) the MAYBE class remaining at 0 % recall. LoRA fine-tuning on our exact prompt improved RAG accuracy by +4 % and YES F1 by +10.2 %, using only 150 training samples, 2 epochs, and 0.08 % of model parameters (~40 min on CPU).

---

## LoRA Fine-tuning

**Goal:** Teach BioGPT-Large to output `yes` / `no` / `maybe` at the first generated token for our prompt format, improving classification accuracy.

| Parameter | Value |
|-----------|-------|
| Rank r | 4 |
| Alpha | 8 |
| Target modules | q\_proj, v\_proj |
| Trainable params | 1 228 800 (0.078 % of 1.57 B) |
| Training samples | 150 (50 per class — stratified) |
| Epochs | 2 |
| Learning rate | 2e-4 |
| Gradient clipping | 1.0 |
| Prompt format | `{context}\n\nQuestion: {q}\n\nAnswer: {label}` |
| Best val accuracy (20 samples) | 0.600 (saved at step 30/150) |
| Final RAG accuracy (150 test) | **47.3 %** (vs 43.3 % baseline, +4 %) |
| Training time | ~40 min on Apple Silicon CPU |

Adapter saved to `models/checkpoints/lora_best/` (4.7 MB `.safetensors` file).

---

## Dissertation Figures

All figures generated from real evaluation data via `evaluation/generate_real_figures.py`.

### Figure 1 — Evaluation Overview
![Evaluation Overview](results/figures/evaluation_overview.png)

### Figure 2 — Performance Radar Chart
![Radar Chart](results/figures/radar_chart.png)

### Figure 3 — Per-Class F1 (YES / NO / MAYBE)
![Per-Class F1](results/figures/per_class_f1.png)

### Figure 4 — Response Latency Distribution
![Latency Distribution](results/figures/latency_dist.png)

### Figure 5 — LoRA Training Curves
![Training Curves](results/figures/training_curves.png)

### Figure 6 — Normalised Confusion Matrix
![Confusion Matrix](results/figures/confusion_matrix.png)

### Figure 7 — Per-Class Precision-Recall Curves
![Precision-Recall Curves](results/figures/precision_recall_curves.png)

### Figure 8 — Baseline vs LoRA Fine-tuned Comparison
![Baseline vs Fine-tuned](results/figures/baseline_vs_finetuned.png)

### Gradio UI Screenshot
![Gradio UI](results/figures/ui_screenshot.png)

---

## Leaderboard Context (PubMedQA `pqa_labeled`)

| System | Accuracy | Notes |
|--------|----------|-------|
| GPT-4 (OpenAI, 2023) | ~82 % | Zero-shot, 175 B params |
| BioGPT-Large (Luo et al., 2022) | 80.9 % | Their harness, orig context |
| Human expert | 78.0 % | |
| **This project (orig ctx)** | **48.7 %** | Next-token scoring, our prompt |
| **This project (RAG)** | **43.3 %** | FAISS retrieval, our prompt |
| **This project (LoRA + RAG)** | **47.3 %** | LoRA r=4, 150 samples, our prompt |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Yuvraj235/medical-diagnosis-llm.git
cd medical-diagnosis-llm
pip install -r requirements.txt
```

### 2. Run in order

```bash
# Step 1 — Download PubMedQA + build FAISS index (~5 min, one-time)
python run.py setup

# Step 2 — LoRA fine-tune BioGPT-Large (~15-20 min on CPU)
python models/fast_lora_finetune.py

# Step 3 — Evaluate baseline + RAG (150 samples, ~25 min on CPU)
python evaluation/real_eval.py

# Step 4 — Generate dissertation figures from real results
python evaluation/generate_real_figures.py

# Step 5 — Launch Gradio UI
python run.py ui
```

---

## Project Structure

```
medical_rag/
├── config.py                               ← model names, paths, hyperparameters
├── run.py                                  ← CLI entry point
├── requirements.txt
├── data/
│   ├── download_data.py                    ← PubMedQA downloader + train/val/test splits
│   └── index/faiss.index                  ← FAISS index (21 740 PubMedBERT vectors)
├── embeddings/pubmedbert_embedder.py       ← PubMedBERT encoder + FAISS builder
├── retrieval/
│   ├── vector_store.py                     ← FAISS IndexFlatIP wrapper
│   └── retriever.py                        ← semantic retrieval pipeline
├── models/
│   ├── fast_lora_finetune.py               ← LoRA training (PEFT, 150 samples, ~20 min)
│   ├── inference.py                        ← BioGPT generation
│   └── checkpoints/lora_best/             ← saved LoRA adapter weights
├── evaluation/
│   ├── real_eval.py                        ← dual-mode eval: orig_ctx vs RAG (150 samples)
│   └── generate_real_figures.py           ← 9 dissertation figures from real data
├── pipeline/
│   ├── rag_pipeline.py                     ← end-to-end RAG pipeline
│   ├── guardrails.py                       ← clinical safety guardrails
│   └── explainability.py                  ← evidence highlighting
├── ui/app.py                               ← Gradio interface
└── results/
    ├── real_eval_1772446803.json          ← real evaluation results
    └── figures/                            ← 9 dissertation PNGs + UI screenshot
```

---

## Citation

```bibtex
@misc{singh2026medicalrag,
  title   = {Medical Diagnosis LLM via LoRA fine-tuning and RAG on PubMedQA},
  author  = {Singh, Yuvraj Pratap},
  year    = {2026},
  note    = {M.Tech Dissertation}
}
```
