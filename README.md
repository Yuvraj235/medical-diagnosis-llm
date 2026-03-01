# Medical Diagnosis LLM — LoRA + RAG Pipeline

M.Tech Dissertation Project
**Author:** Yuvraj Pratap Singh

---

## Overview

A Retrieval-Augmented Generation (RAG) pipeline for biomedical question answering.
The system retrieves relevant PubMed abstracts, grounds a LoRA fine-tuned BioGPT model on the evidence, and produces clinically safe, explainable answers with comprehensive quantitative evaluation.

### Architecture

```
User Question
      │
      ▼
[PubMedBERT Encoder] ──► [FAISS Index] ──► Top-K Evidence Chunks
      │                                            │
      └──────────────────────────────────────────►─┘
                                                   │
                                                   ▼
                                    [BioGPT-Large + LoRA Adapter]
                                                   │
                                                   ▼
                                      [Clinical Guardrails]
                                                   │
                                                   ▼
                              Answer + Decision (yes/no/maybe) + Evidence
```

### Key Components

| Component | Technology |
|-----------|-----------|
| Embedding model | PubMedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) |
| Vector store | FAISS (IndexFlatIP — cosine similarity) |
| Base LLM | BioGPT-Large-PubMedQA (`microsoft/BioGPT-Large-PubMedQA`) |
| Fine-tuning | LoRA (r=16, α=32) via PEFT |
| Dataset | PubMedQA (pqa_labeled 1K + pqa_artificial 211K) |
| UI | Gradio |
| Hardware | Apple Silicon (MPS) / CUDA / CPU |

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/medical_rag.git
cd medical_rag
pip install -r requirements.txt
pip install sacremoses sentencepiece      # extra tokenizers
```

### 2. Download data and build FAISS index

```bash
python run.py setup
```

This will:
- Download PubMedQA (pqa_labeled + pqa_artificial) from HuggingFace
- Create train / val / test splits in `data/pubmedqa/`
- Chunk and encode all abstracts with PubMedBERT
- Build a FAISS index saved to `data/index/`

---

## Run Order

```bash
# 1. One-time setup (data + index)
python run.py setup

# 2. Fine-tune BioGPT with LoRA  (~20-40 min on MPS/GPU)
python run.py finetune

# 3. Evaluate on 100 held-out samples
python run.py evaluate --n-samples 100

#    Fast mock evaluation (no model required)
python run.py evaluate --n-samples 100 --mock

# 4. Launch Gradio UI
python run.py ui

# Ad-hoc query
python run.py query "Does aspirin reduce the risk of colorectal cancer?"
```

---

## Evaluation Metrics

| Category | Metrics |
|----------|---------|
| Classification | Accuracy, Macro-F1, Weighted-F1, Per-class F1 (yes/no/maybe) |
| Text Quality | BLEU-1, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L |
| Semantic | BERTScore (Precision / Recall / F1) |
| Linguistic | Fluency, Relevance, Coherence, Faithfulness |
| Safety | Avg Toxicity, Max Toxicity, Toxicity Rate, Bias Rate |
| Retrieval | Hit Rate @1, Hit Rate @5, MRR, Avg Retrieval Score |
| Latency | Avg Latency (ms), P95 Latency (ms) |

### Output files

```
results/
├── eval_YYYYMMDD_HHMMSS.json          # Full metrics JSON
└── plots_YYYYMMDD_HHMMSS/
    ├── evaluation_overview.png         # Bar chart of all key metrics
    ├── radar_chart.png                 # Radar / spider chart
    ├── per_class_f1.png               # Per-class F1 bar chart
    └── latency_dist.png               # Latency distribution histogram
```

---

## Project Structure

```
medical_rag/
├── config.py                          # Central hyperparameters
├── run.py                             # CLI entry point
├── requirements.txt
├── data/
│   ├── download_data.py               # PubMedQA downloader + splits
│   └── pubmedqa/                      # Downloaded data (git-ignored)
├── embeddings/
│   └── pubmedbert_embedder.py         # Encoder + corpus builder
├── retrieval/
│   ├── vector_store.py                # FAISS index wrapper
│   └── retriever.py                   # Semantic retrieval pipeline
├── models/
│   ├── lora_finetune.py               # LoRA fine-tuning
│   ├── inference.py                   # BioGPT inference
│   └── checkpoints/                   # Saved LoRA adapter (git-ignored)
├── pipeline/
│   ├── rag_pipeline.py                # End-to-end RAG pipeline
│   ├── guardrails.py                  # Clinical safety guardrails
│   └── explainability.py             # Evidence highlighting
├── evaluation/
│   └── run_evaluation.py              # Full quantitative eval suite
├── ui/
│   └── app.py                         # Gradio web interface
└── results/                           # Eval outputs (git-ignored)
```

---

## Citation

```bibtex
@misc{singh2025medicalrag,
  title   = {Medical Diagnosis LLM via LoRA + RAG},
  author  = {Singh, Yuvraj Pratap},
  year    = {2025},
  note    = {M.Tech Dissertation}
}
```
