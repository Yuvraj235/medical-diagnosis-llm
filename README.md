# Medical Diagnosis LLM via Low-Rank Adaptation and Retrieval-Augmentation

A Retrieval-Augmented Generation (RAG) pipeline for medical question answering using PubMedQA data, PubMedBERT embeddings, and a LoRA fine-tuned BioMistral-7B.

## Architecture

```
Document -> Chunker -> Chunks -> PubMedBERT Embeddings -> ChromaDB Vector Store
Query -> PubMedBERT -> Vector DB -> Top-K Chunks -> Prompt Composer -> BioMistral-7B (LoRA) -> Answer
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Load and preprocess data
python src/data_loader.py

# 2. Build vector database index
python src/vector_store.py

# 3. Fine-tune BioMistral-7B with LoRA (takes hours)
python src/lora_finetune.py

# 4. Run end-to-end pipeline
python src/pipeline.py

# 5. Evaluate on PubMedQA test set
python src/evaluate.py
```

Or run the complete notebook: `jupyter notebook notebooks/medical_rag_pipeline.ipynb`

## Project Structure

```
medical-diagnosis-llm/
├── notebooks/
│   └── medical_rag_pipeline.ipynb   # Complete demo + evaluation
├── src/
│   ├── data_loader.py               # PubMedQA data loading & preprocessing
│   ├── embeddings.py                # PubMedBERT embedding generation
│   ├── vector_store.py              # ChromaDB vector database
│   ├── retriever.py                 # Semantic retrieval with MMR
│   ├── lora_finetune.py             # QLoRA fine-tuning BioMistral-7B
│   ├── generator.py                 # LLM answer generation
│   ├── guardrails.py                # Safety & clinical guardrails
│   ├── pipeline.py                  # End-to-end RAG pipeline
│   └── evaluate.py                  # Evaluation metrics & plots
├── configs/
│   └── config.yaml                  # All hyperparameters
├── requirements.txt
└── README.md
```

## Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Embeddings | PubMedBERT | Domain-specific biomedical text embeddings |
| Vector Store | ChromaDB | Efficient semantic similarity search |
| Retriever | MMR Reranking | Diverse, relevant evidence selection |
| LLM | BioMistral-7B | Medical domain language model |
| Fine-tuning | QLoRA (r=16) | Efficient adaptation with 4-bit quantization |
| Guardrails | Custom | Disclaimers, hallucination check, scope filter |

## Dataset

- **PubMedQA** (qiaojin/PubMedQA on HuggingFace)
  - pqa_labeled: 1,000 expert-annotated examples (train/val/test)
  - pqa_artificial: 211,269 machine-generated examples (training + knowledge base)
  - Task: Answer yes/no/maybe to biomedical questions with evidence

## Evaluation Metrics

- **Retrieval**: Hit Rate, MRR, Avg Similarity Score
- **Generation**: Accuracy, F1 (macro/weighted), Per-class Precision/Recall
- **Safety**: Safety pass rate, hallucination detection rate, warning distribution
